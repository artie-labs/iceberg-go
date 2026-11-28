// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

package table_test

import (
	"context"
	"fmt"
	"math"
	"path/filepath"
	"slices"
	"testing"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/decimal128"
	"github.com/apache/arrow-go/v18/arrow/memory"
	"github.com/apache/iceberg-go"
	iceio "github.com/apache/iceberg-go/io"
	"github.com/apache/iceberg-go/table"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// numericRow is one row of the numeric filter table: dec is decimal(7, 2)'s unscaled value and big decimal(38, 10)'s
// text.
type numericRow struct {
	id  int64
	dec *int64
	big *string
	f64 *float64
	f32 *float32
}

func p[T any](v T) *T { return &v }

// numericFiles are the numeric filter table's data files, one per append, so metrics pruning sees distinct bounds.
var numericFiles = [][]numericRow{
	{
		{id: 1, dec: p[int64](1050), big: p("12345678901234567890.1234567890"), f64: p(1.5), f32: p[float32](1.5)},
		{id: 2, dec: nil, big: nil, f64: nil, f32: nil},
		{id: 3, dec: p[int64](-525), big: p("-1.0000000001"), f64: p(math.NaN()), f32: p(float32(math.NaN()))},
		{id: 4, dec: p[int64](0), big: p("0"), f64: p(math.Copysign(0, -1)), f32: p(float32(math.Copysign(0, -1)))},
	},
	{
		{id: 5, dec: p[int64](99999), big: p("99999999999999999999.9999999999"), f64: p(math.Inf(1)), f32: p(float32(math.Inf(1)))},
		{id: 6, dec: p[int64](1050), big: p("12345678901234567890.1234567890"), f64: p(math.Inf(-1)), f32: p(float32(math.Inf(-1)))},
		{id: 7, dec: p[int64](1), big: p("0.0000000001"), f64: p(0.0), f32: p[float32](0)},
	},
	// A file of only NaN and NULL floats, whose bounds are absent.
	{
		{id: 8, dec: p[int64](200), big: p("2"), f64: p(math.NaN()), f32: p(float32(math.NaN()))},
		{id: 9, dec: nil, big: nil, f64: nil, f32: nil},
	},
}

func numericFilterSchema() *iceberg.Schema {
	return iceberg.NewSchema(0,
		iceberg.NestedField{ID: 1, Name: "id", Type: iceberg.PrimitiveTypes.Int64, Required: true},
		iceberg.NestedField{ID: 2, Name: "dec", Type: iceberg.DecimalTypeOf(7, 2), Required: false},
		iceberg.NestedField{ID: 3, Name: "big", Type: iceberg.DecimalTypeOf(38, 10), Required: false},
		iceberg.NestedField{ID: 4, Name: "f64", Type: iceberg.PrimitiveTypes.Float64, Required: false},
		iceberg.NestedField{ID: 5, Name: "f32", Type: iceberg.PrimitiveTypes.Float32, Required: false},
	)
}

func numericFilterTable(t *testing.T, mode, version string) *table.Table {
	t.Helper()
	location := filepath.ToSlash(t.TempDir())
	props := iceberg.Properties{table.PropertyFormatVersion: version, table.WriteDeleteModeKey: mode}
	meta, err := table.NewMetadata(numericFilterSchema(), iceberg.UnpartitionedSpec, table.UnsortedSortOrder, location, props)
	require.NoError(t, err)
	metaLoc := location + "/metadata/v1.metadata.json"
	fsF := func(context.Context) (iceio.IO, error) { return iceio.LocalFS{}, nil }
	cat := &concurrentTestCatalog{metadata: meta, location: metaLoc, fsF: fsF}
	tbl := table.New(table.Identifier{"db", "numeric_filter"}, meta, metaLoc, fsF, cat)

	arrowSchema := arrow.NewSchema([]arrow.Field{
		{Name: "id", Type: arrow.PrimitiveTypes.Int64},
		{Name: "dec", Type: &arrow.Decimal128Type{Precision: 7, Scale: 2}, Nullable: true},
		{Name: "big", Type: &arrow.Decimal128Type{Precision: 38, Scale: 10}, Nullable: true},
		{Name: "f64", Type: arrow.PrimitiveTypes.Float64, Nullable: true},
		{Name: "f32", Type: arrow.PrimitiveTypes.Float32, Nullable: true},
	}, nil)
	for _, rows := range numericFiles {
		b := array.NewRecordBuilder(memory.DefaultAllocator, arrowSchema)
		for _, r := range rows {
			b.Field(0).(*array.Int64Builder).Append(r.id)
			if r.dec == nil {
				b.Field(1).AppendNull()
			} else {
				b.Field(1).(*array.Decimal128Builder).Append(decimal128.FromI64(*r.dec))
			}
			if r.big == nil {
				b.Field(2).AppendNull()
			} else {
				n, err := decimal128.FromString(*r.big, 38, 10)
				require.NoError(t, err)
				b.Field(2).(*array.Decimal128Builder).Append(n)
			}
			if r.f64 == nil {
				b.Field(3).AppendNull()
			} else {
				b.Field(3).(*array.Float64Builder).Append(*r.f64)
			}
			if r.f32 == nil {
				b.Field(4).AppendNull()
			} else {
				b.Field(4).(*array.Float32Builder).Append(*r.f32)
			}
		}
		rec := b.NewRecordBatch()
		b.Release()
		rdr, err := array.NewRecordReader(arrowSchema, []arrow.RecordBatch{rec})
		require.NoError(t, err)
		tbl, err = tbl.Append(context.Background(), rdr, nil)
		require.NoError(t, err)
		rdr.Release()
		rec.Release()
	}
	return tbl
}

func allNumericIDs() []int64 {
	var ids []int64
	for _, rows := range numericFiles {
		for _, r := range rows {
			ids = append(ids, r.id)
		}
	}
	return ids
}

func dec2(unscaled int64) iceberg.Decimal {
	return iceberg.Decimal{Val: decimal128.FromI64(unscaled), Scale: 2}
}

func big10(text string) iceberg.Decimal {
	n, err := decimal128.FromString(text, 38, 10)
	if err != nil {
		panic(err)
	}
	return iceberg.Decimal{Val: n, Scale: 10}
}

// TestDecimalFloatFilters checks that scans and deletes, in every delete mode and format version, select the rows each
// filter matches on decimal and floating-point columns, including NULLs, NaN, signed zeros and infinities. arrow-go
// v18.8 panicked testing a decimal for NULL, and its is_nan read NULL slots as values.
func TestDecimalFloatFilters(t *testing.T) {
	dec, big, f64, f32 := iceberg.Reference("dec"), iceberg.Reference("big"), iceberg.Reference("f64"), iceberg.Reference("f32")
	cases := []struct {
		filter iceberg.BooleanExpression
		want   []int64
	}{
		{iceberg.EqualTo(dec, dec2(1050)), []int64{1, 6}},
		{iceberg.LessThan(dec, dec2(0)), []int64{3}},
		{iceberg.LessThanEqual(dec, dec2(0)), []int64{3, 4}},
		{iceberg.GreaterThan(dec, dec2(1050)), []int64{5}},
		{iceberg.GreaterThanEqual(dec, dec2(1050)), []int64{1, 5, 6}},
		{iceberg.NewAnd(iceberg.NotNull(dec), iceberg.NotEqualTo(dec, dec2(1050))), []int64{3, 4, 5, 7, 8}},
		{iceberg.IsIn(dec, dec2(1050), dec2(-525)), []int64{1, 3, 6}},
		{iceberg.NewAnd(iceberg.NotNull(dec), iceberg.NotIn(dec, dec2(1050), dec2(-525))), []int64{4, 5, 7, 8}},
		{iceberg.IsNull(dec), []int64{2, 9}},
		{iceberg.NotNull(dec), []int64{1, 3, 4, 5, 6, 7, 8}},
		{iceberg.EqualTo(big, big10("12345678901234567890.1234567890")), []int64{1, 6}},
		{iceberg.GreaterThan(big, big10("12345678901234567890.1234567890")), []int64{5}},
		{iceberg.LessThan(big, big10("0")), []int64{3}},
		{iceberg.IsIn(big, big10("0.0000000001"), big10("-1.0000000001")), []int64{3, 7}},
		{iceberg.NewAnd(iceberg.NotNull(big), iceberg.NotIn(big, big10("0.0000000001"), big10("-1.0000000001"))), []int64{1, 4, 5, 6, 8}},
		{iceberg.EqualTo(f64, 0.0), []int64{4, 7}},
		{iceberg.EqualTo(f64, math.Copysign(0, -1)), []int64{4, 7}},
		{iceberg.LessThan(f64, 0.0), []int64{6}},
		{iceberg.LessThanEqual(f64, 0.0), []int64{4, 6, 7}},
		{iceberg.GreaterThan(f64, 1.0), []int64{1, 5}},
		{iceberg.GreaterThanEqual(f64, math.Inf(1)), []int64{5}},
		{iceberg.LessThanEqual(f64, math.Inf(-1)), []int64{6}},
		{iceberg.IsNaN(f64), []int64{3, 8}},
		{iceberg.NotNaN(f64), []int64{1, 2, 4, 5, 6, 7, 9}},
		{iceberg.NewAnd(iceberg.NotNull(f64), iceberg.NotNaN(f64)), []int64{1, 4, 5, 6, 7}},
		{iceberg.IsIn(f64, 1.5, math.Inf(1)), []int64{1, 5}},
		{iceberg.NewAnd(iceberg.NotNull(f64), iceberg.NotIn(f64, 1.5, math.Inf(1))), []int64{3, 4, 6, 7, 8}},
		{iceberg.NewAnd(iceberg.NotNull(f64), iceberg.IsNaN(f64)), []int64{3, 8}},
		{iceberg.NewNot(iceberg.IsNaN(f64)), []int64{1, 2, 4, 5, 6, 7, 9}},
		{iceberg.NewNot(iceberg.NotNaN(f64)), []int64{3, 8}},
		{iceberg.NewAnd(iceberg.NotNull(f64), iceberg.NotEqualTo(f64, 1.5)), []int64{3, 4, 5, 6, 7, 8}},
		{iceberg.NewOr(iceberg.GreaterThan(f64, 1.0), iceberg.IsNaN(f64)), []int64{1, 3, 5, 8}},
		{iceberg.EqualTo(f32, float32(1.5)), []int64{1}},
		{iceberg.LessThan(f32, float32(0)), []int64{6}},
		{iceberg.IsNaN(f32), []int64{3, 8}},
		{iceberg.GreaterThan(f32, float32(1)), []int64{1, 5}},
		{iceberg.IsIn(f32, float32(1.5), float32(math.Inf(-1))), []int64{1, 6}},
		{iceberg.NotNaN(f32), []int64{1, 2, 4, 5, 6, 7, 9}},
		{iceberg.IsNull(big), []int64{2, 9}},
		{iceberg.NewNot(iceberg.IsNull(dec)), []int64{1, 3, 4, 5, 6, 7, 8}},
		{iceberg.NewOr(iceberg.IsNull(dec), iceberg.LessThan(dec, dec2(0))), []int64{2, 3, 9}},
	}

	for _, c := range cases {
		t.Run("scan/"+c.filter.String(), func(t *testing.T) {
			tbl := numericFilterTable(t, table.WriteModeMergeOnRead, "2")
			assert.Equal(t, c.want, idsMatching(t, tbl, c.filter))
		})
		for _, mode := range []string{table.WriteModeCopyOnWrite, table.WriteModeMergeOnRead} {
			for _, version := range []string{"2", "3"} {
				t.Run(fmt.Sprintf("delete/%s/v%s/%s", mode, version, c.filter), func(t *testing.T) {
					tbl := numericFilterTable(t, mode, version)
					tbl, err := tbl.Delete(context.Background(), c.filter, nil)
					require.NoError(t, err)
					var want []int64
					for _, id := range allNumericIDs() {
						if !slices.Contains(c.want, id) {
							want = append(want, id)
						}
					}
					assert.Equal(t, want, idsInTable(t, tbl))
				})
			}
		}
	}
}

// TestDeleteKeepsNaNRowsOfMatchingFile covers a file whose every non-NaN value matches: bounds leave NaN out, so a
// metrics evaluator that ignored NaN counts would drop the whole file, NaN rows included, without filtering its rows.
func TestDeleteKeepsNaNRowsOfMatchingFile(t *testing.T) {
	f64 := iceberg.Reference("f64")
	for _, filter := range []iceberg.BooleanExpression{
		iceberg.LessThan(f64, 10.0),
		iceberg.LessThanEqual(f64, 2.0),
		iceberg.GreaterThan(f64, 0.0),
		iceberg.NewAnd(iceberg.NotNull(f64), iceberg.NotEqualTo(f64, 5.0)),
		iceberg.NewAnd(iceberg.NotNull(f64), iceberg.NotIn(f64, 5.0, 6.0)),
	} {
		for _, mode := range []string{table.WriteModeCopyOnWrite, table.WriteModeMergeOnRead} {
			t.Run(fmt.Sprintf("%s/%s", mode, filter), func(t *testing.T) {
				location := filepath.ToSlash(t.TempDir())
				schema := iceberg.NewSchema(0,
					iceberg.NestedField{ID: 1, Name: "id", Type: iceberg.PrimitiveTypes.Int64, Required: true},
					iceberg.NestedField{ID: 2, Name: "f64", Type: iceberg.PrimitiveTypes.Float64, Required: false},
				)
				props := iceberg.Properties{table.PropertyFormatVersion: "2", table.WriteDeleteModeKey: mode}
				meta, err := table.NewMetadata(schema, iceberg.UnpartitionedSpec, table.UnsortedSortOrder, location, props)
				require.NoError(t, err)
				metaLoc := location + "/metadata/v1.metadata.json"
				fsF := func(context.Context) (iceio.IO, error) { return iceio.LocalFS{}, nil }
				cat := &concurrentTestCatalog{metadata: meta, location: metaLoc, fsF: fsF}
				tbl := table.New(table.Identifier{"db", "nan_rows"}, meta, metaLoc, fsF, cat)

				arrowSchema := arrow.NewSchema([]arrow.Field{
					{Name: "id", Type: arrow.PrimitiveTypes.Int64},
					{Name: "f64", Type: arrow.PrimitiveTypes.Float64, Nullable: true},
				}, nil)
				b := array.NewRecordBuilder(memory.DefaultAllocator, arrowSchema)
				defer b.Release()
				b.Field(0).(*array.Int64Builder).AppendValues([]int64{1, 2, 3}, nil)
				b.Field(1).(*array.Float64Builder).AppendValues([]float64{1, 2, math.NaN()}, nil)
				rec := b.NewRecordBatch()
				defer rec.Release()
				rdr, err := array.NewRecordReader(arrowSchema, []arrow.RecordBatch{rec})
				require.NoError(t, err)
				defer rdr.Release()
				tbl, err = tbl.Append(context.Background(), rdr, nil)
				require.NoError(t, err)

				matched := idsMatching(t, tbl, filter)
				tbl, err = tbl.Delete(context.Background(), filter, nil)
				require.NoError(t, err)
				var want []int64
				for _, id := range []int64{1, 2, 3} {
					if !slices.Contains(matched, id) {
						want = append(want, id)
					}
				}
				assert.Equal(t, want, idsInTable(t, tbl))
				// Comparisons are false on NaN, except <> and NOT IN.
				switch filter.Op() {
				case iceberg.OpAnd:
					assert.Contains(t, matched, int64(3))
				default:
					assert.NotContains(t, matched, int64(3))
				}
			})
		}
	}
}
