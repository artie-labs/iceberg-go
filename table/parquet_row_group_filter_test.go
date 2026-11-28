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
	"bytes"
	"fmt"
	"strconv"
	"testing"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/decimal128"
	"github.com/apache/arrow-go/v18/arrow/memory"
	"github.com/apache/arrow-go/v18/parquet"
	"github.com/apache/arrow-go/v18/parquet/compress"
	"github.com/apache/arrow-go/v18/parquet/file"
	"github.com/apache/arrow-go/v18/parquet/pqarrow"
	"github.com/apache/iceberg-go"
	"github.com/apache/iceberg-go/table"
	"github.com/stretchr/testify/require"
)

// writeRowGroupFilterFile writes column, with Parquet field ID 1 and the given
// writer properties, to a Parquet file of rowGroupRows-row row groups.
func writeRowGroupFilterFile(t *testing.T, column arrow.Array, rowGroupRows int64, props ...parquet.WriterProperty) *file.Reader {
	t.Helper()

	sc := arrow.NewSchema([]arrow.Field{{
		Name: "id", Type: column.DataType(), Nullable: true,
		Metadata: arrow.NewMetadata([]string{"PARQUET:field_id"}, []string{"1"}),
	}}, nil)
	rec := array.NewRecordBatch(sc, []arrow.Array{column}, int64(column.Len()))
	defer rec.Release()

	var buf bytes.Buffer
	props = append([]parquet.WriterProperty{parquet.WithStats(true), parquet.WithMaxRowGroupLength(rowGroupRows)}, props...)
	w, err := pqarrow.NewFileWriter(sc, &buf, parquet.NewWriterProperties(props...), pqarrow.DefaultWriterProps())
	require.NoError(t, err)
	require.NoError(t, w.Write(rec))
	require.NoError(t, w.Close())

	rdr, err := file.NewParquetReader(bytes.NewReader(buf.Bytes()))
	require.NoError(t, err)
	t.Cleanup(func() { _ = rdr.Close() })
	require.Equal(t, (column.Len()+int(rowGroupRows)-1)/int(rowGroupRows), rdr.NumRowGroups())

	return rdr
}

// rowGroupMatches returns MightMatch for each of rdr's row groups.
func rowGroupMatches(t *testing.T, rdr *file.Reader, schema *iceberg.Schema, filter iceberg.BooleanExpression) []bool {
	t.Helper()

	arrowSchema, err := pqarrow.FromParquet(rdr.MetaData().Schema, nil, rdr.MetaData().KeyValueMetadata())
	require.NoError(t, err)
	fileSchema, err := table.ArrowSchemaToIcebergWithOptions(arrowSchema, table.ArrowToIcebergOptions{TableSchema: schema})
	require.NoError(t, err)

	rowGroups, err := table.NewParquetRowGroupFilter(rdr, schema, fileSchema, filter, true)
	require.NoError(t, err)
	matches := make([]bool, rdr.NumRowGroups())
	for rg := range matches {
		matches[rg], err = rowGroups.MightMatch(rg)
		require.NoError(t, err)
	}

	return matches
}

func int64Column(values ...int64) arrow.Array {
	bldr := array.NewInt64Builder(memory.DefaultAllocator)
	defer bldr.Release()
	bldr.AppendValues(values, nil)

	return bldr.NewArray()
}

func int64Range(from, to int64) []int64 {
	values := make([]int64, 0, to-from)
	for v := from; v < to; v++ {
		values = append(values, v)
	}

	return values
}

func TestParquetRowGroupFilter_Stats(t *testing.T) {
	schema := iceberg.NewSchema(0, iceberg.NestedField{ID: 1, Name: "id", Type: iceberg.PrimitiveTypes.Int64})
	column := int64Column(int64Range(0, 1000)...)
	defer column.Release()
	rdr := writeRowGroupFilterFile(t, column, 250, parquet.WithDictionaryDefault(false))

	{
		require.Equal(t, []bool{false, true, false, false},
			rowGroupMatches(t, rdr, schema, iceberg.NewAnd(
				iceberg.GreaterThanEqual(iceberg.Reference("id"), int64(260)),
				iceberg.LessThanEqual(iceberg.Reference("id"), int64(270)))))
	}
	{
		require.Equal(t, []bool{false, false, false, false},
			rowGroupMatches(t, rdr, schema, iceberg.EqualTo(iceberg.Reference("id"), int64(1000))))
	}
	{
		// 200 values, the most a scan checks one by one: the two middle row
		// groups are inside their range but hold none of them.
		values := append(int64Range(0, 100), int64Range(900, 1000)...)
		require.Equal(t, []bool{true, false, false, true},
			rowGroupMatches(t, rdr, schema, iceberg.IsIn(iceberg.Reference("id"), values...)))
	}
	{
		// 201 values are past the limit, so like a scan it keeps every row group.
		values := append(int64Range(0, 101), int64Range(900, 1000)...)
		require.Equal(t, []bool{true, true, true, true},
			rowGroupMatches(t, rdr, schema, iceberg.IsIn(iceberg.Reference("id"), values...)))
	}
}

// stringColumn returns rows strings "k-<row group>-<i>", i from 0 to 99, so a
// string such as "k-0-50a" is inside a row group's min/max range but not in it.
func stringColumn(rowGroups int) arrow.Array {
	bldr := array.NewStringBuilder(memory.DefaultAllocator)
	defer bldr.Release()
	for rg := range rowGroups {
		for i := range 100 {
			bldr.Append(fmt.Sprintf("k-%d-%d", rg, i))
		}
	}

	return bldr.NewArray()
}

func TestParquetRowGroupFilter_Bloom(t *testing.T) {
	schema := iceberg.NewSchema(0, iceberg.NestedField{ID: 1, Name: "id", Type: iceberg.PrimitiveTypes.String})
	column := stringColumn(2)
	defer column.Release()
	withBloom := writeRowGroupFilterFile(t, column, 100,
		parquet.WithDictionaryDefault(false), parquet.WithBloomFilterEnabled(true))
	withoutBloom := writeRowGroupFilterFile(t, column, 100, parquet.WithDictionaryDefault(false))

	{
		absent := iceberg.EqualTo(iceberg.Reference("id"), "k-0-50a")
		require.Equal(t, []bool{false, false}, rowGroupMatches(t, withBloom, schema, absent))
		require.Equal(t, []bool{true, false}, rowGroupMatches(t, withoutBloom, schema, absent))
	}
	{
		require.Equal(t, []bool{true, false},
			rowGroupMatches(t, withBloom, schema, iceberg.IsIn(iceberg.Reference("id"), "k-0-50", "k-0-50a")))
	}
}

func TestParquetRowGroupFilter_Dictionary(t *testing.T) {
	schema := iceberg.NewSchema(0, iceberg.NestedField{ID: 1, Name: "id", Type: iceberg.PrimitiveTypes.String})
	column := stringColumn(2)
	defer column.Release()
	// Without compression the writer falls back to plain pages when a
	// dictionary of distinct values costs more than it saves.
	withDictionary := writeRowGroupFilterFile(t, column, 100,
		parquet.WithDictionaryDefault(true), parquet.WithCompression(compress.Codecs.Zstd))
	plain := writeRowGroupFilterFile(t, column, 100, parquet.WithDictionaryDefault(false))

	{
		absent := iceberg.EqualTo(iceberg.Reference("id"), "k-0-50a")
		require.Equal(t, []bool{false, false}, rowGroupMatches(t, withDictionary, schema, absent))
		require.Equal(t, []bool{true, false}, rowGroupMatches(t, plain, schema, absent))
	}
	{
		require.Equal(t, []bool{true, false},
			rowGroupMatches(t, withDictionary, schema, iceberg.EqualTo(iceberg.Reference("id"), "k-0-50")))
	}
}

func TestParquetRowGroupFilter_PromotedIntToLong(t *testing.T) {
	// The table's column is a long, the file's an int written before the
	// column was promoted.
	schema := iceberg.NewSchema(0, iceberg.NestedField{ID: 1, Name: "id", Type: iceberg.PrimitiveTypes.Int64})
	bldr := array.NewInt32Builder(memory.DefaultAllocator)
	for v := range int32(100) {
		bldr.Append(v)
	}
	column := bldr.NewArray()
	bldr.Release()
	defer column.Release()
	rdr := writeRowGroupFilterFile(t, column, 25, parquet.WithDictionaryDefault(false))

	{
		require.Equal(t, []bool{false, true, false, false},
			rowGroupMatches(t, rdr, schema, iceberg.EqualTo(iceberg.Reference("id"), int64(30))))
	}
	{
		require.Equal(t, []bool{true, false, false, true},
			rowGroupMatches(t, rdr, schema, iceberg.IsIn(iceberg.Reference("id"), int64(3), int64(99))))
	}
	{
		// Past the int range: no int can equal it.
		require.Equal(t, []bool{false, false, false, false},
			rowGroupMatches(t, rdr, schema, iceberg.EqualTo(iceberg.Reference("id"), int64(1)<<40)))
	}
}

func TestParquetRowGroupFilter_Decimal(t *testing.T) {
	schema := iceberg.NewSchema(0, iceberg.NestedField{ID: 1, Name: "id", Type: iceberg.DecimalTypeOf(9, 2)})
	decimal := func(unscaled int64) iceberg.Decimal {
		return iceberg.Decimal{Val: decimal128.FromI64(unscaled), Scale: 2}
	}
	bldr := array.NewDecimal128Builder(memory.DefaultAllocator, &arrow.Decimal128Type{Precision: 9, Scale: 2})
	for v := range int64(400) {
		bldr.Append(decimal128.FromI64(v))
	}
	column := bldr.NewArray()
	bldr.Release()
	defer column.Release()

	// Iceberg writers store a decimal of precision 9 in an INT32, whose
	// statistics are little-endian, and other writers use a fixed-length array.
	for _, asInteger := range []bool{true, false} {
		rdr := writeRowGroupFilterFile(t, column, 100,
			parquet.WithDictionaryDefault(false), parquet.WithStoreDecimalAsInteger(asInteger))
		name := "fixed " + strconv.FormatBool(!asInteger)
		require.Equal(t, []bool{false, true, false, false},
			rowGroupMatches(t, rdr, schema, iceberg.EqualTo(iceberg.Reference("id"), decimal(150))), name)
		require.Equal(t, []bool{true, false, false, true},
			rowGroupMatches(t, rdr, schema, iceberg.IsIn(iceberg.Reference("id"), decimal(5), decimal(399))), name)
	}
}

func TestParquetRowGroupFilter_MissingColumn(t *testing.T) {
	column := int64Column(int64Range(0, 100)...)
	defer column.Release()
	rdr := writeRowGroupFilterFile(t, column, 50, parquet.WithDictionaryDefault(false))

	{
		// The file was written before "extra" was added, so its rows read as null.
		schema := iceberg.NewSchema(0,
			iceberg.NestedField{ID: 1, Name: "id", Type: iceberg.PrimitiveTypes.Int64},
			iceberg.NestedField{ID: 2, Name: "extra", Type: iceberg.PrimitiveTypes.String})
		require.Equal(t, []bool{false, false},
			rowGroupMatches(t, rdr, schema, iceberg.EqualTo(iceberg.Reference("extra"), "x")))
		require.Equal(t, []bool{true, true},
			rowGroupMatches(t, rdr, schema, iceberg.IsNull(iceberg.Reference("extra"))))
	}
	{
		// With an initial default the rows read as "x", so none is pruned.
		schema := iceberg.NewSchema(0,
			iceberg.NestedField{ID: 1, Name: "id", Type: iceberg.PrimitiveTypes.Int64},
			iceberg.NestedField{ID: 2, Name: "extra", Type: iceberg.PrimitiveTypes.String, InitialDefault: "x"})
		require.Equal(t, []bool{true, true},
			rowGroupMatches(t, rdr, schema, iceberg.EqualTo(iceberg.Reference("extra"), "x")))
	}
}

func TestParquetRowGroupFilter_Errors(t *testing.T) {
	schema := iceberg.NewSchema(0,
		iceberg.NestedField{ID: 1, Name: "id", Type: iceberg.PrimitiveTypes.Int64},
		iceberg.NestedField{ID: 2, Name: "other", Type: iceberg.PrimitiveTypes.String})
	column := int64Column(int64Range(0, 100)...)
	defer column.Release()
	rdr := writeRowGroupFilterFile(t, column, 50, parquet.WithDictionaryDefault(false))

	{
		// A file schema without a filter column the file has would read it as null and prune every row group.
		_, err := table.NewParquetRowGroupFilter(rdr, schema, iceberg.NewSchema(0, schema.Field(1)),
			iceberg.EqualTo(iceberg.Reference("id"), int64(1)), true)
		require.EqualError(t, err, "invalid argument: file schema lacks filter field 1, which the Parquet file has")
	}
	{
		// Leaving out a column the filter doesn't use is fine.
		rowGroups, err := table.NewParquetRowGroupFilter(rdr, schema, iceberg.NewSchema(0, schema.Field(0)),
			iceberg.EqualTo(iceberg.Reference("id"), int64(1)), true)
		require.NoError(t, err)
		for rg, want := range []bool{true, false} {
			got, err := rowGroups.MightMatch(rg)
			require.NoError(t, err)
			require.Equal(t, want, got)
		}

		_, err = rowGroups.MightMatch(2)
		require.ErrorIs(t, err, iceberg.ErrInvalidArgument)
		require.EqualError(t, err, "invalid argument: row group 2 of 2")
		_, err = rowGroups.MightMatch(-1)
		require.EqualError(t, err, "invalid argument: row group -1 of 2")
	}
}
