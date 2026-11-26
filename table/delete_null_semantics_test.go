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
	"path/filepath"
	"slices"
	"testing"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/memory"
	"github.com/apache/iceberg-go"
	iceio "github.com/apache/iceberg-go/io"
	"github.com/apache/iceberg-go/table"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestDeleteNullSemantics checks that Delete removes exactly the rows a scan with the same filter returns, in every
// delete mode and format version. Copy-on-write used to keep only rows matching NOT filter, which drops every row the
// filter is null on (a comparison on a NULL column), so those rows vanished without being deleted.
//
// Negated comparisons are guarded with NotNull. Iceberg defines notEqual and notIn as true for null, and the strict
// metrics evaluator follows that, while the Arrow row filter treats them as null, so an unguarded one deletes a whole
// all-null file but not the null rows of a mixed file. A caller wanting SQL semantics guards them, as Spark does.
func TestDeleteNullSemantics(t *testing.T) {
	ref := iceberg.Reference("data")
	id := iceberg.Reference("id")
	isA := iceberg.EqualTo(ref, "a")
	filters := []iceberg.BooleanExpression{
		isA,
		iceberg.IsIn(ref, "a", "b"),
		iceberg.GreaterThan(ref, "a"),
		iceberg.LessThanEqual(ref, "b"),
		iceberg.IsNull(ref),
		iceberg.NotNull(ref),
		iceberg.NewAnd(iceberg.NotNull(ref), iceberg.NotEqualTo(ref, "a")),
		iceberg.NewAnd(iceberg.NotNull(ref), iceberg.NotIn(ref, "a", "b")),
		iceberg.NewAnd(iceberg.NotNull(ref), iceberg.NewNot(isA)),
		iceberg.NewOr(isA, iceberg.EqualTo(id, int64(2))),
		iceberg.NewAnd(iceberg.NewAnd(iceberg.NotNull(ref), iceberg.NewNot(isA)), iceberg.LessThan(id, int64(5))),
		iceberg.NewAnd(iceberg.NotNull(ref), iceberg.NewNot(iceberg.NewOr(isA, iceberg.EqualTo(id, int64(4))))),
	}

	for _, mode := range []string{table.WriteModeCopyOnWrite, table.WriteModeMergeOnRead} {
		for _, version := range []string{"2", "3"} {
			for _, filter := range filters {
				t.Run(fmt.Sprintf("%s/v%s/%s", mode, version, filter), func(t *testing.T) {
					tbl := appendNullableRows(t, newDeleteModeTestTable(t, mode, version))
					matched := idsMatching(t, tbl, filter)

					tbl, err := tbl.Delete(context.Background(), filter, nil)
					require.NoError(t, err)

					var want []int64
					for _, id := range []int64{1, 2, 3, 4, 5, 6} {
						if !slices.Contains(matched, id) {
							want = append(want, id)
						}
					}
					assert.Equal(t, want, idsInTable(t, tbl))
				})
			}
		}
	}
}

// TestDeleteKeepsNullRowsInSingleMatchFile covers a row group with as many NULLs as values. Parquet statistics count
// only non-null values, and comparing that to the null count made the row group look all-null, so a merge-on-read
// delete (and a filtered scan) skipped it.
func TestDeleteKeepsNullRowsInSingleMatchFile(t *testing.T) {
	for _, mode := range []string{table.WriteModeCopyOnWrite, table.WriteModeMergeOnRead} {
		t.Run(mode, func(t *testing.T) {
			ctx := context.Background()
			tbl := newDeleteModeTestTable(t, mode, "2")

			arrowSchema := nullableTestArrowSchema()
			data, err := array.TableFromJSON(memory.DefaultAllocator, arrowSchema, []string{`[{"id":1,"data":"a"},{"id":2,"data":null}]`})
			require.NoError(t, err)
			defer data.Release()
			tbl, err = tbl.Append(ctx, array.NewTableReader(data, -1), nil)
			require.NoError(t, err)

			assert.Equal(t, []int64{1}, idsMatching(t, tbl, iceberg.EqualTo(iceberg.Reference("data"), "a")))

			tbl, err = tbl.Delete(ctx, iceberg.EqualTo(iceberg.Reference("data"), "a"), nil)
			require.NoError(t, err)
			assert.Equal(t, []int64{2}, idsInTable(t, tbl))

			tbl, err = tbl.Delete(ctx, iceberg.NewAnd(iceberg.NotNull(iceberg.Reference("data")), iceberg.NotEqualTo(iceberg.Reference("data"), "a")), nil)
			require.NoError(t, err)
			assert.Equal(t, []int64{2}, idsInTable(t, tbl))
		})
	}
}

// idsMatching returns the sorted ids a scan with filter returns.
func idsMatching(t *testing.T, tbl *table.Table, filter iceberg.BooleanExpression) []int64 {
	t.Helper()

	_, itr, err := tbl.Scan(table.WithRowFilter(filter), table.WithSelectedFields("id")).ToArrowRecords(context.Background())
	require.NoError(t, err)

	var ids []int64
	for rec, err := range itr {
		require.NoError(t, err)
		col := rec.Column(rec.Schema().FieldIndices("id")[0]).(*array.Int64)
		for i := range int(rec.NumRows()) {
			ids = append(ids, col.Value(i))
		}
		rec.Release()
	}
	slices.Sort(ids)

	return ids
}

func nullableTestArrowSchema() *arrow.Schema {
	return arrow.NewSchema([]arrow.Field{
		{Name: "id", Type: arrow.PrimitiveTypes.Int64, Nullable: false},
		{Name: "data", Type: arrow.BinaryTypes.String, Nullable: true},
	}, nil)
}

func appendNullableRows(t *testing.T, tbl *table.Table) *table.Table {
	t.Helper()

	data, err := array.TableFromJSON(memory.DefaultAllocator, nullableTestArrowSchema(), []string{
		`[{"id":1,"data":"a"},{"id":2,"data":null},{"id":3,"data":"b"},` +
			`{"id":4,"data":null},{"id":5,"data":"a"},{"id":6,"data":"c"}]`,
	})
	require.NoError(t, err)
	defer data.Release()

	tbl, err = tbl.Append(context.Background(), array.NewTableReader(data, -1), nil)
	require.NoError(t, err)

	return tbl
}

func newDeleteModeTestTable(t *testing.T, mode, formatVersion string) *table.Table {
	t.Helper()

	location := filepath.ToSlash(t.TempDir())
	schema := iceberg.NewSchema(0,
		iceberg.NestedField{ID: 1, Name: "id", Type: iceberg.PrimitiveTypes.Int64, Required: true},
		iceberg.NestedField{ID: 2, Name: "data", Type: iceberg.PrimitiveTypes.String, Required: false},
	)
	props := iceberg.Properties{
		table.PropertyFormatVersion: formatVersion,
		table.WriteDeleteModeKey:    mode,
	}
	meta, err := table.NewMetadata(schema, iceberg.UnpartitionedSpec, table.UnsortedSortOrder, location, props)
	require.NoError(t, err)

	metaLoc := location + "/metadata/v1.metadata.json"
	fsF := func(context.Context) (iceio.IO, error) { return iceio.LocalFS{}, nil }
	cat := &concurrentTestCatalog{metadata: meta, location: metaLoc, fsF: fsF}

	return table.New(table.Identifier{"db", "delete_null_test"}, meta, metaLoc, fsF, cat)
}
