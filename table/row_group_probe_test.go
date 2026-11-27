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
	"errors"
	"fmt"
	"path/filepath"
	"slices"
	"strings"
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

func TestRowGroupProbe(t *testing.T) {
	ctx := context.Background()
	// newTable returns a table with ids 1 to 20 in one data file of four row groups (1-5, 6-10, 11-15, 16-20).
	newTable := func(t *testing.T, formatVersion string) *table.Table {
		tbl := newMergeOnReadTestTableVersion(t, formatVersion)
		rows := make([]string, 20)
		for i := range rows {
			rows[i] = fmt.Sprintf(`{"id":%d,"data":"v%d"}`, i+1, i+1)
		}
		data, err := array.TableFromJSON(memory.DefaultAllocator, arrow.NewSchema([]arrow.Field{
			{Name: "id", Type: arrow.PrimitiveTypes.Int64, Nullable: false},
			{Name: "data", Type: arrow.BinaryTypes.String, Nullable: true},
		}, nil), []string{"[" + strings.Join(rows, ",") + "]"})
		require.NoError(t, err)
		defer data.Release()
		tbl, err = tbl.Append(ctx, array.NewTableReader(data, -1), nil)
		require.NoError(t, err)

		return tbl
	}
	// idIn selects the rows whose column is one of ids, read as an int64 column.
	idIn := func(column string, ids ...int64) table.RowMatcher {
		return func(_ context.Context, batch arrow.RecordBatch) (*array.Boolean, error) {
			values, ok := batch.Column(batch.Schema().FieldIndices(column)[0]).(*array.Int64)
			if !ok {
				return nil, fmt.Errorf("%s is %s, not int64", column, batch.Column(batch.Schema().FieldIndices(column)[0]).DataType())
			}
			bldr := array.NewBooleanBuilder(memory.DefaultAllocator)
			defer bldr.Release()
			for i := range values.Len() {
				bldr.Append(values.IsValid(i) && slices.Contains(ids, values.Value(i)))
			}

			return bldr.NewBooleanArray(), nil
		}
	}
	scanIDs := func(t *testing.T, tbl *table.Table, column string, opts ...table.ScanOption) []int64 {
		_, records, err := tbl.Scan(append(opts, table.WithSelectedFields(column, "data"))...).ToArrowRecords(ctx)
		require.NoError(t, err)
		var ids []int64
		for rec, err := range records {
			require.NoError(t, err)
			col := rec.Column(rec.Schema().FieldIndices(column)[0]).(*array.Int64)
			for i := range col.Len() {
				ids = append(ids, col.Value(i))
			}
			rec.Release()
		}
		slices.Sort(ids)

		return ids
	}

	t.Run("reads only the row groups the matcher selects a row of", func(t *testing.T) {
		tbl := newTable(t, "2")
		// Kept row groups are returned whole, since the probe only skips row groups.
		assert.Equal(t, []int64{6, 7, 8, 9, 10, 16, 17, 18, 19, 20},
			scanIDs(t, tbl, "id", table.WithRowGroupProbe([]string{"id"}, idIn("id", 7, 18))))
		assert.Empty(t, scanIDs(t, tbl, "id", table.WithRowGroupProbe([]string{"id"}, idIn("id", 99))))
	})
	for _, formatVersion := range []string{"2", "3"} {
		t.Run("applies position deletes after skipping row groups on v"+formatVersion, func(t *testing.T) {
			tbl, err := newTable(t, formatVersion).Delete(ctx, iceberg.IsIn(iceberg.Reference("id"), int64(2), int64(8), int64(17)), nil)
			require.NoError(t, err)
			// Row 17 is deleted, so the first kept row group starts at file position 5 and the second at 15: without
			// each kept row group's file position, the deletes would drop the wrong rows.
			assert.Equal(t, []int64{6, 7, 9, 10, 16, 18, 19, 20},
				scanIDs(t, tbl, "id", table.WithRowFilter(iceberg.GreaterThan(iceberg.Reference("id"), int64(0))),
					table.WithRowGroupProbe([]string{"id"}, idIn("id", 7, 18))))
			// The probe ignores deletes: a row group whose only matching row is deleted is still read.
			assert.Equal(t, []int64{6, 7, 9, 10},
				scanIDs(t, tbl, "id", table.WithRowGroupProbe([]string{"id"}, idIn("id", 8))))
		})
	}
	t.Run("applies equality deletes after skipping row groups", func(t *testing.T) {
		tbl := newTable(t, "2")
		records, release := makeEqDeleteRecords(t, idKeyArrowSchema(t), `[{"id": 7}]`)
		defer release()
		txn := tbl.NewTransaction()
		require.NoError(t, txn.CommitEqualityDeletes(ctx, []int{1}, records, nil))
		tbl, err := txn.Commit(ctx)
		require.NoError(t, err)
		assert.Equal(t, []int64{6, 8, 9, 10},
			scanIDs(t, tbl, "id", table.WithRowGroupProbe([]string{"id"}, idIn("id", 7, 8))))
	})
	t.Run("probes a renamed column under its new name", func(t *testing.T) {
		tbl := newTable(t, "2")
		txn := tbl.NewTransaction()
		require.NoError(t, txn.UpdateSchema(true, false).RenameColumn([]string{"id"}, "key").Commit())
		tbl, err := txn.Commit(ctx)
		require.NoError(t, err)
		assert.Equal(t, []int64{11, 12, 13, 14, 15},
			scanIDs(t, tbl, "key", table.WithRowGroupProbe([]string{"key"}, idIn("key", 12))))
	})
	t.Run("probes a widened column in the table's type", func(t *testing.T) {
		location := filepath.ToSlash(t.TempDir())
		meta, err := table.NewMetadata(iceberg.NewSchema(0,
			iceberg.NestedField{ID: 1, Name: "id", Type: iceberg.PrimitiveTypes.Int32, Required: true},
			iceberg.NestedField{ID: 2, Name: "data", Type: iceberg.PrimitiveTypes.String},
		), iceberg.UnpartitionedSpec, table.UnsortedSortOrder, location, iceberg.Properties{
			table.PropertyFormatVersion: "2", table.ParquetRowGroupLimitKey: "2",
		})
		require.NoError(t, err)
		metaLoc := location + "/metadata/v1.metadata.json"
		fsF := func(context.Context) (iceio.IO, error) { return iceio.LocalFS{}, nil }
		tbl := table.New(table.Identifier{"db", "widened"}, meta, metaLoc, fsF, &concurrentTestCatalog{metadata: meta, location: metaLoc, fsF: fsF})
		data, err := array.TableFromJSON(memory.DefaultAllocator, arrow.NewSchema([]arrow.Field{
			{Name: "id", Type: arrow.PrimitiveTypes.Int32},
			{Name: "data", Type: arrow.BinaryTypes.String},
		}, nil), []string{`[{"id":1,"data":"a"},{"id":2,"data":"b"},{"id":3,"data":"c"},{"id":4,"data":"d"}]`})
		require.NoError(t, err)
		defer data.Release()
		tbl, err = tbl.Append(ctx, array.NewTableReader(data, -1), nil)
		require.NoError(t, err)

		txn := tbl.NewTransaction()
		require.NoError(t, txn.UpdateSchema(true, false).UpdateColumn([]string{"id"}, table.ColumnUpdate{
			FieldType: iceberg.Optional[iceberg.Type]{Valid: true, Val: iceberg.PrimitiveTypes.Int64},
		}).Commit())
		tbl, err = txn.Commit(ctx)
		require.NoError(t, err)
		// idIn fails on an int32 column, so the probe must see the table's int64.
		assert.Equal(t, []int64{3, 4}, scanIDs(t, tbl, "id", table.WithRowGroupProbe([]string{"id"}, idIn("id", 4))))
	})
	t.Run("reads a file without a probed column whole", func(t *testing.T) {
		tbl := newTable(t, "3")
		txn := tbl.NewTransaction()
		require.NoError(t, txn.UpdateSchema(true, false).AddColumn([]string{"extra"}, iceberg.PrimitiveTypes.Int64, "",
			false, iceberg.Int64Literal(5)).Commit())
		tbl, err := txn.Commit(ctx)
		require.NoError(t, err)
		// Every row reads extra as its initial default 5, which only a whole read returns.
		ids := scanIDs(t, tbl, "id", table.WithRowGroupProbe([]string{"extra"}, idIn("extra", 5)))
		assert.Len(t, ids, 20)
	})
	t.Run("fails the scan on an unknown column", func(t *testing.T) {
		_, _, err := newTable(t, "2").Scan(table.WithRowGroupProbe([]string{"missing"}, idIn("missing"))).ToArrowRecords(ctx)
		assert.ErrorContains(t, err, "row group probe")
	})
	t.Run("fails the scan when the matcher fails", func(t *testing.T) {
		failing := func(context.Context, arrow.RecordBatch) (*array.Boolean, error) {
			return nil, errors.New("matcher failed")
		}
		_, records, err := newTable(t, "2").Scan(table.WithRowGroupProbe([]string{"id"}, failing)).ToArrowRecords(ctx)
		require.NoError(t, err)
		for _, err = range records {
			if err != nil {
				break
			}
		}
		assert.ErrorContains(t, err, "matcher failed")
	})
}
