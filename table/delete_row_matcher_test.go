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
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/memory"
	"github.com/apache/arrow-go/v18/parquet"
	"github.com/apache/arrow-go/v18/parquet/pqarrow"
	"github.com/apache/iceberg-go"
	iceio "github.com/apache/iceberg-go/io"
	"github.com/apache/iceberg-go/table"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDeleteRowMatcher(t *testing.T) {
	ctx := context.Background()
	arrowSchema := arrow.NewSchema([]arrow.Field{
		{Name: "id", Type: arrow.PrimitiveTypes.Int64, Nullable: false},
		{Name: "data", Type: arrow.BinaryTypes.String, Nullable: true},
	}, nil)
	newTable := func(t *testing.T, formatVersion string) *table.Table {
		tbl := newMergeOnReadTestTableVersion(t, formatVersion)
		data, err := array.TableFromJSON(memory.DefaultAllocator, arrowSchema, []string{
			`[{"id":1,"data":"a"},{"id":2,"data":"a"},{"id":3,"data":"b"},{"id":4,"data":"b"}]`,
		})
		require.NoError(t, err)
		defer data.Release()
		tbl, err = tbl.Append(ctx, array.NewTableReader(data, -1), nil)
		require.NoError(t, err)

		return tbl
	}
	// pairs deletes the rows whose (id, data) is one of keys.
	pairs := func(keys ...string) table.DeleteRowMatcher {
		return func(_ context.Context, batch arrow.RecordBatch) (*array.Boolean, error) {
			ids := batch.Column(batch.Schema().FieldIndices("id")[0]).(*array.Int64)
			data := batch.Column(batch.Schema().FieldIndices("data")[0]).(*array.String)
			bldr := array.NewBooleanBuilder(memory.DefaultAllocator)
			defer bldr.Release()
			for i := range int(batch.NumRows()) {
				key := fmt.Sprintf("%d:%s", ids.Value(i), data.Value(i))
				bldr.Append(key == keys[0] || (len(keys) > 1 && key == keys[1]))
			}

			return bldr.NewBooleanArray(), nil
		}
	}
	superset := iceberg.NewAnd(
		iceberg.IsIn(iceberg.Reference("id"), int64(1), int64(4)),
		iceberg.IsIn(iceberg.Reference("data"), "a", "b"),
	)

	for _, formatVersion := range []string{"2", "3"} {
		t.Run("deletes only matched rows on v"+formatVersion, func(t *testing.T) {
			tbl, err := newTable(t, formatVersion).Delete(ctx, superset, nil, table.WithDeleteRowMatcher([]string{"id", "data"}, pairs("1:a", "4:b")))
			require.NoError(t, err)
			assert.Equal(t, []int64{2, 3}, idsInTable(t, tbl))
		})
	}
	for _, formatVersion := range []string{"2", "3"} {
		t.Run("deletes again from a file with deletes on v"+formatVersion, func(t *testing.T) {
			tbl, err := newTable(t, formatVersion).Delete(ctx, superset, nil, table.WithDeleteRowMatcher([]string{"id", "data"}, pairs("1:a")))
			require.NoError(t, err)
			tbl, err = tbl.Delete(ctx, superset, nil, table.WithDeleteRowMatcher([]string{"id", "data"}, pairs("1:a", "4:b")))
			require.NoError(t, err)
			assert.Equal(t, []int64{2, 3}, idsInTable(t, tbl))
			assert.Equal(t, "2", tbl.CurrentSnapshot().Summary.Properties["total-position-deletes"])
		})
	}
	t.Run("keeps rows the matcher rejects", func(t *testing.T) {
		// The filter alone would delete id 4 too.
		tbl, err := newTable(t, "2").Delete(ctx, superset, nil, table.WithDeleteRowMatcher([]string{"id", "data"}, pairs("1:a")))
		require.NoError(t, err)
		assert.Equal(t, []int64{2, 3, 4}, idsInTable(t, tbl))
	})
	t.Run("keeps a file the filter matches whole", func(t *testing.T) {
		// id >= 1 strictly matches the only data file, which a plain delete drops without reading.
		tbl, err := newTable(t, "2").Delete(ctx, iceberg.GreaterThanEqual(iceberg.Reference("id"), int64(1)), nil,
			table.WithDeleteRowMatcher([]string{"id", "data"}, pairs("2:a")))
		require.NoError(t, err)
		assert.Equal(t, []int64{1, 3, 4}, idsInTable(t, tbl))
	})
	t.Run("refuses copy-on-write", func(t *testing.T) {
		tbl := newTable(t, "2")
		txn := tbl.NewTransaction()
		require.NoError(t, txn.SetProperties(iceberg.Properties{table.WriteDeleteModeKey: table.WriteModeCopyOnWrite}))
		err := txn.Delete(ctx, superset, nil, table.WithDeleteRowMatcher([]string{"id", "data"}, pairs("1:a")))
		assert.ErrorContains(t, err, "a delete row matcher needs write.delete.mode=merge-on-read")
	})
	t.Run("rejects nil delete row matcher function", func(t *testing.T) {
		tbl := newTable(t, "2")
		_, err := tbl.Delete(ctx, superset, nil, table.WithDeleteRowMatcher([]string{"id"}, nil))
		assert.ErrorContains(t, err, "delete row matcher function is required")
	})
	t.Run("rejects unknown delete row matcher column", func(t *testing.T) {
		tbl := newTable(t, "2")
		_, err := tbl.Delete(ctx, superset, nil, table.WithDeleteRowMatcher([]string{"missing"}, pairs("1:a")))
		assert.ErrorContains(t, err, `delete row matcher column "missing" is not in the table schema`)
	})
	t.Run("skips no-op commit and uses fallback name mapping for field-ID-less files", func(t *testing.T) {
		tbl := newMergeOnReadTestTableVersion(t, "2")
		_, hasMapping := tbl.Properties()[table.DefaultNameMappingKey]
		require.False(t, hasMapping, "setup: table should start without a name mapping")

		// Write a Parquet file without Iceberg field IDs so the scan relies on buildScanMetadata's fallback mapping.
		data, err := array.TableFromJSON(memory.DefaultAllocator, arrowSchema, []string{
			`[{"id":1,"data":"a"},{"id":2,"data":"b"}]`,
		})
		require.NoError(t, err)
		defer data.Release()

		dataPath := filepath.Join(tbl.Location(), "data", "no-field-ids.parquet")
		require.NoError(t, os.MkdirAll(filepath.Dir(dataPath), 0o755))
		f, err := os.Create(dataPath)
		require.NoError(t, err)
		require.NoError(t, pqarrow.WriteTable(data, f, data.NumRows(), parquet.NewWriterProperties(), pqarrow.DefaultWriterProps()))

		info, err := os.Stat(dataPath)
		require.NoError(t, err)
		bldr, err := iceberg.NewDataFileBuilder(*iceberg.UnpartitionedSpec, iceberg.EntryContentData, filepath.ToSlash(dataPath), iceberg.ParquetFile, nil, nil, nil, data.NumRows(), info.Size())
		require.NoError(t, err)

		txn := tbl.NewTransaction()
		require.NoError(t, txn.AddDataFiles(ctx, []iceberg.DataFile{bldr.Build()}, nil, table.WithoutAutoNameMapping()))
		tbl, err = txn.Commit(ctx)
		require.NoError(t, err)

		_, hasMapping = tbl.Properties()[table.DefaultNameMappingKey]
		require.False(t, hasMapping, "setup: WithoutAutoNameMapping should not persist a name mapping")
		snapBefore := tbl.CurrentSnapshot().SnapshotID

		// A zero-match matcher delete scans via fallback mapping and skips staging a snapshot or name mapping.
		var seenMu sync.Mutex
		var seenIDs []int64
		noMatch := func(_ context.Context, batch arrow.RecordBatch) (*array.Boolean, error) {
			ids := batch.Column(batch.Schema().FieldIndices("id")[0]).(*array.Int64)
			bldr := array.NewBooleanBuilder(memory.DefaultAllocator)
			defer bldr.Release()
			seenMu.Lock()
			for i := range int(batch.NumRows()) {
				seenIDs = append(seenIDs, ids.Value(i))
				bldr.Append(false)
			}
			seenMu.Unlock()

			return bldr.NewBooleanArray(), nil
		}
		tbl, err = tbl.Delete(ctx, iceberg.AlwaysTrue{}, iceberg.Properties{"checkpoint": "1"},
			table.WithDeleteRowMatcher([]string{"id", "data"}, noMatch))
		require.NoError(t, err)
		assert.ElementsMatch(t, []int64{1, 2}, seenIDs)
		assert.Equal(t, snapBefore, tbl.CurrentSnapshot().SnapshotID)
		_, hasMapping = tbl.Properties()[table.DefaultNameMappingKey]
		assert.False(t, hasMapping, "no-op matcher delete must not persist a default name mapping")

		// A matching matcher delete persists the name mapping and commits a new snapshot.
		tbl, err = tbl.Delete(ctx, iceberg.AlwaysTrue{}, nil,
			table.WithDeleteRowMatcher([]string{"id", "data"}, pairs("1:a")))
		require.NoError(t, err)
		assert.NotEqual(t, snapBefore, tbl.CurrentSnapshot().SnapshotID)
		_, hasMapping = tbl.Properties()[table.DefaultNameMappingKey]
		assert.True(t, hasMapping, "matcher delete that modifies rows must persist default name mapping")
		assert.Equal(t, []int64{2}, idsInTable(t, tbl))
	})
	t.Run("sees a renamed key column under its new name", func(t *testing.T) {
		tbl := newTable(t, "2")
		txn := tbl.NewTransaction()
		require.NoError(t, txn.UpdateSchema(true, false).RenameColumn([]string{"data"}, "label").Commit())
		tbl, err := txn.Commit(ctx)
		require.NoError(t, err)

		matcher := func(_ context.Context, batch arrow.RecordBatch) (*array.Boolean, error) {
			labels := batch.Column(batch.Schema().FieldIndices("label")[0]).(*array.String)
			bldr := array.NewBooleanBuilder(memory.DefaultAllocator)
			defer bldr.Release()
			for i := range int(batch.NumRows()) {
				bldr.Append(labels.Value(i) == "b")
			}

			return bldr.NewBooleanArray(), nil
		}
		tbl, err = tbl.Delete(ctx, iceberg.IsIn(iceberg.Reference("id"), int64(1), int64(3)), nil,
			table.WithDeleteRowMatcher([]string{"id", "label"}, matcher))
		require.NoError(t, err)
		assert.Equal(t, []int64{1, 2, 4}, idsInTable(t, tbl))
	})
	t.Run("sees a key column added after the file as nulls", func(t *testing.T) {
		tbl := newTable(t, "2")
		txn := tbl.NewTransaction()
		require.NoError(t, txn.UpdateSchema(true, false).AddColumn([]string{"extra"}, iceberg.PrimitiveTypes.String, "", false, nil).Commit())
		tbl, err := txn.Commit(ctx)
		require.NoError(t, err)

		matcher := func(_ context.Context, batch arrow.RecordBatch) (*array.Boolean, error) {
			extra := batch.Column(batch.Schema().FieldIndices("extra")[0])
			bldr := array.NewBooleanBuilder(memory.DefaultAllocator)
			defer bldr.Release()
			for i := range int(batch.NumRows()) {
				bldr.Append(extra.IsNull(i))
			}

			return bldr.NewBooleanArray(), nil
		}
		tbl, err = tbl.Delete(ctx, iceberg.EqualTo(iceberg.Reference("id"), int64(2)), nil,
			table.WithDeleteRowMatcher([]string{"id", "extra"}, matcher))
		require.NoError(t, err)
		assert.Equal(t, []int64{1, 3, 4}, idsInTable(t, tbl))
	})
	t.Run("sees a promoted key column in the table's type", func(t *testing.T) {
		location := filepath.ToSlash(t.TempDir())
		schema := iceberg.NewSchema(0,
			iceberg.NestedField{ID: 1, Name: "id", Type: iceberg.PrimitiveTypes.Int64, Required: true},
			iceberg.NestedField{ID: 2, Name: "k", Type: iceberg.PrimitiveTypes.Int32, Required: true},
		)
		meta, err := table.NewMetadata(schema, iceberg.UnpartitionedSpec, table.UnsortedSortOrder, location, iceberg.Properties{
			table.PropertyFormatVersion: "2", table.WriteDeleteModeKey: table.WriteModeMergeOnRead,
		})
		require.NoError(t, err)
		metaLoc := location + "/metadata/v1.metadata.json"
		fsF := func(context.Context) (iceio.IO, error) { return iceio.LocalFS{}, nil }
		tbl := table.New(table.Identifier{"db", "promoted"}, meta, metaLoc, fsF, &concurrentTestCatalog{metadata: meta, location: metaLoc, fsF: fsF})

		data, err := array.TableFromJSON(memory.DefaultAllocator, arrow.NewSchema([]arrow.Field{
			{Name: "id", Type: arrow.PrimitiveTypes.Int64},
			{Name: "k", Type: arrow.PrimitiveTypes.Int32},
		}, nil), []string{`[{"id":1,"k":10},{"id":2,"k":20},{"id":3,"k":30}]`})
		require.NoError(t, err)
		defer data.Release()
		tbl, err = tbl.Append(ctx, array.NewTableReader(data, -1), nil)
		require.NoError(t, err)

		txn := tbl.NewTransaction()
		require.NoError(t, txn.UpdateSchema(true, false).UpdateColumn([]string{"k"}, table.ColumnUpdate{
			FieldType: iceberg.Optional[iceberg.Type]{Valid: true, Val: iceberg.PrimitiveTypes.Int64},
		}).Commit())
		tbl, err = txn.Commit(ctx)
		require.NoError(t, err)

		matcher := func(_ context.Context, batch arrow.RecordBatch) (*array.Boolean, error) {
			ks, ok := batch.Column(batch.Schema().FieldIndices("k")[0]).(*array.Int64)
			if !ok {
				return nil, fmt.Errorf("k is %s, not int64", batch.Column(batch.Schema().FieldIndices("k")[0]).DataType())
			}
			bldr := array.NewBooleanBuilder(memory.DefaultAllocator)
			defer bldr.Release()
			for i := range int(batch.NumRows()) {
				bldr.Append(ks.Value(i) == 20)
			}

			return bldr.NewBooleanArray(), nil
		}
		tbl, err = tbl.Delete(ctx, iceberg.GreaterThanEqual(iceberg.Reference("id"), int64(1)), nil,
			table.WithDeleteRowMatcher([]string{"k"}, matcher))
		require.NoError(t, err)
		assert.Equal(t, []int64{1, 3}, idsInTable(t, tbl))
	})
}
