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
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/compute"
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
	t.Run("v3 merge-on-read skips already-deleted rows before calling matcher", func(t *testing.T) {
		tbl := newTable(t, "3")
		tbl, err := tbl.Delete(ctx, iceberg.EqualTo(iceberg.Reference("id"), int64(1)), nil)
		require.NoError(t, err)

		// Fail if the matcher ever sees the already-deleted id=1 row.
		boom := errors.New("saw already-deleted row")
		matcher := func(_ context.Context, batch arrow.RecordBatch) (*array.Boolean, error) {
			ids := batch.Column(batch.Schema().FieldIndices("id")[0]).(*array.Int64)
			bldr := array.NewBooleanBuilder(memory.DefaultAllocator)
			defer bldr.Release()
			for i := range int(batch.NumRows()) {
				if ids.Value(i) == 1 {
					return nil, boom
				}
				bldr.Append(ids.Value(i) == 3)
			}

			return bldr.NewBooleanArray(), nil
		}
		tbl, err = tbl.Delete(ctx, iceberg.AlwaysTrue{}, nil, table.WithDeleteRowMatcher([]string{"id"}, matcher))
		require.NoError(t, err)
		assert.Equal(t, []int64{2, 4}, idsInTable(t, tbl))
	})
	t.Run("deletes matched rows on copy-on-write v1 and v2", func(t *testing.T) {
		for _, version := range []string{"1", "2"} {
			location := filepath.ToSlash(t.TempDir())
			schema := iceberg.NewSchema(0,
				iceberg.NestedField{ID: 1, Name: "id", Type: iceberg.PrimitiveTypes.Int64, Required: true},
				iceberg.NestedField{ID: 2, Name: "data", Type: iceberg.PrimitiveTypes.String, Required: false},
			)
			props := iceberg.Properties{table.PropertyFormatVersion: version}
			if version == "2" {
				props[table.WriteDeleteModeKey] = table.WriteModeCopyOnWrite
			}
			meta, err := table.NewMetadata(schema, iceberg.UnpartitionedSpec, table.UnsortedSortOrder, location, props)
			require.NoError(t, err)
			metaLoc := location + "/metadata/v1.metadata.json"
			fsF := func(context.Context) (iceio.IO, error) { return iceio.LocalFS{}, nil }
			tbl := table.New(table.Identifier{"db", "cow_v" + version}, meta, metaLoc, fsF, &concurrentTestCatalog{metadata: meta, location: metaLoc, fsF: fsF})

			data, err := array.TableFromJSON(memory.DefaultAllocator, arrowSchema, []string{
				`[{"id":1,"data":"a"},{"id":2,"data":null},{"id":3,"data":"b"},{"id":4,"data":"b"},{"id":4,"data":"b"}]`,
			})
			require.NoError(t, err)
			defer data.Release()
			tbl, err = tbl.Append(ctx, array.NewTableReader(data, -1), nil)
			require.NoError(t, err)

			mem := memory.NewCheckedAllocator(memory.NewGoAllocator())
			checkedCtx := compute.WithAllocator(ctx, mem)
			// Filter matches all non-null rows strictly on id >= 1; matcher deletes only id 1 and duplicate id 4 rows.
			matcher := func(mctx context.Context, batch arrow.RecordBatch) (*array.Boolean, error) {
				ids := batch.Column(batch.Schema().FieldIndices("id")[0]).(*array.Int64)
				bldr := array.NewBooleanBuilder(compute.GetAllocator(mctx))
				defer bldr.Release()
				for i := range int(batch.NumRows()) {
					id := ids.Value(i)
					if id == 1 {
						bldr.Append(true)

						continue
					}
					if id == 4 {
						bldr.Append(true)

						continue
					}
					bldr.Append(false)
				}

				return bldr.NewBooleanArray(), nil
			}
			tbl, err = tbl.Delete(checkedCtx, iceberg.GreaterThanEqual(iceberg.Reference("id"), int64(1)), nil,
				table.WithDeleteRowMatcher([]string{"id", "data"}, matcher))
			require.NoError(t, err)
			assert.Equal(t, []int64{2, 3}, idsInTable(t, tbl))
			mem.AssertSize(t, 0)

			// A zero-match row-matcher delete leaves the snapshot unchanged and writes no new files.
			snapBefore := tbl.CurrentSnapshot().SnapshotID
			beforeZeroParquet, err := filepath.Glob(filepath.Join(location, "data", "*.parquet"))
			require.NoError(t, err)
			noMatch := func(mctx context.Context, batch arrow.RecordBatch) (*array.Boolean, error) {
				bldr := array.NewBooleanBuilder(compute.GetAllocator(mctx))
				defer bldr.Release()
				for range int(batch.NumRows()) {
					bldr.Append(false)
				}

				return bldr.NewBooleanArray(), nil
			}
			tbl, err = tbl.Delete(checkedCtx, iceberg.AlwaysTrue{}, nil, table.WithDeleteRowMatcher([]string{"id"}, noMatch))
			require.NoError(t, err)
			assert.Equal(t, snapBefore, tbl.CurrentSnapshot().SnapshotID)
			assert.Equal(t, []int64{2, 3}, idsInTable(t, tbl))
			afterZeroParquet, err := filepath.Glob(filepath.Join(location, "data", "*.parquet"))
			require.NoError(t, err)
			assert.Equal(t, beforeZeroParquet, afterZeroParquet)
			mem.AssertSize(t, 0)
		}
	})
	t.Run("copy-on-write matcher error cleans up rewritten files and leaks no Arrow memory", func(t *testing.T) {
		location := filepath.ToSlash(t.TempDir())
		schema := iceberg.NewSchema(0,
			iceberg.NestedField{ID: 1, Name: "id", Type: iceberg.PrimitiveTypes.Int64, Required: true},
		)
		meta, err := table.NewMetadata(schema, iceberg.UnpartitionedSpec, table.UnsortedSortOrder, location, iceberg.Properties{
			table.PropertyFormatVersion: "2",
			table.WriteDeleteModeKey:    table.WriteModeCopyOnWrite,
		})
		require.NoError(t, err)
		metaLoc := location + "/metadata/v1.metadata.json"
		fsF := func(context.Context) (iceio.IO, error) { return iceio.LocalFS{}, nil }
		tbl := table.New(table.Identifier{"db", "cow_err"}, meta, metaLoc, fsF, &concurrentTestCatalog{metadata: meta, location: metaLoc, fsF: fsF})

		idSchema := arrow.NewSchema([]arrow.Field{{Name: "id", Type: arrow.PrimitiveTypes.Int64, Nullable: false}}, nil)
		for _, rowJSON := range []string{`[{"id":1},{"id":2}]`, `[{"id":3},{"id":4}]`} {
			data, err := array.TableFromJSON(memory.DefaultAllocator, idSchema, []string{rowJSON})
			require.NoError(t, err)
			tbl, err = tbl.Append(ctx, array.NewTableReader(data, -1), nil)
			data.Release()
			require.NoError(t, err)
		}

		beforeParquet, err := filepath.Glob(filepath.Join(location, "data", "*.parquet"))
		require.NoError(t, err)

		mem := memory.NewCheckedAllocator(memory.NewGoAllocator())
		checkedCtx := compute.WithAllocator(ctx, mem)
		matcherErr := errors.New("matcher failed on second file")
		var calls atomic.Int64
		var duringParquet []string
		matcher := func(mctx context.Context, batch arrow.RecordBatch) (*array.Boolean, error) {
			call := calls.Add(1)
			if call >= 2 {
				duringParquet, _ = filepath.Glob(filepath.Join(location, "data", "*.parquet"))

				return nil, matcherErr
			}
			ids := batch.Column(batch.Schema().FieldIndices("id")[0]).(*array.Int64)
			firstID := ids.Value(0)
			bldr := array.NewBooleanBuilder(compute.GetAllocator(mctx))
			defer bldr.Release()
			for i := range int(batch.NumRows()) {
				bldr.Append(ids.Value(i) == firstID)
			}

			return bldr.NewBooleanArray(), nil
		}
		txn := tbl.NewTransaction()
		err = txn.Delete(checkedCtx, iceberg.AlwaysTrue{}, nil, table.WithDeleteRowMatcher([]string{"id"}, matcher))
		require.ErrorIs(t, err, matcherErr)
		assert.GreaterOrEqual(t, calls.Load(), int64(2))
		assert.Greater(t, len(duringParquet), len(beforeParquet), "first file must be rewritten before second file errors")
		mem.AssertSize(t, 0)

		afterParquet, err := filepath.Glob(filepath.Join(location, "data", "*.parquet"))
		require.NoError(t, err)
		assert.Equal(t, beforeParquet, afterParquet)
	})
	t.Run("copy-on-write partial-batch filter and matcher preserves non-matching and null-filter rows", func(t *testing.T) {
		tbl := newDeleteModeTestTable(t, table.WriteModeCopyOnWrite, "2")
		data, err := array.TableFromJSON(memory.DefaultAllocator, arrowSchema, []string{
			`[{"id":1,"data":"a"},{"id":2,"data":null},{"id":3,"data":"b"},{"id":4,"data":"b"}]`,
		})
		require.NoError(t, err)
		defer data.Release()
		tbl, err = tbl.Append(ctx, array.NewTableReader(data, -1), nil)
		require.NoError(t, err)

		mem := memory.NewCheckedAllocator(memory.NewGoAllocator())
		checkedCtx := compute.WithAllocator(ctx, mem)
		var sawRows atomic.Int64
		matcher := func(mctx context.Context, batch arrow.RecordBatch) (*array.Boolean, error) {
			sawRows.Add(batch.NumRows())
			ids := batch.Column(batch.Schema().FieldIndices("id")[0]).(*array.Int64)
			bldr := array.NewBooleanBuilder(compute.GetAllocator(mctx))
			defer bldr.Release()
			for i := range int(batch.NumRows()) {
				bldr.Append(ids.Value(i) == 3)
			}

			return bldr.NewBooleanArray(), nil
		}

		// Filter data == "b" matches only rows 3 and 4; row 1 evaluates to false and row 2 evaluates to NULL.
		tbl, err = tbl.Delete(checkedCtx, iceberg.EqualTo(iceberg.Reference("data"), "b"), nil,
			table.WithDeleteRowMatcher([]string{"id", "data"}, matcher))
		require.NoError(t, err)
		assert.Equal(t, int64(2), sawRows.Load())
		assert.Equal(t, []int64{1, 2, 4}, idsInTable(t, tbl))
		mem.AssertSize(t, 0)
	})
	t.Run("copy-on-write multi-file mixed deletes only matched file and leaves untouched file intact", func(t *testing.T) {
		tbl := newDeleteModeTestTable(t, table.WriteModeCopyOnWrite, "2")
		location := strings.TrimPrefix(tbl.Location(), "file://")
		for _, rowJSON := range []string{
			`[{"id":1,"data":"a"},{"id":2,"data":"a"}]`,
			`[{"id":3,"data":"b"},{"id":4,"data":"b"}]`,
		} {
			data, err := array.TableFromJSON(memory.DefaultAllocator, arrowSchema, []string{rowJSON})
			require.NoError(t, err)
			tbl, err = tbl.Append(ctx, array.NewTableReader(data, -1), nil)
			data.Release()
			require.NoError(t, err)
		}

		beforeTasks, err := tbl.Scan().PlanFiles(ctx)
		require.NoError(t, err)
		require.Len(t, beforeTasks, 2)
		beforePaths := map[string]struct{}{
			beforeTasks[0].File.FilePath(): {},
			beforeTasks[1].File.FilePath(): {},
		}
		beforeParquet, err := filepath.Glob(filepath.Join(location, "data", "*.parquet"))
		require.NoError(t, err)

		mem := memory.NewCheckedAllocator(memory.NewGoAllocator())
		checkedCtx := compute.WithAllocator(ctx, mem)
		matcher := func(mctx context.Context, batch arrow.RecordBatch) (*array.Boolean, error) {
			ids := batch.Column(batch.Schema().FieldIndices("id")[0]).(*array.Int64)
			bldr := array.NewBooleanBuilder(compute.GetAllocator(mctx))
			defer bldr.Release()
			for i := range int(batch.NumRows()) {
				bldr.Append(ids.Value(i) == 3)
			}

			return bldr.NewBooleanArray(), nil
		}

		tbl, err = tbl.Delete(checkedCtx, iceberg.AlwaysTrue{}, nil,
			table.WithDeleteRowMatcher([]string{"id"}, matcher))
		require.NoError(t, err)
		assert.Equal(t, []int64{1, 2, 4}, idsInTable(t, tbl))
		mem.AssertSize(t, 0)

		afterTasks, err := tbl.Scan().PlanFiles(ctx)
		require.NoError(t, err)
		require.Len(t, afterTasks, 2)
		var preserved int
		for _, task := range afterTasks {
			if _, ok := beforePaths[task.File.FilePath()]; ok {
				preserved++
			}
		}
		assert.Equal(t, 1, preserved, "unchanged file must remain in the new snapshot without being rewritten")

		afterParquet, err := filepath.Glob(filepath.Join(location, "data", "*.parquet"))
		require.NoError(t, err)
		assert.Len(t, afterParquet, len(beforeParquet)+1, "skipped file output must be removed, leaving only one new rewritten file")
	})
	t.Run("copy-on-write v3 row matcher preserves row lineage and trims lineage columns from matcher batch", func(t *testing.T) {
		tbl := newDeleteModeTestTable(t, table.WriteModeCopyOnWrite, "3")
		batch1, err := array.TableFromJSON(memory.DefaultAllocator, arrowSchema, []string{
			`[{"id":1,"data":"a"},{"id":2,"data":"a"}]`,
		})
		require.NoError(t, err)
		defer batch1.Release()
		tbl, err = tbl.Append(ctx, array.NewTableReader(batch1, -1), nil)
		require.NoError(t, err)
		seq1 := tbl.CurrentSnapshot().SequenceNumber

		batch2, err := array.TableFromJSON(memory.DefaultAllocator, arrowSchema, []string{
			`[{"id":3,"data":"b"},{"id":4,"data":"b"}]`,
		})
		require.NoError(t, err)
		defer batch2.Release()
		tbl, err = tbl.Append(ctx, array.NewTableReader(batch2, -1), nil)
		require.NoError(t, err)
		seq2 := tbl.CurrentSnapshot().SequenceNumber
		require.Greater(t, seq2, seq1)

		mem := memory.NewCheckedAllocator(memory.NewGoAllocator())
		checkedCtx := compute.WithAllocator(ctx, mem)
		matcher := func(mctx context.Context, batch arrow.RecordBatch) (*array.Boolean, error) {
			if len(batch.Schema().FieldIndices(iceberg.RowIDColumnName)) > 0 {
				return nil, errors.New("matcher batch must not expose _row_id")
			}
			if len(batch.Schema().FieldIndices(iceberg.LastUpdatedSequenceNumberColumnName)) > 0 {
				return nil, errors.New("matcher batch must not expose _last_updated_sequence_number")
			}
			ids := batch.Column(batch.Schema().FieldIndices("id")[0]).(*array.Int64)
			bldr := array.NewBooleanBuilder(compute.GetAllocator(mctx))
			defer bldr.Release()
			for i := range int(batch.NumRows()) {
				id := ids.Value(i)
				if id == 2 {
					bldr.Append(true)

					continue
				}
				if id == 3 {
					bldr.Append(true)

					continue
				}
				bldr.Append(false)
			}

			return bldr.NewBooleanArray(), nil
		}

		// AlwaysTrue exercises classifyAllDataFiles populating fileSeqByPath for v3 row lineage synthesis.
		tbl, err = tbl.Delete(checkedCtx, iceberg.AlwaysTrue{}, nil,
			table.WithDeleteRowMatcher([]string{"id"}, matcher))
		require.NoError(t, err)
		mem.AssertSize(t, 0)
		require.Greater(t, tbl.CurrentSnapshot().SequenceNumber, seq2)

		_, itr, err := tbl.Scan(table.WithRowLineage()).ToArrowRecords(ctx)
		require.NoError(t, err)
		gotRowID := make(map[int64]int64)
		gotSeq := make(map[int64]int64)
		for rec, err := range itr {
			require.NoError(t, err)
			idCol := rec.Column(rec.Schema().FieldIndices("id")[0]).(*array.Int64)
			rowIDCol := rec.Column(rec.Schema().FieldIndices(iceberg.RowIDColumnName)[0]).(*array.Int64)
			seqCol := rec.Column(rec.Schema().FieldIndices(iceberg.LastUpdatedSequenceNumberColumnName)[0]).(*array.Int64)
			for i := range int(rec.NumRows()) {
				require.False(t, seqCol.IsNull(i))
				id := idCol.Value(i)
				gotRowID[id] = rowIDCol.Value(i)
				gotSeq[id] = seqCol.Value(i)
			}
			rec.Release()
		}

		assert.Equal(t, map[int64]int64{1: 0, 4: 3}, gotRowID)
		assert.Equal(t, map[int64]int64{1: seq1, 4: seq2}, gotSeq)
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
	t.Run("releases arrow memory across delete modes", func(t *testing.T) {
		for _, tc := range []struct {
			name    string
			version string
			mode    string
		}{
			{name: "v1_cow", version: "1", mode: table.WriteModeCopyOnWrite},
			{name: "v2_cow", version: "2", mode: table.WriteModeCopyOnWrite},
			{name: "v3_cow", version: "3", mode: table.WriteModeCopyOnWrite},
			{name: "v2_mor", version: "2", mode: table.WriteModeMergeOnRead},
			{name: "v3_mor", version: "3", mode: table.WriteModeMergeOnRead},
		} {
			t.Run(tc.name, func(t *testing.T) {
				tbl := newDeleteModeTestTable(t, tc.mode, tc.version)
				data, err := array.TableFromJSON(memory.DefaultAllocator, arrowSchema, []string{
					`[{"id":1,"data":"a"},{"id":2,"data":"a"},{"id":3,"data":"b"},{"id":4,"data":"b"}]`,
				})
				require.NoError(t, err)
				defer data.Release()

				tbl, err = tbl.Append(ctx, array.NewTableReader(data, -1), nil)
				require.NoError(t, err)

				alloc := memory.NewCheckedAllocator(memory.DefaultAllocator)
				deleteCtx := compute.WithAllocator(ctx, alloc)
				matcher := func(callCtx context.Context, batch arrow.RecordBatch) (*array.Boolean, error) {
					ids := batch.Column(batch.Schema().FieldIndices("id")[0]).(*array.Int64)
					bldr := array.NewBooleanBuilder(compute.GetAllocator(callCtx))
					defer bldr.Release()
					for i := range int(batch.NumRows()) {
						bldr.Append(ids.Value(i)%2 == 0)
					}

					return bldr.NewBooleanArray(), nil
				}

				tbl, err = tbl.Delete(deleteCtx, iceberg.AlwaysTrue{}, nil,
					table.WithDeleteRowMatcher([]string{"id"}, matcher))
				require.NoError(t, err)
				alloc.AssertSize(t, 0)
				assert.Equal(t, []int64{1, 3}, idsInTable(t, tbl))
			})
		}
	})
	t.Run("CleanupStagedFiles removes uncommitted and conflicted files but preserves uncertain or committed files", func(t *testing.T) {
		tbl := newTable(t, "2")
		location := strings.TrimPrefix(tbl.Location(), "file://")
		listDataFiles := func() []string {
			matches, err := filepath.Glob(filepath.Join(location, "data", "*"))
			require.NoError(t, err)

			return matches
		}
		listAvroFiles := func() []string {
			matches, err := filepath.Glob(filepath.Join(location, "metadata", "*.avro"))
			require.NoError(t, err)

			return matches
		}
		beforeData := listDataFiles()
		beforeAvro := listAvroFiles()

		{
			// Pre-commit abort cleans up staged delete files, manifests, and manifest lists.
			txn := tbl.NewTransaction()
			require.NoError(t, txn.Delete(ctx, iceberg.EqualTo(iceberg.Reference("id"), int64(1)), nil))
			assert.Greater(t, len(listDataFiles()), len(beforeData))
			assert.Greater(t, len(listAvroFiles()), len(beforeAvro))
			require.NoError(t, txn.CleanupStagedFiles(ctx))
			assert.Equal(t, beforeData, listDataFiles())
			assert.Equal(t, beforeAvro, listAvroFiles())

			// Aborted transaction rejects further mutations and TableCommit without writing files.
			_, tcErr := txn.TableCommit()
			assert.ErrorContains(t, tcErr, "aborted")
			assert.ErrorContains(t, txn.Delete(ctx, iceberg.EqualTo(iceberg.Reference("id"), int64(2)), nil), "aborted")
			data, err := array.TableFromJSON(memory.DefaultAllocator, arrowSchema, []string{`[{"id":5,"data":"c"}]`})
			require.NoError(t, err)
			defer data.Release()
			assert.ErrorContains(t, txn.Append(ctx, array.NewTableReader(data, -1), nil), "aborted")
			assert.ErrorContains(t, txn.Overwrite(ctx, array.NewTableReader(data, -1), nil), "aborted")
			assert.Equal(t, beforeData, listDataFiles())
			assert.Equal(t, beforeAvro, listAvroFiles())
		}

		{
			// WriteEqualityDeletes staged files are tracked and removed on abort.
			eqSchema := arrow.NewSchema([]arrow.Field{
				{Name: "id", Type: arrow.PrimitiveTypes.Int64, Nullable: false},
			}, nil)
			eqTbl, err := array.TableFromJSON(memory.DefaultAllocator, eqSchema, []string{`[{"id":1}]`})
			require.NoError(t, err)
			defer eqTbl.Release()

			txn := tbl.NewTransaction()
			eqFiles, err := txn.WriteEqualityDeletes(ctx, []int{1}, array.IterFromReader(array.NewTableReader(eqTbl, -1)))
			require.NoError(t, err)
			require.NotEmpty(t, eqFiles)
			assert.Greater(t, len(listDataFiles()), len(beforeData))
			require.NoError(t, txn.CleanupStagedFiles(ctx))
			assert.Equal(t, beforeData, listDataFiles())
			_, err = txn.WriteEqualityDeletes(ctx, []int{1}, array.IterFromReader(array.NewTableReader(eqTbl, -1)))
			assert.ErrorContains(t, err, "aborted")
			assert.Equal(t, beforeData, listDataFiles())
		}

		{
			// Definite conflict (ErrCommitFailed) removes staged files, manifests, and manifest lists.
			conflictCat := &fixedCommitErrCatalog{err: table.ErrCommitFailed}
			conflictTbl := table.New(tbl.Identifier(), tbl.Metadata(), tbl.MetadataLocation(), tbl.FS, conflictCat)
			conflictTxn := conflictTbl.NewTransaction()
			require.NoError(t, conflictTxn.Delete(ctx, iceberg.EqualTo(iceberg.Reference("id"), int64(2)), nil))
			assert.Greater(t, len(listDataFiles()), len(beforeData))
			assert.Greater(t, len(listAvroFiles()), len(beforeAvro))
			_, err := conflictTxn.Commit(ctx)
			require.ErrorIs(t, err, table.ErrCommitFailed)
			require.NoError(t, conflictTxn.CleanupStagedFiles(ctx))
			assert.Equal(t, beforeData, listDataFiles())
			assert.Equal(t, beforeAvro, listAvroFiles())
		}

		var preservedData, preservedAvro []string
		{
			// Uncertain commit error (non-ErrCommitFailed) marks transaction committed so CleanupStagedFiles preserves all staged files.
			uncertainErr := errors.New("connection reset after write")
			uncertainCat := &fixedCommitErrCatalog{err: uncertainErr}
			uncertainTbl := table.New(tbl.Identifier(), tbl.Metadata(), tbl.MetadataLocation(), tbl.FS, uncertainCat)
			uncertainTxn := uncertainTbl.NewTransaction()
			require.NoError(t, uncertainTxn.Delete(ctx, iceberg.EqualTo(iceberg.Reference("id"), int64(3)), nil))
			preservedData = listDataFiles()
			preservedAvro = listAvroFiles()
			assert.Greater(t, len(preservedData), len(beforeData))
			assert.Greater(t, len(preservedAvro), len(beforeAvro))
			_, err := uncertainTxn.Commit(ctx)
			require.ErrorIs(t, err, uncertainErr)
			require.NoError(t, uncertainTxn.CleanupStagedFiles(ctx))
			assert.Equal(t, preservedData, listDataFiles())
			assert.Equal(t, preservedAvro, listAvroFiles())
			for _, path := range append(preservedData, preservedAvro...) {
				_, statErr := os.Stat(path)
				require.NoError(t, statErr)
			}
		}

		{
			// Real OCC retry that hits ErrCommitDiverged cleans up staged files (with retryable cleanup on cancelled ctx) and blocks TableCommit.
			retryLocation := filepath.ToSlash(t.TempDir())
			retrySchema := iceberg.NewSchema(0,
				iceberg.NestedField{ID: 1, Name: "id", Type: iceberg.PrimitiveTypes.Int64, Required: true},
				iceberg.NestedField{ID: 2, Name: "data", Type: iceberg.PrimitiveTypes.String, Required: false},
			)
			retryMeta, err := table.NewMetadata(retrySchema, iceberg.UnpartitionedSpec, table.UnsortedSortOrder, retryLocation, iceberg.Properties{
				table.PropertyFormatVersion: "2",
				table.WriteDeleteModeKey:    table.WriteModeCopyOnWrite,
				table.CommitNumRetriesKey:   "2",
			})
			require.NoError(t, err)
			retryMetaLoc := retryLocation + "/metadata/v1.metadata.json"
			ctxErr := errors.New("canceled fs lookup")
			fsF := func(c context.Context) (iceio.IO, error) {
				if c.Err() != nil {
					return nil, ctxErr
				}

				return iceio.LocalFS{}, nil
			}
			cat := &concurrentTestCatalog{metadata: retryMeta, location: retryMetaLoc, fsF: fsF}
			retryTbl := table.New(table.Identifier{"db", "diverged"}, retryMeta, retryMetaLoc, fsF, cat)

			initData, err := array.TableFromJSON(memory.DefaultAllocator, arrowSchema, []string{
				`[{"id":1,"data":"a"},{"id":2,"data":"a"}]`,
			})
			require.NoError(t, err)
			defer initData.Release()
			retryTbl, err = retryTbl.Append(ctx, array.NewTableReader(initData, -1), nil)
			require.NoError(t, err)

			// Stage a delete on the current snapshot, then concurrently drop the data file so OCC retry hits real ErrCommitDiverged.
			retryTxn := retryTbl.NewTransaction()
			require.NoError(t, retryTxn.Delete(ctx, iceberg.EqualTo(iceberg.Reference("id"), int64(1)), nil))
			_, err = retryTbl.Delete(ctx, iceberg.AlwaysTrue{}, nil)
			require.NoError(t, err)

			listRetryFiles := func() []string {
				dataMatches, gErr := filepath.Glob(filepath.Join(retryLocation, "data", "*"))
				require.NoError(t, gErr)
				avroMatches, gErr := filepath.Glob(filepath.Join(retryLocation, "metadata", "*.avro"))
				require.NoError(t, gErr)

				return append(dataMatches, avroMatches...)
			}
			beforeCleanup := listRetryFiles()
			_, err = retryTxn.Commit(ctx)
			require.ErrorIs(t, err, table.ErrCommitDiverged)
			_, tcErr := retryTxn.TableCommit()
			assert.ErrorContains(t, tcErr, "aborted")

			canceledCtx, cancel := context.WithCancel(ctx)
			cancel()
			require.ErrorIs(t, retryTxn.CleanupStagedFiles(canceledCtx), ctxErr)
			assert.Equal(t, beforeCleanup, listRetryFiles())
			require.NoError(t, retryTxn.CleanupStagedFiles(ctx))
			assert.Less(t, len(listRetryFiles()), len(beforeCleanup))
			_, err = retryTxn.Commit(ctx)
			assert.ErrorContains(t, err, "aborted")
		}

		{
			// OCC retry rebuilds the manifest list on attempt 1; CleanupStagedFiles removes data, manifest, attempt-0 list, and rebuilt attempt-1 list.
			occLocation := filepath.ToSlash(t.TempDir())
			occSchema := iceberg.NewSchema(0,
				iceberg.NestedField{ID: 1, Name: "id", Type: iceberg.PrimitiveTypes.Int64, Required: true},
				iceberg.NestedField{ID: 2, Name: "data", Type: iceberg.PrimitiveTypes.String, Required: false},
			)
			occMeta, err := table.NewMetadata(occSchema, iceberg.UnpartitionedSpec, table.UnsortedSortOrder, occLocation, iceberg.Properties{
				table.PropertyFormatVersion: "2",
				table.CommitNumRetriesKey:   "1",
			})
			require.NoError(t, err)
			occMetaLoc := occLocation + "/metadata/v1.metadata.json"
			fsF := func(context.Context) (iceio.IO, error) { return iceio.LocalFS{}, nil }
			baseCat := &concurrentTestCatalog{metadata: occMeta, location: occMetaLoc, fsF: fsF}
			occTbl := table.New(table.Identifier{"db", "occ_rebuilt"}, occMeta, occMetaLoc, fsF, baseCat)

			batch1, err := array.TableFromJSON(memory.DefaultAllocator, arrowSchema, []string{`[{"id":1,"data":"a"}]`})
			require.NoError(t, err)
			defer batch1.Release()
			occTbl, err = occTbl.Append(ctx, array.NewTableReader(batch1, -1), nil)
			require.NoError(t, err)

			// Stage an append on a catalog that always returns ErrCommitFailed on CommitTable but loads the latest table on LoadTable.
			failingCat := &alwaysConflictCatalog{loader: baseCat}
			conflictTbl := table.New(occTbl.Identifier(), occTbl.Metadata(), occTbl.MetadataLocation(), fsF, failingCat)
			batch2, err := array.TableFromJSON(memory.DefaultAllocator, arrowSchema, []string{`[{"id":2,"data":"b"}]`})
			require.NoError(t, err)
			defer batch2.Release()
			occTxn := conflictTbl.NewTransaction()
			require.NoError(t, occTxn.Append(ctx, array.NewTableReader(batch2, -1), nil))

			// Advance the catalog head concurrently so attempt 1 rebuilds the manifest list against the new parent.
			batch3, err := array.TableFromJSON(memory.DefaultAllocator, arrowSchema, []string{`[{"id":3,"data":"c"}]`})
			require.NoError(t, err)
			defer batch3.Release()
			_, err = occTbl.Append(ctx, array.NewTableReader(batch3, -1), nil)
			require.NoError(t, err)

			listOccData := func() []string {
				m, gErr := filepath.Glob(filepath.Join(occLocation, "data", "*"))
				require.NoError(t, gErr)

				return m
			}
			listOccAvro := func() []string {
				m, gErr := filepath.Glob(filepath.Join(occLocation, "metadata", "*.avro"))
				require.NoError(t, gErr)

				return m
			}
			avroBeforeRetry := listOccAvro()
			_, err = occTxn.Commit(ctx)
			require.ErrorIs(t, err, table.ErrCommitFailed)
			avroAfterRetry := listOccAvro()
			assert.NotEqual(t, avroBeforeRetry, avroAfterRetry, "retry must have written a rebuilt manifest list")

			dataBeforeCleanup := len(listOccData())
			require.NoError(t, occTxn.CleanupStagedFiles(ctx))
			assert.Equal(t, dataBeforeCleanup-1, len(listOccData()))
			assert.Len(t, listOccAvro(), 4, "manifest, attempt-0 manifest list, and attempt-1 rebuilt manifest list must all be removed")
		}
	})
}

type fixedCommitErrCatalog struct {
	err error
}

func (c *fixedCommitErrCatalog) LoadTable(context.Context, table.Identifier) (*table.Table, error) {
	return nil, errors.New("unexpected LoadTable")
}

func (c *fixedCommitErrCatalog) CommitTable(context.Context, table.Identifier, []table.Requirement, []table.Update) (table.Metadata, string, error) {
	return nil, "", c.err
}

type alwaysConflictCatalog struct {
	loader table.CatalogIO
}

func (c *alwaysConflictCatalog) LoadTable(ctx context.Context, ident table.Identifier) (*table.Table, error) {
	return c.loader.LoadTable(ctx, ident)
}

func (c *alwaysConflictCatalog) CommitTable(context.Context, table.Identifier, []table.Requirement, []table.Update) (table.Metadata, string, error) {
	return nil, "", table.ErrCommitFailed
}
