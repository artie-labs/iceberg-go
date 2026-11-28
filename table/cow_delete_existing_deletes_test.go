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
	"slices"
	"testing"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/memory"
	"github.com/apache/iceberg-go"
	"github.com/apache/iceberg-go/table"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// A copy-on-write delete that rewrites a file must read it with the delete files that already apply to it: the
// rewritten file replaces the original, so a row those deletes removed would otherwise come back.
func TestCopyOnWriteDeleteKeepsExistingDeletes(t *testing.T) {
	for _, version := range []int{2, 3} {
		tbl := newV2RowLineageTestTable(t)
		if version == 3 {
			tbl = newV3RowLineageTestTable(t)
		}
		tbl = appendIDs(t, tbl, `[{"id": 1, "data": "a"}, {"id": 2, "data": "b"}, {"id": 3, "data": "c"}, {"id": 4, "data": "d"}, {"id": 5, "data": "e"}]`)
		rowIDs := map[int64]int64{}
		if version == 3 {
			rowIDs = readRowIDsByID(t, t.Context(), tbl)
		}

		// A position delete on v2, a deletion vector on v3.
		tbl = setDeleteMode(t, tbl, table.WriteModeMergeOnRead)
		tbl, err := tbl.Delete(t.Context(), iceberg.EqualTo(iceberg.Reference("id"), int64(2)), nil)
		require.NoError(t, err, version)
		want := []int64{1, 3, 4, 5}
		if version == 2 {
			records, release := makeEqDeleteRecords(t, idKeyArrowSchema(t), `[{"id": 4}]`)
			tx := tbl.NewTransaction()
			require.NoError(t, tx.CommitEqualityDeletes(t.Context(), []int{1}, records, nil))
			tbl, err = tx.Commit(t.Context())
			release()
			require.NoError(t, err)
			want = []int64{1, 3, 5}
		}
		require.Equal(t, want, sortedIDs(t, tbl), version)

		// Deleting id=1 rewrites the file that holds every row.
		tbl = setDeleteMode(t, tbl, table.WriteModeCopyOnWrite)
		tbl, err = tbl.Delete(t.Context(), iceberg.EqualTo(iceberg.Reference("id"), int64(1)), nil)
		require.NoError(t, err, version)
		assert.Equal(t, want[1:], sortedIDs(t, tbl), version)
		if version == 3 {
			// The survivors keep their row ids through the rewrite.
			got := readRowIDsByID(t, t.Context(), tbl)
			assert.Equal(t, map[int64]int64{3: rowIDs[3], 4: rowIDs[4], 5: rowIDs[5]}, got)
		}
	}
}

func appendIDs(t *testing.T, tbl *table.Table, rows string) *table.Table {
	t.Helper()
	schema := arrow.NewSchema([]arrow.Field{
		{Name: "id", Type: arrow.PrimitiveTypes.Int64},
		{Name: "data", Type: arrow.BinaryTypes.String, Nullable: true},
	}, nil)
	data, err := array.TableFromJSON(memory.DefaultAllocator, schema, []string{rows})
	require.NoError(t, err)
	defer data.Release()
	tbl, err = tbl.Append(t.Context(), array.NewTableReader(data, -1), nil)
	require.NoError(t, err)

	return tbl
}

func setDeleteMode(t *testing.T, tbl *table.Table, mode string) *table.Table {
	t.Helper()
	tx := tbl.NewTransaction()
	require.NoError(t, tx.SetProperties(iceberg.Properties{table.WriteDeleteModeKey: mode}))
	tbl, err := tx.Commit(t.Context())
	require.NoError(t, err)

	return tbl
}

func sortedIDs(t *testing.T, tbl *table.Table) []int64 {
	t.Helper()
	ids := scanIDs(t, tbl)
	slices.Sort(ids)

	return ids
}
