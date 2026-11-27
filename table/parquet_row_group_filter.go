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

package table

import (
	"fmt"
	"slices"

	"github.com/apache/arrow-go/v18/parquet/file"
	"github.com/apache/iceberg-go"
	tblutils "github.com/apache/iceberg-go/table/internal"
)

// ParquetRowGroupFilter reports which row groups of one Parquet data file may
// hold rows that match a row filter. It makes the checks a scan makes to skip
// row groups: column min/max statistics, dictionary pages of columns whose
// data pages are all dictionary encoded, and bloom filters. Like a scan, it
// checks the values of an IN set only when there are at most 200 of them, and
// otherwise only their range. It is not safe for concurrent use.
type ParquetRowGroupFilter struct {
	checker   *tblutils.RowGroupChecker
	rowGroups int
}

// NewParquetRowGroupFilter returns the row group filter of rowFilter for the
// Parquet file that rdr reads. rowFilter is unbound and refers to columns of
// schema by name, as in [WithRowFilter]. schema is usually the table's current
// schema. fileSchema is the file's schema with the table's field IDs and the
// file's own types, as [ArrowSchemaToIcebergWithOptions] returns it for the
// file's Arrow schema. It must hold every column rowFilter refers to that the
// file has, and may leave out the rest. A filter column it lacks while the
// file has it is an error, since it would read as null and prune row groups
// that match.
//
// A column the file lacks is read as null, or as its initial default, as in a
// scan. A filter on a column with an initial default that the file lacks
// prunes nothing. Statistics and bloom filters are matched to the filter's
// columns by Parquet field ID, and dictionary pages are read through rdr.
func NewParquetRowGroupFilter(
	rdr *file.Reader,
	schema, fileSchema *iceberg.Schema,
	rowFilter iceberg.BooleanExpression,
	caseSensitive bool,
) (*ParquetRowGroupFilter, error) {
	if rowFilter == nil {
		rowFilter = iceberg.AlwaysTrue{}
	}
	bound, err := iceberg.BindExpr(schema, rowFilter, caseSensitive)
	if err != nil {
		return nil, err
	}
	filterIDs, err := iceberg.ExtractFieldIDs(bound)
	if err != nil {
		return nil, err
	}
	columns := rdr.MetaData().Schema
	for i := range columns.NumColumns() {
		id := int(columns.Column(i).SchemaNode().FieldID())
		if _, ok := fileSchema.FindFieldByID(id); !ok && slices.Contains(filterIDs, id) {
			return nil, fmt.Errorf("%w: file schema lacks filter field %d, which the Parquet file has",
				iceberg.ErrInvalidArgument, id)
		}
	}
	plan, err := compilePruningPlan(fileSchema, schema, bound, caseSensitive)
	if err != nil {
		return nil, err
	}

	// The statistics evaluator reads only the columns passed to it, which are
	// the leaf columns of the fields the file-bound filter refers to.
	var cols []int
	if plan.statsFilter != nil {
		fieldIDs, err := iceberg.ExtractFieldIDs(plan.statsFilter)
		if err != nil {
			return nil, err
		}
		for i := range columns.NumColumns() {
			if slices.Contains(fieldIDs, int(columns.Column(i).SchemaNode().FieldID())) {
				cols = append(cols, i)
			}
		}
	}

	return &ParquetRowGroupFilter{
		checker:   tblutils.NewRowGroupChecker(plan.rowGroupTester(), rdr, cols),
		rowGroups: rdr.NumRowGroups(),
	}, nil
}

// MightMatch reports whether row group rowGroup may hold a row that matches
// the filter. False means it holds none.
func (f *ParquetRowGroupFilter) MightMatch(rowGroup int) (bool, error) {
	if rowGroup < 0 || rowGroup >= f.rowGroups {
		return false, fmt.Errorf("%w: row group %d of %d", iceberg.ErrInvalidArgument, rowGroup, f.rowGroups)
	}

	return f.checker.Keep(rowGroup)
}
