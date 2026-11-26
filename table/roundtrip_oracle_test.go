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

//go:build integration

package table_test

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"maps"
	"math/big"
	"math/rand"
	"os"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/memory"
	"github.com/apache/iceberg-go"
	"github.com/apache/iceberg-go/catalog"
	"github.com/apache/iceberg-go/internal/recipe"
	"github.com/apache/iceberg-go/table"
)

// The round-trip oracle runs a random, seeded sequence of appends (from Go and
// from Spark), deletes and column adds against one merge-on-read table. It keeps the expected
// rows in memory, checks a Go scan after every op and a Spark read at the end.
// Opt-in: ORACLE_SEEDS=1,2,3 make integration-oracle. Replay a failure with its seed.

type oracleCol struct {
	name string
	typ  iceberg.PrimitiveType
}

func (c oracleCol) String() string { return c.name + " " + c.typ.String() }

var oracleTypes = []iceberg.PrimitiveType{
	iceberg.PrimitiveTypes.Int32, iceberg.PrimitiveTypes.Int64, iceberg.PrimitiveTypes.String,
	iceberg.PrimitiveTypes.Bool, iceberg.PrimitiveTypes.Float64, iceberg.PrimitiveTypes.Date,
}

var oracleStrings = []string{"", "a", "b", "it's", "ünï", strings.Repeat("x", 300)}

type oracle struct {
	rng    *rand.Rand
	cols   []oracleCol // cols[0] is the required int64 "id"
	rows   map[int64]map[string]any
	nextID int64
	ops    []string
}

func (o *oracle) value(t iceberg.PrimitiveType) any {
	if o.rng.Intn(5) == 0 {
		return nil
	}
	switch t {
	case iceberg.PrimitiveTypes.Int32:
		return int32(o.rng.Intn(20) - 10)
	case iceberg.PrimitiveTypes.Int64:
		return int64(o.rng.Intn(1000))
	case iceberg.PrimitiveTypes.String:
		return oracleStrings[o.rng.Intn(len(oracleStrings))]
	case iceberg.PrimitiveTypes.Bool:
		return o.rng.Intn(2) == 0
	case iceberg.PrimitiveTypes.Float64:
		return float64(o.rng.Intn(40)-20) / 4
	default: // date
		return time.Date(2020, 1, 1+o.rng.Intn(30), 0, 0, 0, 0, time.UTC).Format(time.DateOnly)
	}
}

func (o *oracle) newRows(n int) []map[string]any {
	out := make([]map[string]any, n)
	for i := range out {
		o.nextID++
		row := map[string]any{"id": o.nextID}
		for _, c := range o.cols[1:] {
			row[c.name] = o.value(c.typ)
		}
		out[i] = row
		o.rows[o.nextID] = row
	}

	return out
}

// randomFilter returns a delete filter and the same predicate applied to a model row.
func (o *oracle) randomFilter() (iceberg.BooleanExpression, func(map[string]any) bool, string) {
	c := o.cols[o.rng.Intn(len(o.cols))]
	if c.name == "id" {
		x := o.nextID/2 + 1
		return iceberg.LessThan(iceberg.Reference("id"), x),
			func(r map[string]any) bool { return r["id"].(int64) < x }, fmt.Sprintf("id < %d", x)
	}
	switch o.rng.Intn(3) {
	case 0:
		return iceberg.IsNull(iceberg.Reference(c.name)),
			func(r map[string]any) bool { return r[c.name] == nil }, c.name + " IS NULL"
	case 1:
		return iceberg.NotNull(iceberg.Reference(c.name)),
			func(r map[string]any) bool { return r[c.name] != nil }, c.name + " IS NOT NULL"
	}
	v := o.value(c.typ)
	for v == nil {
		v = o.value(c.typ)
	}
	match := func(r map[string]any) bool { return r[c.name] != nil && canon(r[c.name]) == canon(v) }
	desc := fmt.Sprintf("%s = %v", c.name, v)
	ref := iceberg.Reference(c.name)
	switch v := v.(type) {
	case int32:
		return iceberg.EqualTo(ref, v), match, desc
	case int64:
		return iceberg.EqualTo(ref, v), match, desc
	case string:
		if c.typ == iceberg.PrimitiveTypes.Date {
			d, _ := time.Parse(time.DateOnly, v)
			return iceberg.EqualTo(ref, iceberg.Date(d.Unix()/86400)), match, desc
		}
		return iceberg.EqualTo(ref, v), match, desc
	case bool:
		return iceberg.EqualTo(ref, v), match, desc
	default:
		return iceberg.EqualTo(ref, v.(float64)), match, desc
	}
}

func sqlLiteral(v any) string {
	switch v := v.(type) {
	case nil:
		return "NULL"
	case string:
		if _, err := time.Parse(time.DateOnly, v); err == nil {
			return "DATE '" + v + "'"
		}
		return "'" + strings.ReplaceAll(v, "'", `\'`) + "'"
	case float64:
		return "CAST(" + strconv.FormatFloat(v, 'f', -1, 64) + " AS DOUBLE)"
	default:
		return fmt.Sprint(v)
	}
}

// canon renders a value from the model, a Go scan or Spark JSON the same way.
func canon(v any) string {
	switch v := v.(type) {
	case nil:
		return "null"
	case string:
		return strconv.Quote(v)
	case bool:
		return strconv.FormatBool(v)
	case json.Number:
		r, _ := new(big.Rat).SetString(string(v))
		return r.RatString()
	case float64:
		return new(big.Rat).SetFloat64(v).RatString()
	default:
		r, _ := new(big.Rat).SetString(fmt.Sprint(v))
		return r.RatString()
	}
}

func (o *oracle) canonRows(rows []map[string]any) map[string]string {
	out := make(map[string]string, len(rows))
	for _, r := range rows {
		parts := make([]string, len(o.cols))
		for i, c := range o.cols {
			parts[i] = c.name + "=" + canon(r[c.name])
		}
		out[canon(r["id"])] = strings.Join(parts, " ")
	}

	return out
}

// diff returns "" when got matches the model, else the first few differences.
func (o *oracle) diff(got []map[string]any) string {
	want := o.canonRows(slices.Collect(func(yield func(map[string]any) bool) {
		for _, r := range o.rows {
			if !yield(r) {
				return
			}
		}
	}))
	have := o.canonRows(got)
	var out []string
	for id, w := range want {
		if h, ok := have[id]; !ok {
			out = append(out, "missing: "+w)
		} else if h != w {
			out = append(out, "want: "+w+"\n  got: "+h)
		}
	}
	for id, h := range have {
		if _, ok := want[id]; !ok {
			out = append(out, "extra: "+h)
		}
	}
	slices.Sort(out)
	if len(out) > 10 {
		out = append(out[:10], fmt.Sprintf("... %d more", len(out)-10))
	}

	return strings.Join(out, "\n")
}

func decodeRows(b []byte) ([]map[string]any, error) {
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.UseNumber()
	var rows []map[string]any
	err := dec.Decode(&rows)

	return rows, err
}

func goScan(ctx context.Context, tbl *table.Table) ([]map[string]any, error) {
	arrTbl, err := tbl.Scan().ToArrowTable(ctx)
	if err != nil {
		return nil, err
	}
	defer arrTbl.Release()
	rdr := array.NewTableReader(arrTbl, -1)
	defer rdr.Release()
	var rows []map[string]any
	for rdr.Next() {
		b, err := rdr.RecordBatch().MarshalJSON()
		if err != nil {
			return nil, err
		}
		batch, err := decodeRows(b)
		if err != nil {
			return nil, err
		}
		rows = append(rows, batch...)
	}

	return rows, rdr.Err()
}

// demuxDocker strips the 8-byte frame headers from a non-TTY docker exec stream.
func demuxDocker(b []byte) []byte {
	var out []byte
	for len(b) >= 8 && (b[0] == 1 || b[0] == 2) && b[1] == 0 && b[2] == 0 && b[3] == 0 {
		n := int(binary.BigEndian.Uint32(b[4:8]))
		if 8+n > len(b) {
			break
		}
		if b[0] == 1 {
			out = append(out, b[8:8+n]...)
		}
		b = b[8+n:]
	}

	return append(out, b...)
}

func (s *SparkIntegrationTestSuite) sparkRows(o *oracle, name string) ([]map[string]any, error) {
	cols := make([]string, len(o.cols))
	for i, c := range o.cols {
		cols[i] = "`" + c.name + "`"
	}
	sql := fmt.Sprintf("SELECT to_json(array_sort(collect_list(struct(%s)))) AS j FROM default.%s", strings.Join(cols, ", "), name)
	out, err := recipe.ExecuteSpark(s.T(), "./validation.py", "--sql", sql)
	if err != nil {
		return nil, err
	}
	for _, line := range strings.Split(string(demuxDocker([]byte(out))), "\n") {
		if strings.HasPrefix(line, "|[") {
			return decodeRows([]byte(strings.TrimRight(strings.TrimPrefix(line, "|"), " |")))
		}
	}

	return nil, fmt.Errorf("no JSON row in spark output:\n%s", out)
}

func (s *SparkIntegrationTestSuite) TestRoundTripOracle() {
	seeds := os.Getenv("ORACLE_SEEDS")
	if seeds == "" {
		s.T().Skip("set ORACLE_SEEDS (e.g. 1,2,3) to run the round-trip oracle")
	}
	steps := 12
	if v, err := strconv.Atoi(os.Getenv("ORACLE_STEPS")); err == nil {
		steps = v
	}
	for _, seedStr := range strings.Split(seeds, ",") {
		if strings.TrimSpace(seedStr) == "" {
			continue
		}
		seed, err := strconv.ParseInt(strings.TrimSpace(seedStr), 10, 64)
		s.Require().NoError(err)
		s.Run(fmt.Sprintf("seed=%d", seed), func() {
			if msg := s.runOracle(seed, steps); msg != "" {
				s.Failf("oracle mismatch", "replay with ORACLE_SEEDS=%d ORACLE_STEPS=%d\n%s", seed, steps, msg)
			}
		})
	}
}

// runOracle returns "" on success, else the op log and what went wrong.
func (s *SparkIntegrationTestSuite) runOracle(seed int64, steps int) string {
	o := &oracle{rng: rand.New(rand.NewSource(seed)), rows: map[int64]map[string]any{}}
	fail := func(format string, args ...any) string {
		return "ops:\n  " + strings.Join(o.ops, "\n  ") + "\n" + fmt.Sprintf(format, args...)
	}

	o.cols = []oracleCol{{"id", iceberg.PrimitiveTypes.Int64}}
	fields := []iceberg.NestedField{{ID: 1, Name: "id", Type: iceberg.PrimitiveTypes.Int64, Required: true}}
	for i := range 1 + o.rng.Intn(3) {
		c := oracleCol{fmt.Sprintf("c%d", i), oracleTypes[o.rng.Intn(len(oracleTypes))]}
		o.cols = append(o.cols, c)
		fields = append(fields, iceberg.NestedField{ID: i + 2, Name: c.name, Type: c.typ})
	}
	name := fmt.Sprintf("go_oracle_%d", seed)
	ident := catalog.ToIdentifier("default", name)
	_ = s.cat.DropTable(s.ctx, ident)
	tbl, err := s.cat.CreateTable(s.ctx, ident, iceberg.NewSchema(0, fields...),
		catalog.WithProperties(iceberg.Properties{
			table.PropertyFormatVersion:   "2",
			table.WriteDeleteModeKey:      table.WriteModeMergeOnRead, // what transfer uses
			table.ParquetRowGroupLimitKey: "5",
		}))
	if err != nil {
		return fail("create: %v", err)
	}
	o.ops = append(o.ops, fmt.Sprintf("create merge-on-read cols=%v", o.cols))

	for step := range steps {
		op := o.rng.Intn(100)
		switch {
		case step == 0 || op < 35:
			rows := o.newRows(1 + o.rng.Intn(20))
			o.ops = append(o.ops, fmt.Sprintf("go append %d rows", len(rows)))
			arrSchema, err := table.SchemaToArrowSchema(tbl.Schema(), nil, true, false)
			if err != nil {
				return fail("arrow schema: %v", err)
			}
			b, _ := json.Marshal(rows)
			arrTbl, err := array.TableFromJSON(memory.DefaultAllocator, arrSchema, []string{string(b)})
			if err != nil {
				return fail("build arrow: %v", err)
			}
			tbl, err = tbl.AppendTable(s.ctx, arrTbl, 1000, nil)
			arrTbl.Release()
			if err != nil {
				return fail("go append: %v", err)
			}
		case op < 50:
			rows := o.newRows(1 + o.rng.Intn(10))
			o.ops = append(o.ops, fmt.Sprintf("spark append %d rows", len(rows)))
			names := make([]string, len(o.cols))
			for i, c := range o.cols {
				names[i] = "`" + c.name + "`"
			}
			vals := make([]string, len(rows))
			for i, r := range rows {
				lits := make([]string, len(o.cols))
				for j, c := range o.cols {
					lits[j] = sqlLiteral(r[c.name])
				}
				vals[i] = "(" + strings.Join(lits, ", ") + ")"
			}
			sql := fmt.Sprintf("INSERT INTO default.%s (%s) VALUES %s", name, strings.Join(names, ", "), strings.Join(vals, ", "))
			if _, err := recipe.ExecuteSpark(s.T(), "./validation.py", "--sql", sql); err != nil {
				return fail("spark append: %v", err)
			}
			if tbl, err = s.cat.LoadTable(s.ctx, ident); err != nil {
				return fail("reload: %v", err)
			}
		case op < 85:
			filter, match, desc := o.randomFilter()
			o.ops = append(o.ops, "go delete where "+desc)
			if tbl, err = tbl.Delete(s.ctx, filter, nil); err != nil {
				return fail("go delete: %v", err)
			}
			maps.DeleteFunc(o.rows, func(_ int64, r map[string]any) bool { return match(r) })
		default:
			c := oracleCol{fmt.Sprintf("c%d", len(o.cols)-1), oracleTypes[o.rng.Intn(len(oracleTypes))]}
			o.ops = append(o.ops, fmt.Sprintf("add column %s %s", c.name, c.typ))
			tx := tbl.NewTransaction()
			if err := tx.UpdateSchema(true, false).AddColumn([]string{c.name}, c.typ, "", false, nil).Commit(); err != nil {
				return fail("add column: %v", err)
			}
			if tbl, err = tx.Commit(s.ctx); err != nil {
				return fail("add column commit: %v", err)
			}
			o.cols = append(o.cols, c)
		}

		got, err := goScan(s.ctx, tbl)
		if err != nil {
			return fail("go scan: %v", err)
		}
		if d := o.diff(got); d != "" {
			return fail("go scan differs from model:\n%s", d)
		}
	}

	got, err := s.sparkRows(o, name)
	if err != nil {
		return fail("spark read: %v", err)
	}
	if d := o.diff(got); d != "" {
		return fail("spark read differs from model:\n%s", d)
	}

	return ""
}
