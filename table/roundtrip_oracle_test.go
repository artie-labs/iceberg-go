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
	"cmp"
	"context"
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

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/memory"
	"github.com/apache/iceberg-go"
	"github.com/apache/iceberg-go/catalog"
	"github.com/apache/iceberg-go/internal/recipe"
	"github.com/apache/iceberg-go/table"
)

// The round-trip oracle runs a random, seeded sequence of appends (from Go and
// from Spark), deletes, merges (a filter delete + append in one transaction, or
// equality deletes + rows in one row delta) and column adds against one
// merge-on-read table. It keeps the expected rows in memory, checks a Go scan
// after every op and a Spark read at the end.
// Opt-in: ORACLE_SEEDS=2,3,4 make integration-oracle. Replay a failure with its seed.

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
	rng     *rand.Rand
	cols    []oracleCol // cols[0] is the required int64 "id"
	rows    map[int64]map[string]any
	nextID  int64
	written int              // rows ever written, old versions included
	dead    []map[string]any // old row versions removed by equality deletes that a later delete may still match
	eqKeys  int              // equality delete keys ever written
	ops     []string
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
	o.written += n
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

// upsert picks existing and new keys like a merge batch in base/lib/iceberg/merge, and returns the new row versions
// with the delete that removes the old ones. The filter is base's keyFilter (lookupFilter per column for a composite
// key): each key column IN the batch's values and inside their min/max. A composite key is (id, c0); its filter also
// names decoy rows, and a row matcher keeps only the batch's exact (id, c0) keys. The model is updated by key.
// deletes is the delete's predicate on a model row, for charging position deletes to dead rows.
func (o *oracle) upsert(composite bool) (
	rows []map[string]any, filter iceberg.BooleanExpression, opts []table.DeleteOption, deletes func(map[string]any) bool, desc string,
) {
	rows, picked, existing := o.pickRows(composite)
	desc = fmt.Sprintf("merge %d existing + %d new keys", len(picked), len(rows)-len(picked))

	if !composite {
		if len(picked) == 0 { // base skips the delete when no live row has a batch key
			return rows, nil, nil, nil, desc
		}
		ids := map[string]bool{}
		for _, r := range rows {
			ids[canon(r["id"])] = true
		}
		filter = o.lookupFilter(o.cols[0], rows)
		deletes = func(r map[string]any) bool { return ids[canon(r["id"])] }
		return rows, filter, nil, deletes, desc + " on id, filter " + filter.String()
	}

	keys := map[string]bool{}
	for _, r := range rows {
		keys[canon(r["id"])+"|"+canon(r["c0"])] = true
	}
	decoys := existing[len(picked):]
	decoys = decoys[:min(len(decoys), 2)]
	if len(picked)+len(decoys) == 0 {
		return rows, nil, nil, nil, desc
	}
	filterRows := slices.Clone(rows)
	for _, id := range decoys {
		filterRows = append(filterRows, o.rows[id])
	}
	filter = iceberg.NewAnd(o.lookupFilter(o.cols[0], filterRows), o.lookupFilter(o.cols[1], filterRows))
	match := func(_ context.Context, batch arrow.RecordBatch) (*array.Boolean, error) {
		b, err := batch.MarshalJSON()
		if err != nil {
			return nil, err
		}
		got, err := decodeRows(b)
		if err != nil {
			return nil, err
		}
		bldr := array.NewBooleanBuilder(memory.DefaultAllocator)
		defer bldr.Release()
		for _, r := range got {
			bldr.Append(keys[canon(r["id"])+"|"+canon(r["c0"])])
		}

		return bldr.NewBooleanArray(), nil
	}
	opts = []table.DeleteOption{table.WithDeleteRowMatcher([]string{"id", "c0"}, match)}
	deletes = func(r map[string]any) bool { return keys[canon(r["id"])+"|"+canon(r["c0"])] }

	return rows, filter, opts, deletes, desc + fmt.Sprintf(" on (id, c0), decoys %v, filter %s", decoys, filter)
}

// pickRows returns new versions of up to 5 random existing keys followed by 1-5 new rows, and updates the model.
// picked are the existing keys in rows, existing every live key it drew from (picked first). A composite key keeps c0
// and never draws a row with a null c0.
func (o *oracle) pickRows(composite bool) (rows []map[string]any, picked, existing []int64) {
	for _, id := range slices.Sorted(maps.Keys(o.rows)) {
		if !composite || o.rows[id]["c0"] != nil {
			existing = append(existing, id)
		}
	}
	o.rng.Shuffle(len(existing), func(i, j int) { existing[i], existing[j] = existing[j], existing[i] })
	picked = existing[:min(len(existing), o.rng.Intn(6))]
	o.written += len(picked)

	for _, id := range picked {
		row := map[string]any{"id": id}
		for _, c := range o.cols[1:] {
			row[c.name] = o.value(c.typ)
		}
		if composite {
			row["c0"] = o.rows[id]["c0"]
		}
		rows = append(rows, row)
		o.rows[id] = row
	}
	for _, row := range o.newRows(1 + o.rng.Intn(5)) {
		for composite && row["c0"] == nil {
			row["c0"] = o.value(o.cols[1].typ)
		}
		rows = append(rows, row)
	}

	return rows, picked, existing
}

// eqMerge picks a batch like upsert and makes some existing keys CDC deletes: the key is in the batch, the row is not.
// It returns every batch key, which base's equality path deletes whether or not it is live, and the rows to write. A
// composite key also gets decoys: a live, unpicked id with another c0, which must not delete that row.
func (o *oracle) eqMerge(composite bool) (keys, rows []map[string]any, desc string) {
	old := maps.Clone(o.rows)
	batch, picked, existing := o.pickRows(composite)
	for _, id := range picked {
		o.dead = append(o.dead, old[id])
	}
	keyCols := o.cols[:1]
	if composite {
		keyCols = o.cols[:2]
	}
	deleted := 0
	for i, r := range batch {
		k := map[string]any{}
		for _, c := range keyCols {
			k[c.name] = r[c.name]
		}
		keys = append(keys, k)
		if i < len(picked) && o.rng.Intn(3) == 0 {
			delete(o.rows, r["id"].(int64))
			o.written--
			deleted++
			continue
		}
		rows = append(rows, r)
	}

	var decoys []map[string]any
	if composite {
		for _, id := range existing[len(picked):min(len(existing), len(picked)+2)] {
			c0 := o.value(o.cols[1].typ)
			for c0 == nil || canon(c0) == canon(o.rows[id]["c0"]) {
				c0 = o.value(o.cols[1].typ)
			}
			decoys = append(decoys, map[string]any{"id": id, "c0": c0})
		}
		keys = append(keys, decoys...)
	}
	o.eqKeys += len(keys)

	return keys, rows, fmt.Sprintf("equality merge %d existing (%d deleted) + %d new keys on %v, decoys %v",
		len(picked), deleted, len(batch)-len(picked), keyCols, decoys)
}

// eqDelta commits like base's stageEqualityDeletes: one row delta with an equality delete for every key, and the rows.
func eqDelta(ctx context.Context, tbl *table.Table, keys, rows []map[string]any) (*table.Table, error) {
	arrSchema, err := table.SchemaToArrowSchema(tbl.Schema(), nil, true, false)
	if err != nil {
		return nil, err
	}
	var keyFields []arrow.Field
	var fieldIDs []int
	for i, f := range tbl.Schema().Fields() {
		if _, ok := keys[0][f.Name]; ok {
			keyFields = append(keyFields, arrSchema.Field(i))
			fieldIDs = append(fieldIDs, f.ID)
		}
	}
	keyRec, err := recordFromRows(arrow.NewSchema(keyFields, nil), keys)
	if err != nil {
		return nil, err
	}
	defer keyRec.Release()

	tx := tbl.NewTransaction()
	delFiles, err := tx.WriteEqualityDeletes(ctx, fieldIDs, func(yield func(arrow.RecordBatch, error) bool) {
		yield(keyRec, nil)
	})
	if err != nil {
		return nil, err
	}
	rd := tx.NewRowDelta(nil).AddDeletes(delFiles...)
	if len(rows) > 0 {
		rec, err := recordFromRows(arrSchema, rows)
		if err != nil {
			return nil, err
		}
		// WriteRecords releases rec.
		for df, err := range table.WriteRecords(ctx, tbl, arrSchema, func(yield func(arrow.RecordBatch, error) bool) { yield(rec, nil) }) {
			if err != nil {
				return nil, err
			}
			rd.AddRows(df)
		}
	}
	if err := rd.Commit(ctx); err != nil {
		return nil, err
	}

	return tx.Commit(ctx)
}

func recordFromRows(sc *arrow.Schema, rows []map[string]any) (arrow.RecordBatch, error) {
	b, err := json.Marshal(rows)
	if err != nil {
		return nil, err
	}
	rec, _, err := array.RecordFromJSON(memory.DefaultAllocator, sc, bytes.NewReader(b))

	return rec, err
}

func summaryInt(tbl *table.Table, key string) (int, error) {
	raw, ok := tbl.CurrentSnapshot().Summary.Properties[key]
	n, err := strconv.Atoi(raw)
	if !ok || err != nil {
		return 0, fmt.Errorf("%s is %q (present=%v)", key, raw, ok)
	}

	return n, nil
}

// lookupFilter is c IN the rows' values, and for ordered types also inside their min/max, like base's lookupFilter.
func (o *oracle) lookupFilter(c oracleCol, rows []map[string]any) iceberg.BooleanExpression {
	var vals []any
	for _, r := range rows {
		vals = append(vals, r[c.name])
	}
	slices.SortFunc(vals, func(a, b any) int {
		switch a := a.(type) {
		case int32:
			return cmp.Compare(a, b.(int32))
		case int64:
			return cmp.Compare(a, b.(int64))
		case string:
			return cmp.Compare(a, b.(string))
		}
		return 0
	})
	lits := make([]iceberg.Literal, len(vals))
	for i, v := range vals {
		switch v := v.(type) {
		case string:
			if c.typ == iceberg.PrimitiveTypes.Date {
				d, _ := time.Parse(time.DateOnly, v)
				lits[i] = iceberg.NewLiteral(iceberg.Date(d.Unix() / 86400))
			} else {
				lits[i] = iceberg.NewLiteral(v)
			}
		case int32:
			lits[i] = iceberg.NewLiteral(v)
		case int64:
			lits[i] = iceberg.NewLiteral(v)
		case bool:
			lits[i] = iceberg.NewLiteral(v)
		default:
			lits[i] = iceberg.NewLiteral(v.(float64))
		}
	}
	ref := iceberg.Reference(c.name)
	in := iceberg.SetPredicate(iceberg.OpIn, ref, lits)
	switch c.typ {
	case iceberg.PrimitiveTypes.Bool, iceberg.PrimitiveTypes.Float64:
		return in
	}

	return iceberg.NewAnd(in, iceberg.LiteralPredicate(iceberg.OpGTEQ, ref, lits[0]),
		iceberg.LiteralPredicate(iceberg.OpLTEQ, ref, lits[len(lits)-1]))
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

func (o *oracle) canonRow(r map[string]any) string {
	parts := make([]string, len(o.cols))
	for i, c := range o.cols {
		parts[i] = c.name + "=" + canon(r[c.name])
	}

	return strings.Join(parts, " ")
}

// diff returns "" when got matches the model, else the first few differences. A duplicated id lists every copy, so a
// stale copy next to the right one still shows.
func (o *oracle) diff(got []map[string]any) string {
	want := map[string]string{}
	for _, r := range o.rows {
		want[canon(r["id"])] = o.canonRow(r)
	}
	have := map[string][]string{}
	for _, r := range got {
		id := canon(r["id"])
		have[id] = append(have[id], o.canonRow(r))
	}
	ids := slices.Collect(maps.Keys(want))
	for id := range have {
		if _, ok := want[id]; !ok {
			ids = append(ids, id)
		}
	}

	var out []string
	for _, id := range ids {
		w, ok := want[id]
		h := have[id]
		switch {
		case len(h) > 1:
			if !ok {
				w = "none"
			}
			out = append(out, fmt.Sprintf("duplicate: id=%s %d rows, want: %s\n  got: %s", id, len(h), w, strings.Join(h, "\n  got: ")))
		case !ok:
			out = append(out, "extra: "+h[0])
		case len(h) == 0:
			out = append(out, "missing: "+w)
		case h[0] != w:
			out = append(out, "want: "+w+"\n  got: "+h[0])
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
	for _, line := range strings.Split(out, "\n") {
		if strings.HasPrefix(line, "|[") {
			return decodeRows([]byte(strings.TrimRight(strings.TrimPrefix(line, "|"), " |")))
		}
	}

	return nil, fmt.Errorf("no JSON row in spark output:\n%s", out)
}

// mergeDelete deletes like base's stageFilterDeletes: a v2 merge writes full file_path metrics on its position deletes
// only for this delete.
func mergeDelete(ctx context.Context, tx *table.Transaction, version string, filter iceberg.BooleanExpression, opts []table.DeleteOption) error {
	if version != "2" {
		return tx.Delete(ctx, filter, nil, opts...)
	}
	key := table.MetricsModeColumnConfPrefix + "." + iceberg.PositionalDeleteSchema.Field(0).Name
	if err := tx.SetProperties(iceberg.Properties{key: "full"}); err != nil {
		return err
	}
	if err := tx.Delete(ctx, filter, nil, opts...); err != nil {
		return err
	}

	return tx.RemoveProperties([]string{key})
}

func (s *SparkIntegrationTestSuite) TestRoundTripOracle() {
	seeds := os.Getenv("ORACLE_SEEDS")
	if seeds == "" {
		s.T().Skip("set ORACLE_SEEDS (e.g. 2,3,4) to run the round-trip oracle")
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
	name := fmt.Sprintf("go_oracle_%d_%d", seed, os.Getpid()) // the Docker stack is shared between runs
	ident := catalog.ToIdentifier("default", name)
	_ = s.cat.DropTable(s.ctx, ident)
	version := strconv.Itoa(2 + o.rng.Intn(2)) // v3 merges write deletion vectors
	tbl, err := s.cat.CreateTable(s.ctx, ident, iceberg.NewSchema(0, fields...),
		catalog.WithProperties(iceberg.Properties{
			table.PropertyFormatVersion:   version,
			table.WriteDeleteModeKey:      table.WriteModeMergeOnRead, // what transfer uses
			table.ParquetRowGroupLimitKey: "5",
		}))
	if err != nil {
		return fail("create: %v", err)
	}
	o.ops = append(o.ops, fmt.Sprintf("create %s v%s merge-on-read cols=%v", name, version, o.cols))

	prevDels := 0
	for step := range steps {
		written, live := o.written, len(o.rows)
		var deletes func(map[string]any) bool // the op's delete predicate, nil for an op that writes no position deletes
		eqOp := false
		op := o.rng.Intn(100)
		for version == "3" && op >= 30 && op < 45 { // no Spark appends on v3: redraw so the other ops keep their share
			op = o.rng.Intn(100)
		}
		switch {
		case step == 0 || op < 30:
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
		case op < 45: // never on v3: Spark 1.8 in the stack cannot write v3 (first-row-id is null)
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
		case op < 60:
			filter, match, desc := o.randomFilter()
			o.ops = append(o.ops, "go delete where "+desc)
			if tbl, err = tbl.Delete(s.ctx, filter, nil); err != nil {
				return fail("go delete: %v", err)
			}
			maps.DeleteFunc(o.rows, func(_ int64, r map[string]any) bool { return match(r) })
			deletes = match
		case op < 73:
			var rows []map[string]any
			var filter iceberg.BooleanExpression
			var opts []table.DeleteOption
			var desc string
			rows, filter, opts, deletes, desc = o.upsert(o.rng.Intn(2) == 0)
			o.ops = append(o.ops, "go "+desc)
			arrSchema, err := table.SchemaToArrowSchema(tbl.Schema(), nil, true, false)
			if err != nil {
				return fail("arrow schema: %v", err)
			}
			b, _ := json.Marshal(rows)
			arrTbl, err := array.TableFromJSON(memory.DefaultAllocator, arrSchema, []string{string(b)})
			if err != nil {
				return fail("build arrow: %v", err)
			}
			rdr := array.NewTableReader(arrTbl, -1)
			tx := tbl.NewTransaction()
			if filter != nil {
				err = mergeDelete(s.ctx, tx, version, filter, opts)
			}
			if err == nil {
				err = tx.Append(s.ctx, rdr, nil)
			}
			if err == nil {
				tbl, err = tx.Commit(s.ctx)
			}
			rdr.Release()
			arrTbl.Release()
			if err != nil {
				return fail("go merge: %v", err)
			}
		case op < 87:
			eqOp = true
			keys, rows, desc := o.eqMerge(o.rng.Intn(2) == 0 && o.cols[1].typ != iceberg.PrimitiveTypes.Float64)
			o.ops = append(o.ops, "go "+desc)
			if tbl, err = eqDelta(s.ctx, tbl, keys, rows); err != nil {
				return fail("go equality merge: %v", err)
			}
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
		// A repeated position delete leaves reads unchanged but inflates the total that production derives row counts
		// from. Dropping a file can lower the total, so this is a bound: an op adds at most one delete per row it removed.
		dels, err := summaryInt(tbl, "total-position-deletes")
		if err != nil {
			return fail("%v", err)
		}
		// Documented limitation (see positionDeletesForFilter): a delete's scan does not apply equality deletes, so a
		// dead row the delete matches may get a position delete too. Allow one per such row, then forget the row.
		removed := (o.written - written) - (len(o.rows) - live)
		allowed := 0
		if deletes != nil {
			o.dead = slices.DeleteFunc(o.dead, func(r map[string]any) bool {
				if deletes(r) {
					allowed++
					return true
				}
				return false
			})
		}
		if eqOp && dels != prevDels {
			return fail("total-position-deletes went %d -> %d on an equality merge", prevDels, dels)
		}
		if dels-prevDels > removed+allowed {
			return fail("total-position-deletes went %d -> %d, but the op removed %d rows and matched %d equality-deleted rows",
				prevDels, dels, removed, allowed)
		}
		eqDels, err := summaryInt(tbl, "total-equality-deletes")
		if err != nil && o.eqKeys > 0 { // missing before the first equality delete
			return fail("%v", err)
		}
		if eqDels != o.eqKeys {
			return fail("total-equality-deletes is %d, want %d", eqDels, o.eqKeys)
		}
		prevDels = dels
	}

	got, err := s.sparkRows(o, name)
	if err != nil {
		return fail("spark read: %v", err)
	}
	if d := o.diff(got); d != "" {
		return fail("spark read differs from model:\n%s", d)
	}
	_ = s.cat.DropTable(s.ctx, ident)

	return ""
}
