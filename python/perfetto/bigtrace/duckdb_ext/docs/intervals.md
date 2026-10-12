# Interval operators (spec §5) in DuckDB

The interval family of the `perfetto` DuckDB extension. It covers every
operator in §5 of `docs/spec.md`. Where TP main already implements an operator
in PerfettoSQL Next (`INTERVAL INTERSECTION OF`, `INTERVAL FLATTEN`), the
DuckDB version reproduces TP's exact semantics and is tested row-for-row
against it.

```python
import duckdb
con = duckdb.connect(config={'allow_unsigned_extensions': 'true'})
con.execute("LOAD '/usr/local/google/home/zezeozue/bt_bench/perfetto_duckdb/perfetto.duckdb_extension'")
```

## Conventions

- **Operators are table macros:** `FROM pf_interval_<op>(...)`.
  - Relations are passed as strings: either a table or view name, or a
    parenthesized subquery such as `'(SELECT * FROM slice WHERE dur > 0)'`.
  - The macros build SQL and run it with `query()`. CTEs of the calling query
    are not visible inside, so use tables, views or inline subqueries.
- **`PER` is `per := ['trace_idx', 'utid']`, a VARCHAR list.** With
  multi-trace data, `trace_idx` must lead the list. NULL PER keys group
  together, as in `GROUP BY`.
- **Operand columns are attached as a STRUCT** named after the alias, so
  `f.freq` works just as in PerfettoSQL.
- **`agg_clause` is a DuckDB aggregate list passed as a string**, for example
  `'count(*) AS concurrency, arg_max(name, depth) AS deepest'`.
  - §3 vocabulary mapping: `COUNT(*)` → `count(*)`, `SUM/MIN/MAX/AVG` →
    same names.
  - `ARG_MAX(key, value)` → `arg_max(value, key)`, which returns `value` at
    the max `key`. `ARG_MIN` likewise.
  - `GROUP_CONCAT(x)` → `string_agg(x, ',')`.
  - Internally the kernel emits a segment → member-row mapping, and the
    macro joins the member rows back and runs your aggregates with a plain
    `GROUP BY`. Any DuckDB aggregate works.
- **Interval semantics** (TP's, everywhere):
  - A row with `dur > 0` covers `[ts, ts + dur)`. Touching intervals do not
    intersect.
  - `dur = 0` is a point. It intersects an interval covering it
    (`start <= ts < end`) or a point at the same `ts`.
  - Rows with NULL `ts` or `dur` cover no time and are skipped.
  - In operators backed by a native kernel, `ts < 0` or `dur < 0` raises
    `ts or dur is below zero`, as in TP. Filter `dur = -1` (incomplete
    slices) first.
- **Performance model.** Each kernel is a native aggregate (`pf_interval_*_agg`
  in `src/intervals.cc`). DuckDB's parallel hash aggregate partitions by the
  PER columns, and each group is solved with one sort plus sweep:
  O(n log n + output), with no cross joins.

## 5.1 Sources

### INTERVAL INTERSECTION OF — `pf_interval_intersection(rels, aliases, per := [])`

Takes N operands (2 to 15 or more). There is one output row per combination of
one row from each operand whose intersection is non-empty, within matching PER
groups.
- **Output:** `ts`, `dur`, the PER columns, then one STRUCT per alias holding
  that operand's full row, including its own `ts`, `dur` and PER copies.
- **Algorithm:** TP's narrowing, smallest operand first. Each step is a 2-way
  sort-sweep over overlapping sets.

```sql
-- INTERVAL INTERSECTION OF (cpu_freq AS f, cpu_idle AS i) PER cpu
--   |> AGGREGATE SUM(dur) AS ns GROUP BY cpu, f.freq, i.state;
SELECT cpu, f.freq, i.state, sum(dur) AS ns
FROM pf_interval_intersection(['cpu_freq', 'cpu_idle'], ['f', 'i'], per := ['trace_idx', 'cpu'])
GROUP BY ALL;
```

### INTERVAL UNION OF — `pf_interval_union(rels, aliases, per := [])`

Fragments wherever any operand covers.
- **Absent operands:** each fragment carries every operand's row, as a NULL
  struct where that operand is absent.
- **Fragment boundaries:** consecutive time with the same combination of live
  rows is one fragment, with boundaries wherever the combination changes. If an
  operand overlaps itself, there is one fragment per combination of its live
  rows.
- **Points:** point rows give zero-width fragments, following the FLATTEN rules
  below.

```sql
SELECT * FROM pf_interval_union(['cpu_freq', 'cpu_idle'], ['f', 'i'], per := ['trace_idx', 'cpu']);
```

### INTERVALS FROM EVENTS — `pf_intervals_from_events(rel, per := [], stop := NULL, keep_last_open := false, close_last_at := NULL)`

Each event opens an interval, closed by the next event in its lane (events
ordered by `ts`, ties in input order).
- **STOP:** with `stop`, the interval closes at the first stop event at or
  after `ts`, if that comes earlier.
- **Ties:** events are not coalesced, so equal timestamps give `dur = 0`.
- **The last open interval** is dropped by default.
  - `keep_last_open := true` (KEEPING LAST OPEN) keeps it with `dur = -1`.
  - `close_last_at := 'expr'` (CLOSING LAST AT) uses an SQL expression over
    the event row, for example a joined trace end.
- **Output:** the event's columns plus `dur`.

```sql
-- counter_leading_intervals!(counter) equivalent:
SELECT * FROM pf_intervals_from_events(
  '(SELECT c.*, b.end_ts FROM counter c JOIN trace_bounds b USING (trace_idx))',
  per := ['trace_idx', 'track_id'], close_last_at := 'end_ts');
```

## 5.2 Reshaping

### INTERVAL SPLIT — `pf_interval_split(rel, operand, alias, per := [])`

Each current row is cut at the operand's boundaries that fall inside it.
- **Payload:** every piece keeps the row's columns with new `ts`/`dur`, plus
  the covering operand row as a STRUCT, which is NULL in gaps.
- **Lossless:** the pieces tile the row. If operand rows overlap each other,
  a piece is repeated once per covering row.
- **Points:** an operand point inside the row adds a zero-width piece at that
  point.

```sql
-- Slices cut by thread state:
SELECT id, ts, dur, st.state FROM pf_interval_split('thread_slices', 'thread_state', 'st', per := ['trace_idx', 'utid']);
```

### INTERVAL SUBTRACT — `pf_interval_subtract(rel, operand, per := [])`

Removes the operand's merged coverage from each row. The surviving pieces
keep the row's payload.
- A point is removed if the coverage covers it.
- Operand points remove nothing from positive intervals.

### INTERVAL FILL GAPS WITH — `pf_interval_fill_gaps(rel, bg, per := [], bg_has_per := true)` (alias: `pf_interval_fill_within`)

Matches TP's `INTERVAL FILL GAPS WITH bg`:
- Every input row passes through unchanged (including overlapping rows and `dur = -1` rows, which cover to the end of `bg`).
- Each span of `bg` not covered by the input becomes a filler row carrying that `bg` row's payload matched by column name (stream columns absent from `bg` are NULL on fillers; `bg` columns absent from the stream are added, NULL on input rows).
- When `bg_has_per := true` (default), `bg` defines the lanes, so lanes with no input rows are still filled. When `bg_has_per := false`, every `bg` row applies to every lane of the stream.

```sql
SELECT * FROM pf_interval_fill_gaps('thread_state', '(SELECT trace_idx, utid, start_ts AS ts, end_ts - start_ts AS dur, ''unknown'' AS state FROM thread_lifetimes)', per := ['trace_idx', 'utid']);
```

### INTERVAL FLATTEN — `pf_interval_flatten(rel, aggs, per := [])` and `pf_interval_flatten_sum(rel, sums := [], per := [])`

Exactly TP main's `INTERVAL FLATTEN`:
- **Segments:** split at every distinct row start and every positive row end.
  Each segment aggregates the positive rows covering it. Gaps produce nothing,
  and adjacent equal segments are not merged.
- **Points:** points at a timestamp produce one extra zero-width segment. It
  aggregates those points plus the positive rows live at that instant
  (starting there: included; ending there: excluded).
- **Output:** `ts`, `dur`, the PER columns, then the aggregates.
- **`pf_interval_flatten`** takes any aggregate list.
- **`pf_interval_flatten_sum`** is the fast path for TP's own vocabulary,
  `COUNT(*)` and `SUM(col)`, computed in the sweep without expanding segment
  members.
  - Output columns are `count` and `sum_<col>`.
  - SUM ignores NULLs and is NULL when every contributing value is NULL.

```sql
-- INTERVAL FLATTEN PER cpu AGGREGATE COUNT(*) AS concurrency, ARG_MAX(depth, name) AS deepest;
SELECT * FROM pf_interval_flatten('slices_on_cpu', 'count(*) AS concurrency, arg_max(name, depth) AS deepest',
                                  per := ['trace_idx', 'cpu']);
-- TP vocabulary, fast path:
SELECT * FROM pf_interval_flatten_sum('(SELECT * FROM slice WHERE dur >= 0)', ['depth'], per := ['trace_idx', 'track_id']);
```

### INTERVAL MERGE OVERLAPPING — `pf_interval_merge_overlapping(rel, aggs := 'count(*) AS count', per := [])`

Coalesces overlapping and abutting rows into their coverage.
- **Points:** a point merges into an interval that covers or abuts it.
- **Output:** `ts`, `dur`, the PER columns, then the aggregates over the
  member rows.

### INTERVAL MERGE CONSECUTIVE BY — `pf_interval_merge_consecutive(rel, keys, aggs := 'count(*) AS count', per := [])`

Within each lane, in `(ts, dur)` order, merges runs of rows that agree on
`keys`. Each row must touch or overlap the previous row; a gap or a key change
starts a new run.
- **Output:** `ts`, `dur`, the PER columns, the key columns, then the
  aggregates.

```sql
SELECT * FROM pf_interval_merge_consecutive('thread_state', ['state'], 'count(*) AS n', per := ['trace_idx', 'utid']);
```

### INTERVAL SET BOUNDS — `pf_interval_set_bounds(rel, begin_at, end_at := NULL, for_dur := NULL)`

`ts := begin_at` and `dur := end_at - begin_at` (or `dur := for_dur`), all as
SQL expressions over the row. The new `ts` and `dur` are appended as the last
two columns.

### INTERVAL QUANTIZE — `pf_interval_quantize(rel, width, aligned_to := 'zero', bounds := NULL)`

Cuts rows at `origin + k * width`, and tags each piece with `bucket`, the start
timestamp of its bucket.
- **`aligned_to`** is one of:
  - `'zero'`
  - `'trace_start'`: `start_ts` from `bounds`, a `(trace_idx, start_ts)`
    relation, or the input's min `ts` per `trace_idx` if no `bounds` is given
  - an SQL expression
- **Points** keep `dur = 0`.
- **The bucket math** is floor-based, so it also works before the origin.
- PER is not needed, because buckets depend only on the origin.

```sql
SELECT bucket, state, sum(dur) FROM pf_interval_quantize('thread_state', 1000000, 'trace_start', bounds := 'trace_bounds') GROUP BY ALL;
```

## 5.3 Matching

`relationship` is one of `'overlapping'`, `'covering begin'`,
`'covering end'`, `'covering bounds'` or `'within bounds'`. In these
definitions, `a` is my row and `b` is the operand row:
- **covering begin / end:** `b` covers my `ts` (or my `ts + dur`) as an
  instant: `b.ts <= t < b.end`, or `b.ts = t` if `b` is a point.
- **covering bounds:** `b.ts <= a.ts AND a.end <= b.end`. If `a` is a point,
  this is the same as covering begin.
- **within bounds:** `a.ts <= b.ts AND b.end <= a.end`.
  - If `b` is a point: `a.ts <= b.ts < a.end`.
  - If `a` is a point: only a point at the same `ts`.
- **overlapping:** the intersection rule above.

### [LEFT] INTERVAL JOIN — `pf_interval_join(rel, operand, alias, relationship := 'overlapping', per := [], left_join := false)`

Attaches the operand row as STRUCT `alias`, with bounds unchanged and one
output row per match. With `left_join := true`, unmatched rows are kept with a
NULL struct. Implemented as a sort-sweep, never a cross join.

```sql
-- Thread state at each slice's start:
SELECT id, name, st.state FROM pf_interval_join('thread_slices', 'thread_state', 'st', 'covering begin',
                                                per := ['trace_idx', 'utid'], left_join := true);
```

### [LEFT] INTERVAL FIND — `pf_interval_find(rel, operand, alias, edge, direction, anchor := 'begin', within := NULL, per := [], left_join := false)`

Attaches the single nearest operand row whose start (`edge := 'starting'`) or
end (`'ending'`) is at or after (`direction := 'after'`) or at or before
(`'before'`) the anchor.
- **Anchor:** `'begin'` is `ts`, `'end'` is `ts + dur`, and `'bounds'` is the
  end for AFTER and the begin for BEFORE.
- **`within := d`** limits the search distance.
- **Ties:** the first operand row in `(key, ts, dur)` order in the search
  direction.
- **Implementation:** DuckDB ASOF JOIN on a unique HUGEINT key.
- **Not supported:** `anchor OF rel`. Join the anchor columns in first.

```sql
-- Next slice on the same track starting after each slice ends, at most 1 ms away:
SELECT id, nxt.id FROM pf_interval_find('slices', 'slices', 'nxt', 'starting', 'after', 'end',
                                        within := 1000000, per := ['trace_idx', 'track_id'], left_join := true);
```

### INTERVAL KEEP IF / DROP IF — `pf_interval_keep_if(rel, operand, relationship := 'overlapping', per := [])`, `pf_interval_drop_if(...)`

A temporal semijoin or antijoin against the operand's **merged coverage**:
overlapping and abutting operand rows are first merged into disjoint intervals
plus isolated points. Rows pass through unchanged.

## Testing

`python tests/test_intervals.py` runs three groups of checks:
1. **Brute-force references.** Every operator is compared with an independent
   brute-force Python reference, on hand-written edge cases and on 40
   randomized cases. The edge cases cover:
   - empty input
   - `dur = 0`
   - ties and duplicate rows
   - nested and abutting intervals
   - NULL `ts` / `dur`
   - NULL PER keys
2. **Errors** on negative `dur`.
3. **TP main (PerfettoSQL Next pipe syntax) as oracle, row by row** on 4 real
   traces:
   - `INTERVAL FLATTEN PER track_id AGGREGATE COUNT(*), SUM(depth)`, through
     both the fast path and the generic path
   - `INTERVAL INTERSECTION OF` 2-way (thread_state × thread slices) and 3-way
     (adding sched) `PER utid`
   - 50k–600k rows per trace, all identical

## Benchmarks (100 Android traces: 19.8M slices, 29.8M thread states, 22.8M sched slices, 29.0M counters)

| Operator | Output Rows | PerfettoSQL Next (1T) | DuckDB 1T | 1T Speedup (%) | DuckDB 32T | 32T Speedup vs TP 1T (%) | 32T Speedup vs TP 32-proc Wall (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `INTERVAL FLATTEN` (`COUNT, SUM`) | 29,848,019 | 12.15 s | **2.408 s** | **5.05× (+405%)** | **0.568 s** | **21.39× (+2,039%)** | **28.35× (+2,735%)** |
| `INTERVAL INTERSECTION OF` (2-way `PER utid`) | 11,907,457 | 7.15 s | **2.706 s** | **2.64× (+164%)** | **0.461 s** | **15.51× (+1,451%)** | **31.43× (+3,043%)** |
| `INTERVAL INTERSECTION OF` (3-way `PER utid`) | 8,883,213 | 7.59 s | **3.488 s** | **2.18× (+118%)** | **0.524 s** | **14.48× (+1,348%)** | **27.12× (+2,612%)** |
| `INTERVAL MERGE OVERLAPPING` (`PER track_id`) | 5,140,429 | 20.27 s | **0.867 s** | **23.38× (+2,238%)** | **0.162 s** | **125.12× (+12,412%)** | **78.95× (+7,795%)** |
| `INTERVALS FROM EVENTS` (counter intervals) | 29,019,702 | 6.75 s | **2.676 s** | **2.52× (+152%)** | **0.495 s** | **13.64× (+1,264%)** | **25.82× (+2,482%)** |

