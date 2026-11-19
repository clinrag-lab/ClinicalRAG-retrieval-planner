# Relational store and ICD graph

`DuckDBRelationalSource` (`src/clinrag/sources/duckdb_relational.py`) answers the
planner's relational queries from a DuckDB file built once from the MIMIC-IV hosp
tables. `IcdGraph` (`src/clinrag/sources/icd_graph.py`) expands diagnosis
categories into the ICD codes present in the data. Both implement the interfaces
in `sources/base.py` unchanged.

```
python scripts/build_duckdb.py --tables-dir data/mimic-iv-demo --out data/clinrag.duckdb
python -m clinrag.eval.runner --sources duckdb --workload data/workload --split dev
```

`--sources duckdb` pairs it with the reference TF-IDF notes, `--sources duckdb_faiss`
with the FAISS notes; both read `--db` (default `data/clinrag.duckdb`).

## Design

The reference source (`DemoRelationalSource`) loads the CSVs into memory on every
start and does per-row work in every query: it joins admissions, and builds
`version || ':' || upper(trim(icd_code))` for every diagnosis row it scans. The
build script moves that work to build time.

| choice | why |
|---|---|
| explicit types: `icd_code VARCHAR`, `hadm_id BIGINT`, `icd_version INTEGER` | ICD-9 codes such as `0389` lose their leading zero if inferred as numbers; ids join without casts |
| codes trimmed and uppercased once, plus a stored `dx_key` (`"10:I509"`) | `RelationalQuery.icd_codes` are version-qualified (`qualify()`); queries compare keys instead of building a string per row, and the version can never be dropped |
| `labevents` keeps only rows a lab query can return (hadm_id in admissions, non-null valuenum) | every lab query had the same admissions join and null filters; done once. The build rejects duplicate admission ids, the one case where dropping the join would change row counts |
| `labevents` sorted by `(itemid, hadm_id)` | every lab query filters on one itemid (and per-admission questions on hadm_id); DuckDB's min/max zone maps skip the row groups of other items |
| `diagnoses` sorted by `dx_key`, admissions-only | code lookups read a contiguous range |
| primary keys on `admissions`, `d_labitems`, `d_icd_diagnoses`, and the in-memory `note_counts` | the small lookup tables used in joins and labels |
| only the columns the source reads; titles only for codes in the data | the demo file is 4.2 MB |
| file attached `READ_ONLY` to an in-memory connection | several processes (runner, calibration) can share one file; note metadata from `register_notes` (note_id and hadm_id, never text) stays in memory |
| every result fully fetched before the next statement; no DataFrame registration | DuckDB 2.0-safe; `register_notes` inserts with list parameters |

The build writes to a temporary file and renames it on success, and refuses to
overwrite an existing file without `--force`. A `build_info` table records a
schema version; the source refuses a file from another version.

`IcdGraph` now expands every category once at construction (13 categories over
1,474 demo codes in about 9 ms). `expand()` returns a copy and
`descendant_count()`, the planner's graph probe, is a dict lookup: 0.22 µs per
call instead of 658 µs. The codes returned for every category are unchanged.

## Equivalence guarantee

`DuckDBRelationalSource` follows the `RelationalQuery` semantics in `base.py`
exactly and matches `DemoRelationalSource` row for row: one row per lab event for
lab queries, one row per diagnosis row for codes-only queries, codes matched on
(icd_version, icd_code), `lab.hadm_id` restricting to one admission, and the same
item fields and text. `count`, `matching_ids`, `run`, `count_notes` and
`count_matches` all agree.

Evidence (`tests/test_duckdb_relational.py`):

- a grid of about 1,200 queries on the fixtures and on an extended copy (each lab
  item x every op x min/median/max thresholds x hadm_id unset/set/unknown x ICD
  code sets, plus codes-only per category); the extended copy adds ICD-9 codes, a
  leading-zero code, a lower-case code with spaces, and lab and diagnosis rows the
  queries must drop;
- edge cases: leading zero, empty results, unknown codes and items, the wrong ICD
  version, invalid queries, note probes before `register_notes`;
- the runner's per-question records with `--sources duckdb` equal those with the
  reference except for timings, on the fixtures and on the demo dev split
  (all 1,350 records: 6 routers x 225 questions; same plans, evidence, tokens,
  recall and fallbacks);
- the benchmark below re-checks every result against the reference at each scale.

## Benchmark: probe cost vs. data size

`python -m clinrag.eval.relational_bench` (command and method in
docs/experiments.md) replicates the demo tables up to 100 times and times each
method on 24 class C/D queries. Medians over queries:

| scale (lab rows) | source | count | matching_ids | count_notes | count_matches | run |
|---|---|---|---|---|---|---|
| x1 (0.11M) | reference | 1.09 | 1.71 | 1.62 | 2.71 | 2.59 |
| x1 (0.11M) | DuckDB store | 0.29 | 0.90 | 0.74 | 1.70 | 1.11 |
| x100 (10.8M) | reference | 10.08 | 16.53 | 14.05 | 13.67 | 55.42 |
| x100 (10.8M) | DuckDB store | 1.18 | 3.39 | 2.77 | 2.97 | 37.24 |

- At x100 every probe on the DuckDB store takes at most 3.4 ms (reference: up to
  16.5 ms); `count` is 8.9x faster, `count_matches` 3.9x.
- `count_matches`, the planner's one-query probe, grows from 1.7 to 3.0 ms over
  100x more data, and costs a median 8% of running the query at x100 (24% on the
  reference). Probing instead of running saves a median 34 ms per query at x100.
- `count()` equals the rows `run()` returns at every scale (exact by
  construction), and the two sources return identical results for every method.
- Building the file takes 1.2 s at x100 (112 MB); 0.1 s for the demo (4.2 MB).

M2 Pro, 10 threads, DuckDB 1.5.6; full results in `outputs/relational_bench/`.

End to end on the demo (runner, dev split, 6 routers, 3 repeats), the median
retrieval latency per question drops from 2.37 to 1.40 ms (median per-question
speedup 1.54x): 1.83x on class A (relational only), 1.55x on classes C and D,
and 1.0x on class B, which does not query the relational store. Planning
overhead on C and D, where the probes run, is 1.27x lower. The full run,
analyses included, takes 59 s instead of 82 s.

## ICD categories: definition choices

Hypertensive heart disease with heart failure (ICD-10 I11.0, I13.0, I13.2;
ICD-9 402.x1, 404.x1, 404.x3) is not in `heart_failure`, and hypertensive CKD
(ICD-10 I12, I13; ICD-9 403, 404) is not in `chronic_kidney_disease`. Both
versions exclude them, so this is a definition choice, not a mapping bug; heart
failure usually also carries its own I50/428 code. Including the combination
codes would add 22 demo admissions to heart failure.

Three blocks did cover a condition in one ICD version but not the other
(diabetes, COPD, hypothyroidism). Those fixes change the generated notes and gold
answers, so they are in a separate PR, "Fix ICD-9/ICD-10 version mismatches in
icd_categories"; this branch keeps the team's mapping.

## Known limitations

- **Demo scale.** The demo has 275 admissions. At that size every query costs
  about a millisecond and fixed per-call overhead dominates; the speedups only
  show with more data.
- **The x100 scale is a replicated demo.** Each copy has the same codes and
  values, so the selectivity of every query is constant across scales and the
  query plans see the same distributions; real data at that size would be more
  varied.
- **`run()` is dominated by Python above x10.** Most of its time is building the
  evidence items (text and token counts) in Python, which both sources share, so
  `run()` gains far less than the probes.
- **Memory.** The x100 benchmark peaked at about 3 GB, mostly the reference
  source's in-memory tables.
- **Generated notes come from the same tables.** The notes are generated from each
  admission's diagnoses and labs, so filters and note concepts are more
  correlated than in real discharge summaries.
- **No stored statistics.** The probes count exactly with SQL; there are no
  histograms or sampling. That is what keeps them exact, and the benchmark shows
  it stays cheap at these sizes; much larger tables would need an index on
  `(itemid, valuenum)` or sampling.
