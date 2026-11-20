"""Relational probe cost vs. data size: DuckDBRelationalSource vs. DemoRelationalSource.

Usage:
    python -m clinrag.eval.relational_bench --tables-dir data/mimic-iv-demo --notes-dir data/generated \\
        --workload data/workload --scales 1 10 50 100 --out outputs/relational_bench

Supports C1 (cheap cost estimates without stored statistics): the planner's
relational probes (count, matching_ids, count_notes, count_matches) must stay
cheap as the tables grow, and much cheaper than running the query.

For each scale S the demo hosp tables are replicated S times in memory, copy k
with hadm_id, subject_id and labevent_id shifted by k * offset, so every copy
is a new set of admissions with the same codes and values. The generated
notes' metadata (note_id, hadm_id; never text) is replicated the same way and
registered with both sources. Then:

- reference: DemoRelationalSource over the scaled tables in memory, as the
  runner loads the demo;
- duckdb:    DuckDBRelationalSource over a DuckDB file built from the same
  tables, written under --out and deleted after the scale (--keep-db keeps it).

Queries: the distinct relational queries of the class C and D questions of
--split (graph codes expanded), at most --max-queries, evenly spaced. Each
method is called once to warm up, then --repeats times; the per-query median
is kept, and the table reports the median (and p90) over queries.

Writes to --out:
    relational_bench.csv        scale, source, method: median and p90 latency over queries
    relational_bench_queries.csv one row per (scale, source, method, query)
    relational_bench.json       arguments, build times, file sizes, peak memory, probe accuracy, speedups
    relational_bench.png        median latency vs. scale, one line per method and source
"""

from __future__ import annotations

import argparse
import csv
import json
import resource
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from clinrag.question import RELATIONAL
from clinrag.sources.base import RelationalQuery, RelationalSource
from clinrag.sources.duckdb_relational import DuckDBRelationalSource, build_database_from_connection
from clinrag.sources.icd_graph import IcdGraph
from clinrag.sources.reference import DemoRelationalSource
from clinrag.workload.schema import WorkloadQuestion, read_jsonl
from clinrag.workload.tables import connect_tables, table_path

METHODS = ("count", "matching_ids", "count_notes", "count_matches", "run")
PROBES = ("count", "matching_ids", "count_notes", "count_matches")
SOURCES = ("reference", "duckdb")
TABLES = ("admissions", "diagnoses_icd", "labevents")
LOOKUPS = ("d_labitems", "d_icd_diagnoses")
# Shifts per copy; demo hadm_id < 3e7, subject_id < 1.1e7, labevent_id < 5e5.
ID_OFFSET = 100_000_000
LAB_ID_OFFSET = 1_000_000_000


# ---- scaled data ------------------------------------------------------------------------


def scaled_connection(tables_dir: Path, scale: int) -> duckdb.DuckDBPyConnection:
    """In-memory connection with the hosp tables replicated scale times (new ids per copy)."""
    if scale < 1:
        raise ValueError("scale must be at least 1")
    con = connect_tables(tables_dir, required=TABLES, optional=LOOKUPS)
    shifts = {
        "admissions": ("subject_id", "hadm_id"),
        "diagnoses_icd": ("subject_id", "hadm_id"),
        "labevents": ("subject_id", "hadm_id", "labevent_id"),
    }
    for table, columns in shifts.items():
        replace = ", ".join(
            f"{c} + k * {LAB_ID_OFFSET if c == 'labevent_id' else ID_OFFSET} AS {c}" for c in columns
        )
        con.execute(
            f"CREATE TABLE {table}_scaled AS SELECT t.* REPLACE ({replace}) "
            f"FROM (SELECT *, rowid AS _row FROM {table}) t, range({scale}) r(k) ORDER BY k, t._row"
        ).fetchall()
        con.execute(f"ALTER TABLE {table}_scaled DROP COLUMN _row").fetchall()
        con.execute(f"DROP TABLE {table}").fetchall()
        con.execute(f"ALTER TABLE {table}_scaled RENAME TO {table}").fetchall()
    return con


def note_metadata(notes_dir: Path, scale: int) -> tuple[list[str], list[int]]:
    """(note_ids, hadm_ids) of the generated notes, replicated like the tables."""
    path = table_path(Path(notes_dir), "discharge")
    if path is None:
        raise FileNotFoundError(f"discharge.csv(.gz) not found in {notes_dir}")
    frame = pd.read_csv(path, usecols=["note_id", "hadm_id"], dtype={"note_id": str})
    note_ids, hadm_ids = [], []
    for k in range(scale):
        note_ids += [f"{n}#{k}" for n in frame["note_id"]]
        hadm_ids += [int(h) + k * ID_OFFSET for h in frame["hadm_id"]]
    return note_ids, hadm_ids


def table_rows(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    return {t: int(con.execute(f"SELECT count(*) FROM {t}").fetchall()[0][0]) for t in TABLES}


# ---- queries ----------------------------------------------------------------------------


def bench_queries(questions: Sequence[WorkloadQuestion], graph: IcdGraph, max_queries: int) -> list[RelationalQuery]:
    """Distinct relational queries of the class C and D questions, evenly spaced, at most max_queries."""
    seen: dict[RelationalQuery, None] = {}
    for q in questions:
        spec = q.spec
        if spec.qclass not in ("C", "D") or RELATIONAL not in spec.required_sources:
            continue
        codes = tuple(graph.expand(spec.category)) if spec.category else ()
        seen.setdefault(RelationalQuery(lab=spec.relational_filter, icd_codes=codes), None)
    queries = list(seen)
    if len(queries) > max_queries:
        positions = np.linspace(0, len(queries) - 1, max_queries).round().astype(int)
        queries = [queries[i] for i in dict.fromkeys(positions)]
    return queries


def describe(query: RelationalQuery) -> str:
    lab = query.lab
    text = "" if lab is None else f"item {lab.itemid}" + ("" if lab.op is None else f" {lab.op} {lab.threshold:g}")
    if query.icd_codes:
        text += f"{' and ' if text else ''}{len(query.icd_codes)} ICD codes"
    return text


# ---- timing -----------------------------------------------------------------------------


@dataclass
class Timing:
    median_ms: float
    result: Any


def time_call(fn: Callable[[], Any], repeats: int, clock: Callable[[], float] = time.perf_counter) -> Timing:
    """One warm-up call, then repeats timed calls; the median in ms and the last result."""
    result = fn()
    times = []
    for _ in range(repeats):
        start = clock()
        result = fn()
        times.append((clock() - start) * 1000.0)
    return Timing(statistics.median(times), result)


def bench_source(
    source: RelationalSource, queries: Sequence[RelationalQuery], repeats: int
) -> list[dict[str, Any]]:
    """One row per (query, method): median ms and the result size."""
    rows = []
    for i, query in enumerate(queries):
        for method in METHODS:
            timing = time_call(lambda: getattr(source, method)(query), repeats)
            result = timing.result
            if method == "run":
                size = len(result.items)
            elif method == "matching_ids":
                size = len(result)
            elif method == "count_matches":
                size = result[0]
            else:
                size = int(result)
            rows.append({"query": i, "method": method, "median_ms": timing.median_ms, "result": size,
                         "full_result": result})
    return rows


# ---- one scale --------------------------------------------------------------------------


def run_scale(
    tables_dir: Path,
    notes_dir: Path,
    questions: Sequence[WorkloadQuestion],
    scale: int,
    work_dir: Path,
    repeats: int,
    max_queries: int,
    keep_db: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Per-query rows for both sources and a summary of the scale (build cost, probe accuracy)."""
    start = time.perf_counter()
    con = scaled_connection(tables_dir, scale)
    scale_s = time.perf_counter() - start
    rows_per_table = table_rows(con)
    note_ids, hadm_ids = note_metadata(notes_dir, scale)

    db_path = work_dir / f"clinrag_x{scale}.duckdb"
    start = time.perf_counter()
    build_database_from_connection(con, db_path, overwrite=True, source=f"{tables_dir} x{scale}")
    build_s = time.perf_counter() - start
    file_mb = db_path.stat().st_size / 1e6

    reference = DemoRelationalSource(con)
    reference.register_notes(note_ids, hadm_ids)
    duck = DuckDBRelationalSource(db_path)
    duck.register_notes(note_ids, hadm_ids)
    graph = IcdGraph.from_connection(con)
    queries = bench_queries(questions, graph, max_queries)
    if not queries:
        raise ValueError("no class C or D relational queries in the workload")

    per_query: list[dict[str, Any]] = []
    results: dict[str, dict[tuple[int, str], Any]] = {}
    for name, source in (("reference", reference), ("duckdb", duck)):
        rows = bench_source(source, queries, repeats)
        results[name] = {(r["query"], r["method"]): r.pop("full_result") for r in rows}
        per_query += [{"scale": scale, "source": name, **r, "query_text": describe(queries[r["query"]])} for r in rows]

    # count() predicts the rows run() returns exactly by construction; check it, and that both
    # sources agree on every result.
    count_errors = [
        abs(results[s][(i, "count")] - len(results[s][(i, "run")].items)) for s in SOURCES for i in range(len(queries))
    ]
    mismatches = sum(
        _comparable(results["reference"][key]) != _comparable(results["duckdb"][key]) for key in results["reference"]
    )
    duck.close()
    con.close()
    if not keep_db:
        db_path.unlink(missing_ok=True)

    summary = {
        "scale": scale,
        "rows": rows_per_table,
        "notes": len(note_ids),
        "queries": len(queries),
        "scale_tables_s": scale_s,
        "duckdb_build_s": build_s,
        "duckdb_file_mb": file_mb,
        "count_abs_error_max": max(count_errors),
        "result_mismatches": int(mismatches),
        # Process peak so far; scales run in ascending order, so this is the peak up to this scale.
        "peak_rss_mb": peak_rss_mb(),
    }
    return per_query, summary


def _comparable(result: Any) -> Any:
    if hasattr(result, "items"):
        return sorted(json.dumps(i, sort_keys=True) for i in result.items)
    return result


def peak_rss_mb() -> float:
    """Peak resident memory of this process so far, in MB (ru_maxrss: bytes on macOS, KB on Linux)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e6 if sys.platform == "darwin" else peak * 1024 / 1e6


# ---- summaries --------------------------------------------------------------------------


def aggregate(per_query: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Median and p90 over queries per (scale, source, method)."""
    groups: dict[tuple[int, str, str], list[float]] = {}
    for r in per_query:
        groups.setdefault((r["scale"], r["source"], r["method"]), []).append(r["median_ms"])
    rows = []
    def order(key: tuple[int, str, str]) -> tuple[int, int, int]:
        return key[0], SOURCES.index(key[1]), METHODS.index(key[2])

    for (scale, source, method), values in sorted(groups.items(), key=lambda kv: order(kv[0])):
        rows.append({
            "scale": scale,
            "source": source,
            "method": method,
            "queries": len(values),
            "median_ms": float(np.median(values)),
            "p90_ms": float(np.percentile(values, 90)),
            "mean_ms": float(np.mean(values)),
        })
    return rows


def headline(per_query: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per scale: speedups of duckdb over reference, and probe time saved vs. running the query.

    probe_saving_ms: median over queries of run() - count_matches() time, the
    planner's one-query probe, per source; probe_fraction: the same as a ratio.
    """
    by_key = {(r["scale"], r["source"], r["method"], r["query"]): r["median_ms"] for r in per_query}
    out = []
    for scale in sorted({r["scale"] for r in per_query}):
        queries = sorted({r["query"] for r in per_query if r["scale"] == scale})
        row: dict[str, Any] = {"scale": scale}
        for method in METHODS:
            ratios = [by_key[(scale, "reference", method, q)] / by_key[(scale, "duckdb", method, q)] for q in queries]
            row[f"speedup_{method}"] = float(np.median(ratios))
        for source in SOURCES:
            saved = [by_key[(scale, source, "run", q)] - by_key[(scale, source, "count_matches", q)] for q in queries]
            frac = [by_key[(scale, source, "count_matches", q)] / by_key[(scale, source, "run", q)] for q in queries]
            row[f"{source}_probe_saving_ms"] = float(np.median(saved))
            row[f"{source}_probe_fraction"] = float(np.median(frac))
        out.append(row)
    return out


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: Sequence[dict[str, Any]], path: Path) -> None:
    """Median latency vs. scale (log-log), colour per method, solid duckdb, dashed reference."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = dict(zip(METHODS, ("#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#d62728")))
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for source, style in (("reference", "--"), ("duckdb", "-")):
        for method in METHODS:
            pts = sorted((r["scale"], r["median_ms"]) for r in rows if r["source"] == source and r["method"] == method)
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, style, marker="o", color=colors[method], linewidth=2,
                        label=f"{method} ({'DuckDB store' if source == 'duckdb' else 'reference'})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    scales = sorted({r["scale"] for r in rows})
    ax.set_xticks(scales)
    ax.set_xticklabels([f"×{s}" for s in scales])
    ax.set_xlabel("data size (copies of the MIMIC-IV demo hosp tables)")
    ax.set_ylabel("median latency per query (ms, log)")
    top = max(scales)

    def slowest_probe(source: str) -> float:
        return max(r["median_ms"] for r in rows if r["scale"] == top and r["source"] == source and r["method"] in PROBES)

    ax.set_title(f"At ×{top} data, every relational probe takes ≤ {slowest_probe('duckdb'):.1f} ms on the DuckDB store\n"
                 f"vs. ≤ {slowest_probe('reference'):.1f} ms on the reference (medians over queries)", fontsize=11)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---- main -------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tables-dir", type=Path, default=Path("data/mimic-iv-demo"))
    parser.add_argument("--notes-dir", type=Path, default=Path("data/generated"))
    parser.add_argument("--workload", type=Path, default=Path("data/workload"))
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--scales", type=int, nargs="+", default=[1, 10, 50, 100])
    parser.add_argument("--repeats", type=int, default=5, help="timed calls per method and query (after one warm-up)")
    parser.add_argument("--max-queries", type=int, default=24)
    parser.add_argument("--out", type=Path, default=Path("outputs/relational_bench"))
    parser.add_argument("--keep-db", action="store_true", help="keep the scaled DuckDB files under --out/work")
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    work = args.out / "work"
    work.mkdir(exist_ok=True)
    questions = read_jsonl(args.workload / f"{args.split}.jsonl")

    per_query: list[dict[str, Any]] = []
    scales = []
    for scale in sorted(set(args.scales)):
        start = time.perf_counter()
        rows, summary = run_scale(args.tables_dir, args.notes_dir, questions, scale, work, args.repeats,
                                  args.max_queries, args.keep_db)
        per_query += rows
        scales.append(summary)
        print(f"x{scale}: {summary['rows']['labevents']} lab rows, {summary['queries']} queries, "
              f"build {summary['duckdb_build_s']:.1f} s, peak memory {summary['peak_rss_mb']:.0f} MB, "
              f"{time.perf_counter() - start:.1f} s total")
    if not args.keep_db and not any(work.iterdir()):
        work.rmdir()

    table = aggregate(per_query)
    heads = headline(per_query)
    write_csv(args.out / "relational_bench.csv", table)
    write_csv(args.out / "relational_bench_queries.csv", per_query)
    (args.out / "relational_bench.json").write_text(json.dumps({
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "duckdb_version": duckdb.__version__,
        "duckdb_threads": int(duckdb.connect().execute("SELECT current_setting('threads')").fetchall()[0][0]),
        "scales": scales,
        "headline": heads,
    }, indent=2) + "\n", encoding="utf-8")
    plot(table, args.out / "relational_bench.png")
    for row in heads:
        print(json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}))
    print(f"results: {args.out}")
    return args.out


if __name__ == "__main__":
    main()
