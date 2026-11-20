"""Relational probe benchmark on the fixture tables: scaled copies, the build from a connection, outputs."""

import csv
import json
from pathlib import Path

import pytest

from clinrag.eval import relational_bench as bench
from clinrag.question import StructuredFilter
from clinrag.sources.base import RelationalQuery
from clinrag.sources.duckdb_relational import DuckDBRelationalSource, build_database_from_connection
from clinrag.sources.icd_graph import IcdGraph
from clinrag.sources.reference import DemoRelationalSource
from clinrag.workload.schema import write_jsonl
from tests.test_classifier import workload  # noqa: F401  (module-scoped fixture workload)

FIXTURES = Path(__file__).parent / "fixtures"
CREATININE = 50912


def test_scaled_connection_replicates_with_new_ids():
    base = bench.table_rows(bench.scaled_connection(FIXTURES, 1))
    con = bench.scaled_connection(FIXTURES, 3)
    assert bench.table_rows(con) == {t: 3 * n for t, n in base.items()}
    hadm = con.execute("SELECT count(*), count(DISTINCT hadm_id) FROM admissions").fetchall()[0]
    assert hadm[0] == hadm[1] == 3 * base["admissions"]
    labs = con.execute("SELECT count(DISTINCT labevent_id) FROM labevents").fetchall()[0][0]
    assert labs == 3 * base["labevents"]
    # Same codes and values in every copy, so the graph and selectivities do not change.
    assert IcdGraph.from_connection(con).codes == IcdGraph.from_connection(bench.scaled_connection(FIXTURES, 1)).codes
    copy_two = con.execute(
        f"SELECT count(*) FROM labevents WHERE hadm_id >= 2 * {bench.ID_OFFSET}"
    ).fetchall()[0][0]
    assert copy_two == base["labevents"]
    with pytest.raises(ValueError):
        bench.scaled_connection(FIXTURES, 0)


def test_note_metadata_scales_with_the_tables():
    note_ids, hadm_ids = bench.note_metadata(FIXTURES, 2)
    assert len(note_ids) == len(set(note_ids)) == len(hadm_ids) == 16
    assert hadm_ids[8] == hadm_ids[0] + bench.ID_OFFSET


def test_build_from_scaled_connection_matches_reference(tmp_path):
    con = bench.scaled_connection(FIXTURES, 2)
    db = build_database_from_connection(con, tmp_path / "x2.duckdb")
    reference, duck = DemoRelationalSource(con), DuckDBRelationalSource(db)
    note_ids, hadm_ids = bench.note_metadata(FIXTURES, 2)
    reference.register_notes(note_ids, hadm_ids)
    duck.register_notes(note_ids, hadm_ids)
    graph = IcdGraph.from_connection(con)
    queries = [
        RelationalQuery(lab=StructuredFilter(CREATININE, ">", 2.0)),
        RelationalQuery(lab=StructuredFilter(CREATININE, ">", 1.0), icd_codes=tuple(graph.expand("renal_genitourinary"))),
        RelationalQuery(icd_codes=tuple(graph.expand("endocrine"))),
    ]
    for query in queries:
        assert duck.count_matches(query) == reference.count_matches(query)
        assert duck.matching_ids(query) == reference.matching_ids(query)
        assert duck.run(query).items == reference.run(query).items
    assert duck.count(queries[0]) == 8  # 4 rows per copy


def test_build_from_connection_needs_the_tables(tmp_path):
    import duckdb

    with pytest.raises(ValueError, match="no table admissions"):
        build_database_from_connection(duckdb.connect(), tmp_path / "x.duckdb")


def test_time_call_warms_up_and_takes_the_median():
    calls = []
    ticks = iter([0.0, 0.001, 0.0, 0.005, 0.0, 0.002])
    timing = bench.time_call(lambda: calls.append(1) or len(calls), 3, clock=lambda: next(ticks))
    assert len(calls) == 4  # one warm-up call
    assert timing.median_ms == pytest.approx(2.0)
    assert timing.result == 4


def test_bench_main_on_fixtures(workload, tmp_path):  # noqa: F811
    write_jsonl(tmp_path / "workload" / "dev.jsonl", workload.dev)
    out = bench.main([
        "--tables-dir", str(FIXTURES), "--notes-dir", str(FIXTURES), "--workload", str(tmp_path / "workload"),
        "--scales", "1", "3", "--repeats", "1", "--max-queries", "4", "--out", str(tmp_path / "bench"),
    ])
    with (out / "relational_bench.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert {(int(r["scale"]), r["source"], r["method"]) for r in rows} == {
        (s, src, m) for s in (1, 3) for src in bench.SOURCES for m in bench.METHODS
    }
    assert all(float(r["median_ms"]) > 0 for r in rows)
    summary = json.loads((out / "relational_bench.json").read_text(encoding="utf-8"))
    for scale in summary["scales"]:
        assert scale["count_abs_error_max"] == 0
        assert scale["result_mismatches"] == 0
        assert 0 < scale["queries"] <= 4
        assert scale["peak_rss_mb"] > 0
    assert [h["scale"] for h in summary["headline"]] == [1, 3]
    assert (out / "relational_bench.png").stat().st_size > 0
    assert not (out / "work").exists()  # scaled DuckDB files are deleted
