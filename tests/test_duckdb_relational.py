"""DuckDBRelationalSource: row-for-row equivalence with DemoRelationalSource, edge cases, wiring.

The database is built from the fixture tables into a temp dir. A second dataset
(extended) adds rows the fixture lacks: ICD-9 codes, one with a leading zero, a
code stored with stray whitespace and lower case, lab rows the lab queries must
drop (no hadm_id, no valuenum, an admission missing from admissions), and a
diagnosis of an admission missing from admissions.
"""

import csv
import itertools
import json
import shutil
from pathlib import Path

import pytest

from clinrag.eval import runner
from clinrag.question import StructuredFilter
from clinrag.sources.base import RelationalQuery, RelationalSource
from clinrag.sources.duckdb_relational import (
    DuckDBRelationalSource,
    build_database,
    build_duckdb_sources,
)
from clinrag.sources.icd_graph import IcdGraph
from clinrag.sources.reference import DemoRelationalSource, build_reference_sources
from clinrag.workload.icd_categories import CATEGORIES
from clinrag.workload.schema import write_jsonl
from tests.test_classifier import workload  # noqa: F401  (module-scoped fixture workload)

FIXTURES = Path(__file__).parent / "fixtures"
CREATININE = 50912
OPS = (None, "<", "<=", ">", ">=", "=", "!=")


def h(n):
    return 20000000 + n


def _append(path: Path, rows: list[list]) -> None:
    text = path.read_text(encoding="utf-8")
    with path.open("a", encoding="utf-8", newline="") as f:
        if text and not text.endswith("\n"):
            f.write("\n")
        csv.writer(f, lineterminator="\n").writerows(rows)


@pytest.fixture(scope="module")
def extended_dir(tmp_path_factory):
    """Fixture tables plus ICD-9 rows and rows the lab and diagnosis queries must drop."""
    d = tmp_path_factory.mktemp("extended")
    for path in FIXTURES.glob("*.csv"):
        shutil.copy(path, d / path.name)
    _append(d / "diagnoses_icd.csv", [
        [10000001, h(1), 4, "0389", 9],      # septicemia: ICD-9 with a leading zero
        [10000002, h(3), 3, "4280", 9],      # heart failure, ICD-9
        [10000003, h(4), 3, "5849", 9],      # acute kidney injury, ICD-9
        [10000003, h(5), 3, " i509 ", 10],   # needs trim and upper case
        [10000009, h(99), 1, "N186", 10],    # admission not in admissions
    ])
    _append(d / "d_icd_diagnoses.csv", [
        ["0389", 9, "Unspecified septicemia"],
        ["4280", 9, "Congestive heart failure, unspecified"],
    ])
    _append(d / "labevents.csv", [
        [21, 10000001, "", CREATININE, "2150-03-03 15:00:00", "9.9", 9.9, "mg/dL", ""],      # no hadm_id
        [22, 10000001, h(1), CREATININE, "2150-03-03 16:00:00", "___", "", "mg/dL", ""],     # no valuenum
        [23, 10000009, h(99), CREATININE, "2150-03-03 17:00:00", "8.0", 8.0, "mg/dL", ""],   # not in admissions
        [24, 10000001, h(1), 51006, "2150-03-03 18:00:00", "20", 20, "", ""],                # item without a label
        [25, 10000002, h(3), CREATININE, "2148-11-02 05:30:00", "1.1", 1.1, "mg/dL", ""],    # charttime tie
    ])
    return d


def _pair(tables_dir: Path, db_dir: Path):
    reference = build_reference_sources(tables_dir, tables_dir)
    duck = build_duckdb_sources(build_database(tables_dir, db_dir / "clinrag.duckdb"), tables_dir)
    return reference, duck


@pytest.fixture(scope="module")
def fixture_pair(tmp_path_factory):
    return _pair(FIXTURES, tmp_path_factory.mktemp("db"))


@pytest.fixture(scope="module")
def extended_pair(extended_dir, tmp_path_factory):
    return _pair(extended_dir, tmp_path_factory.mktemp("db_extended"))


def _lab_values(tables_dir: Path) -> dict[int, list[float]]:
    values: dict[int, list[float]] = {}
    with (tables_dir / "labevents.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["valuenum"]:
                values.setdefault(int(row["itemid"]), []).append(float(row["valuenum"]))
    return values


def query_grid(graph: IcdGraph, tables_dir: Path) -> list[RelationalQuery]:
    """Each lab itemid x ops x thresholds x hadm_id x code sets, plus codes-only per category.

    For every admission (and an unknown one), each lab item is also queried with
    agg latest and max, with and without a comparison and a code set.
    Thresholds are the item's min, median and max values, so every op meets ties.
    Code sets: none, three chapters, a block, an unknown code, and a known code
    string under the wrong ICD version.
    """
    code_sets = [()] + [tuple(graph.expand(c)) for c in ("circulatory", "renal_genitourinary", "endocrine", "heart_failure")]
    code_sets += [("10:ZZZ99",), ("9:" + graph.expand("circulatory")[0].split(":", 1)[1],)]
    hadms = [None, h(2), h(99)]
    queries = []
    for itemid, values in sorted(_lab_values(tables_dir).items()):
        values = sorted(set(values))
        thresholds = sorted({values[0], values[len(values) // 2], values[-1]})
        conditions = [(None, None)] + [(op, t) for op in OPS[1:] for t in thresholds]
        for (op, t), hadm, codes in itertools.product(conditions, hadms, code_sets):
            queries.append(RelationalQuery(lab=StructuredFilter(itemid, op, t, hadm), icd_codes=codes))
        aggregated = [(None, None), (">", thresholds[0]), ("<=", thresholds[len(thresholds) // 2])]
        for (op, t), n, agg, codes in itertools.product(aggregated, (*range(1, 9), 99), ("latest", "max"), code_sets[:2]):
            queries.append(RelationalQuery(lab=StructuredFilter(itemid, op, t, h(n), agg=agg), icd_codes=codes))
    queries.append(RelationalQuery(lab=StructuredFilter(123456)))  # unknown itemid
    queries += [RelationalQuery(icd_codes=tuple(graph.expand(c))) for c in CATEGORIES if graph.expand(c)]
    queries += [RelationalQuery(icd_codes=codes) for codes in code_sets if codes]
    return queries


def test_agg_returns_the_same_single_row_as_the_reference(extended_pair):
    """latest and max, including the charttime tie on admission 03 (labevent 7 and 25 share a time)."""
    reference, duck = extended_pair
    for agg, expected in (("latest", 1.1), ("max", 1.1)):
        query = RelationalQuery(lab=StructuredFilter(CREATININE, hadm_id=h(3), agg=agg))
        rows = duck.relational.run(query).items
        assert rows == reference.relational.run(query).items
        assert [r["valuenum"] for r in rows] == [expected] and duck.relational.count(query) == 1
    latest = RelationalQuery(lab=StructuredFilter(CREATININE, hadm_id=h(2), agg="latest"))
    highest = RelationalQuery(lab=StructuredFilter(CREATININE, hadm_id=h(2), agg="max"))
    assert [r["valuenum"] for r in duck.relational.run(latest).items] == [4.9]
    assert [r["valuenum"] for r in duck.relational.run(highest).items] == [6.4]
    row = duck.relational.run(latest).items[0]
    assert (row["label"], row["unit"], row["charttime"]) == ("Creatinine", "mg/dL", "2151-07-18 06:00:00")
    missing = RelationalQuery(lab=StructuredFilter(CREATININE, hadm_id=h(99), agg="max"))
    assert duck.relational.run(missing).items == [] and duck.relational.count(missing) == 0


def _canonical(items):
    return sorted(json.dumps(i, sort_keys=True) for i in items)


def assert_equivalent(reference: RelationalSource, duck: RelationalSource, query: RelationalQuery) -> None:
    expected, actual = reference.run(query), duck.run(query)
    assert duck.count(query) == reference.count(query) == len(expected.items), query
    assert duck.matching_ids(query) == reference.matching_ids(query) == expected.ids, query
    assert actual.ids == expected.ids, query
    # Order-insensitive: the contract does not order diagnosis rows within an admission.
    assert _canonical(actual.items) == _canonical(expected.items), query
    assert actual.tokens == expected.tokens, query
    assert duck.count_notes(query) == reference.count_notes(query), query
    assert duck.count_matches(query) == reference.count_matches(query), query


@pytest.mark.parametrize("dataset", ["fixture_pair", "extended_pair"])
def test_equivalent_to_reference_on_query_grid(dataset, request, extended_dir):
    reference, duck = request.getfixturevalue(dataset)
    tables_dir = FIXTURES if dataset == "fixture_pair" else extended_dir
    queries = query_grid(reference.graph, tables_dir)
    assert len(queries) > 1000
    nonempty = 0
    for query in queries:
        assert_equivalent(reference.relational, duck.relational, query)
        nonempty += reference.relational.count(query) > 0
    assert nonempty > 100  # the grid exercises real matches, not just empty results


def test_lab_rows_keep_reference_order(extended_pair):
    reference, duck = extended_pair
    for query in (RelationalQuery(lab=StructuredFilter(CREATININE)),
                  RelationalQuery(lab=StructuredFilter(CREATININE, ">", 1.0), icd_codes=tuple(reference.graph.expand("renal_genitourinary")))):
        assert duck.relational.run(query).items == reference.relational.run(query).items


def test_graph_matches_reference(fixture_pair, extended_pair):
    for reference, duck in (fixture_pair, extended_pair):
        assert duck.graph.codes == reference.graph.codes
        for name in CATEGORIES:
            assert duck.graph.expand(name) == reference.graph.expand(name)
            assert duck.graph.descendant_count(name) == reference.graph.descendant_count(name)


def test_icd9_leading_zero_survives(extended_pair):
    reference, duck = extended_pair
    query = RelationalQuery(icd_codes=("9:0389",))
    result = duck.relational.run(query)
    assert result.ids == [h(1)]
    assert result.items[0]["icd_code"] == "0389" and result.items[0]["icd_version"] == 9
    assert "Unspecified septicemia" in result.items[0]["text"]
    assert duck.relational.count(RelationalQuery(icd_codes=("9:389",))) == 0
    assert ("0389", 9) in duck.graph.codes


def test_codes_are_trimmed_and_uppercased(extended_pair):
    _, duck = extended_pair
    assert "10:I509" in duck.graph.expand("heart_failure")
    ids = duck.relational.matching_ids(RelationalQuery(icd_codes=("10:I509",)))
    assert ids == [h(4), h(5), h(7)]


def test_rows_outside_lab_semantics_are_dropped(extended_pair):
    _, duck = extended_pair
    rows = duck.relational.run(RelationalQuery(lab=StructuredFilter(CREATININE, ">", 7.5))).items
    assert rows == []  # 9.9 has no hadm_id and 8.0 belongs to an admission missing from admissions
    assert duck.relational.matching_ids(RelationalQuery(icd_codes=("10:N186",))) == [h(2)]
    unlabeled = duck.relational.run(RelationalQuery(lab=StructuredFilter(51006))).items
    assert unlabeled[0]["text"].startswith(f"Admission {h(1)}: item 51006 20 at")


def test_empty_and_unknown(fixture_pair):
    _, duck = fixture_pair
    rel = duck.relational
    for query in (
        RelationalQuery(lab=StructuredFilter(CREATININE, ">", 100.0)),
        RelationalQuery(lab=StructuredFilter(999999)),
        RelationalQuery(icd_codes=("10:ZZZ99",)),
        RelationalQuery(lab=StructuredFilter(CREATININE, hadm_id=h(99))),
    ):
        result = rel.run(query)
        assert (rel.count(query), rel.matching_ids(query), result.ids, result.items, result.tokens) == (0, [], [], [], 0)
        assert rel.count_notes(query) == 0
        assert rel.count_matches(query) == (0, 0, 0)


def test_invalid_queries(fixture_pair):
    reference, duck = fixture_pair
    with pytest.raises(ValueError):
        RelationalQuery()
    with pytest.raises(ValueError):
        StructuredFilter(CREATININE, ">")
    unqualified = RelationalQuery(icd_codes=("I509",))
    for source in (reference.relational, duck.relational):
        with pytest.raises(ValueError):
            source.count(unqualified)


def test_note_probes_need_registered_notes(tmp_path):
    source = DuckDBRelationalSource(build_database(FIXTURES, tmp_path / "db.duckdb"))
    query = RelationalQuery(lab=StructuredFilter(CREATININE, ">", 2.0))
    with pytest.raises(NotImplementedError):
        source.count_notes(query)
    source.register_notes(["a", "b", "c", "d"], [h(2), h(2), h(4), h(99)])
    assert source.count_notes(query) == 3
    assert source.count_matches(query) == (4, 3, 3)
    source.register_notes([], [])  # replaces the metadata
    assert source.count_matches(query) == (4, 3, 0)
    with pytest.raises(ValueError):
        source.register_notes(["a"], [])


def test_file_is_read_only_and_shareable(tmp_path):
    path = build_database(FIXTURES, tmp_path / "db.duckdb")
    before = path.read_bytes()
    first, second = DuckDBRelationalSource(path), DuckDBRelationalSource(path)
    first.register_notes(["n1"], [h(2)])
    query = RelationalQuery(lab=StructuredFilter(CREATININE, ">", 2.0))
    assert first.count_notes(query) == 1
    assert second.count(query) == 4
    first.close()
    second.close()
    assert path.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["db.duckdb"]


def test_build_refuses_to_overwrite_and_checks_inputs(tmp_path):
    path = build_database(FIXTURES, tmp_path / "db.duckdb")
    with pytest.raises(FileExistsError):
        build_database(FIXTURES, path)
    assert build_database(FIXTURES, path, overwrite=True) == path
    with pytest.raises(FileNotFoundError):
        DuckDBRelationalSource(tmp_path / "missing.duckdb")
    empty = tmp_path / "tables"
    empty.mkdir()
    shutil.copy(FIXTURES / "admissions.csv", empty)
    with pytest.raises(FileNotFoundError):
        build_database(empty, tmp_path / "other.duckdb")
    assert not (tmp_path / "other.duckdb").exists()


def test_duplicate_admissions_are_rejected(tmp_path):
    tables = tmp_path / "tables"
    shutil.copytree(FIXTURES, tables)
    _append(tables / "admissions.csv", [[10000001, h(1), "2150-03-02 14:20:00", "2150-03-06 11:00:00", "EW EMER."]])
    with pytest.raises(ValueError, match="duplicate hadm_id"):
        build_database(tables, tmp_path / "db.duckdb")
    assert not (tmp_path / "db.duckdb").exists()


def test_build_sources_duckdb_kind(tmp_path):
    db = build_database(FIXTURES, tmp_path / "db.duckdb")
    sources = runner.build_sources("duckdb", notes_dir=FIXTURES, db_path=db)
    assert isinstance(sources.relational, DuckDBRelationalSource)
    assert sources.notes.total_notes() == 8
    assert {"duckdb", "duckdb_faiss"} <= set(runner.SOURCE_KINDS)
    with pytest.raises(FileNotFoundError):
        runner.build_sources("duckdb_faiss", db_path=db, index_dir=tmp_path / "no_index")


# Fields that are timings, or derived from them; everything else must match.
TIMING_FIELDS = {"latency_ms", "planning_overhead_ms", "latency_runs_ms", "overhead_runs_ms", "measured_cost",
                 "answer_latency_ms"}


def _strip_timings(record):
    record = {k: v for k, v in record.items() if k not in TIMING_FIELDS}
    record["attempts"] = [{k: v for k, v in a.items() if "latency" not in k and "ms" not in k}
                          for a in record["attempts"]]
    return record


def test_runner_records_match_reference_except_timing(workload, tmp_path):  # noqa: F811
    write_jsonl(tmp_path / "workload" / "dev.jsonl", workload.dev)
    write_jsonl(tmp_path / "workload" / "test.jsonl", workload.test)
    db = build_database(FIXTURES, tmp_path / "db.duckdb")
    common = ["--notes-dir", str(FIXTURES), "--workload", str(tmp_path / "workload"), "--split", "test",
              "--repeats", "1", "--no-analysis",
              "--routers", "oracle", "p1", "p2_tfidf", "query_all", "always_filter_first", "always_search_first"]
    ref_out = runner.main(["--sources", "fake", "--tables-dir", str(FIXTURES), *common, "--out", str(tmp_path / "ref")])
    duck_out = runner.main(["--sources", "duckdb", "--db", str(db), *common, "--out", str(tmp_path / "duck")])
    ref = [json.loads(line) for line in (ref_out / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    duck = [json.loads(line) for line in (duck_out / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(ref) == len(duck) > 0
    for a, b in zip(ref, duck):
        assert _strip_timings(a) == _strip_timings(b), a["qid"]
