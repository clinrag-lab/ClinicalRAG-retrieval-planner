"""Reference sources on the fixture tables and notes.

Fixture facts (tests/test_workload.py): creatinine max per admission 01 1.1, 02 6.4,
03 0.9, 04 3.2, 05 1.0, 06 0.7, 07 7.1, 08 1.6; renal (N) codes on 02 04 07;
endocrine (E) codes on 01 02 08. One discharge note per admission; the notes of
02 and 07 mention dialysis.
"""

from pathlib import Path

import pytest

from clinrag.config import PlannerConfig
from clinrag.planner.execute import execute
from clinrag.planner.planner import CostPlanner
from clinrag.question import QuestionSpec, StructuredFilter
from clinrag.sources.base import GraphSource, NotesSource, RelationalQuery, RelationalSource, Sources
from clinrag.sources.reference import DemoRelationalSource, TfidfNotesSource, build_reference_sources

FIXTURES = Path(__file__).parent / "fixtures"
CREATININE = 50912


def h(n):
    return 20000000 + n


@pytest.fixture(scope="module")
def sources():
    return build_reference_sources(FIXTURES, FIXTURES)


def test_build_reference_sources_implements_interfaces(sources):
    assert isinstance(sources, Sources)
    assert isinstance(sources.relational, RelationalSource) and isinstance(sources.relational, DemoRelationalSource)
    assert isinstance(sources.notes, NotesSource) and isinstance(sources.notes, TfidfNotesSource)
    assert isinstance(sources.graph, GraphSource)


def test_lab_filter_matches_gold_rows(sources):
    query = RelationalQuery(lab=StructuredFilter(CREATININE, ">", 2.0))
    result = sources.relational.run(query)
    assert sources.relational.matching_ids(query) == [h(2), h(4), h(7)] == result.ids
    assert sources.relational.count(query) == len(result.items) == 4
    assert all(item["valuenum"] > 2.0 and item["itemid"] == CREATININE for item in result.items)
    assert all({"hadm_id", "text", "valuenum", "charttime"} <= set(item) for item in result.items)


def test_lab_filter_on_one_admission(sources):
    query = RelationalQuery(lab=StructuredFilter(CREATININE, hadm_id=h(2)))
    result = sources.relational.run(query)
    assert result.ids == [h(2)]
    assert max(item["valuenum"] for item in result.items) == 6.4
    assert "Creatinine 6.4 mg/dL" in result.items[0]["text"]


def test_icd_codes_are_matched_on_version_and_code(sources):
    codes = tuple(sources.graph.expand("renal_genitourinary"))
    assert codes and all(c.startswith("10:N") for c in codes)
    dx = RelationalQuery(icd_codes=codes)
    assert sources.relational.matching_ids(dx) == [h(2), h(4), h(7)]
    assert all(item["icd_version"] == 10 for item in sources.relational.run(dx).items)
    # The same code string under the other ICD version matches nothing.
    wrong_version = RelationalQuery(icd_codes=tuple("9:" + c.split(":", 1)[1] for c in codes))
    assert sources.relational.count(wrong_version) == 0
    # Lab filter restricted to admissions with an endocrine code: only 02 is both.
    both = RelationalQuery(lab=StructuredFilter(CREATININE, ">", 2.0), icd_codes=tuple(sources.graph.expand("endocrine")))
    assert sources.relational.matching_ids(both) == [h(2)]


def test_notes_counts_and_probe(sources):
    notes = sources.notes
    assert notes.total_notes() == 8
    assert notes.count_notes_for([h(2), h(7), 999]) == 2
    assert notes.count_notes_for([h(2), h(2)]) == 1
    scores = notes.probe_search("dialysis", 3)
    assert len(scores) == 3 and scores == sorted(scores, reverse=True)


def test_notes_search_unfiltered_and_restricted(sources):
    result = sources.notes.search("dialysis", 3)
    assert [i["hadm_id"] for i in result.items[:2]] == [h(7), h(2)]
    scores = [i["score"] for i in result.items]
    assert scores == sorted(scores, reverse=True) and scores[0] > 0
    assert all({"hadm_id", "note_id", "text", "score"} <= set(i) for i in result.items)
    assert result.tokens > 0

    restricted = sources.notes.search("dialysis", 5, hadm_ids=[h(4), h(5)])
    assert {i["hadm_id"] for i in restricted.items} == {h(4), h(5)}
    assert sources.notes.search("dialysis", 5, hadm_ids=[]).items == []
    assert sources.notes.search("dialysis", 5, hadm_ids=[999]).items == []


def test_planner_runs_end_to_end_on_reference_sources(sources):
    question = QuestionSpec(
        qid="c", text="Among admissions with creatinine > 2, which notes mention dialysis?", qclass="C",
        relational_filter=StructuredFilter(CREATININE, ">", 2.0), note_query="dialysis",
        required_sources=frozenset({"relational", "notes"}),
    )
    config = PlannerConfig()
    decision = CostPlanner(sources, config).plan(question)
    assert decision.probes.matching_admissions == 3 and decision.probes.matching_notes == 3
    result = execute(decision.plan, sources, question, config)
    assert {i["hadm_id"] for i in result.items if i["source"] == "notes"} <= {h(2), h(4), h(7)}
