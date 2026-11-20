"""Lazy probes, SQL note counts and the graph cache."""

from pathlib import Path

import pytest

from clinrag.planner.cost import estimate_cost
from clinrag.planner.planner import CostPlanner
from clinrag.planner.probes import has_order_choice, run_probes
from clinrag.question import QuestionSpec, StructuredFilter
from clinrag.sources.base import RelationalQuery
from clinrag.sources.icd_graph import IcdGraph
from clinrag.sources.reference import DemoRelationalSource, build_reference_sources
from tests.fakes import CREATININE
from tests.test_planner import CONFIG, HIGH_CREATININE, make_sources, mixed_question

FIXTURES = Path(__file__).parent / "fixtures"


def a_question():
    return QuestionSpec(qid="a", text="max creatinine in admission 2", qclass="A",
                        relational_filter=StructuredFilter(CREATININE, hadm_id=2), required_sources=frozenset({"relational"}))


def b_question(hadm=None):
    return QuestionSpec(qid="b", text="notes on dialysis", qclass="B", note_query="dialysis", note_hadm_id=hadm,
                        required_sources=frozenset({"notes"}))


def d_question():
    return QuestionSpec(qid="d", text="renal + dialysis", qclass="D", note_query="dialysis", category="renal",
                        required_sources=frozenset({"graph", "relational", "notes"}))


def test_order_choice():
    assert has_order_choice(mixed_question()) and has_order_choice(d_question())
    assert not has_order_choice(a_question()) and not has_order_choice(b_question())


def test_relational_only_question_skips_counts_and_trial_search():
    sources = make_sources(high={1, 2})
    probes = run_probes(a_question(), sources, CONFIG)
    assert probes.ran == ("total_notes",)
    assert sources.relational.calls == [] and sources.notes.calls == ["total_notes"]
    assert probes.relational_rows is None and probes.probe_scores == ()


def test_notes_only_questions_skip_the_trial_search():
    sources = make_sources(high=set())
    assert run_probes(b_question(), sources, CONFIG).ran == ("total_notes",)
    scoped = run_probes(b_question(hadm=7), sources, CONFIG)
    assert scoped.ran == ("total_notes", "scoped_notes") and scoped.scoped_notes == 10
    assert "probe_search" not in sources.notes.calls


def test_unprobed_plans_are_still_costed():
    # relational_only and query_all are both costed; unknown rows cost no tokens (a lower bound).
    decision = CostPlanner(make_sources(high={1}), CONFIG).plan(a_question())
    assert decision.plan.name == "relational_only"
    step = decision.chosen.cost.steps[0]
    assert step.details["unprobed"] and step.est_tokens == 0 and step.est_latency_ms == CONFIG.relational_ms
    assert {c.plan.name for c in decision.candidates} == {"relational_only", "query_all"}


def test_order_choice_runs_every_probe_with_the_id_fallback():
    # The fakes implement neither count_matches nor count_notes.
    sources = make_sources(high={1, 2, 3, 4}, diagnoses=[(1, "N179"), (2, "N186")])
    probes = run_probes(mixed_question(), sources, CONFIG)
    assert probes.ran == ("relational_count", "matching_ids", "note_count_for_ids", "total_notes", "trial_search")
    assert (probes.relational_rows, probes.matching_admissions, probes.matching_notes) == (4, 4, 40)
    graph = run_probes(d_question(), sources, CONFIG)
    assert graph.ran[0] == "graph_expand" and graph.graph_codes == 2


# ---- SQL note counts on the reference sources ------------------------------------------


@pytest.fixture(scope="module")
def reference():
    return build_reference_sources(FIXTURES, FIXTURES)


def queries(sources):
    return [
        RelationalQuery(lab=StructuredFilter(CREATININE, ">", 2.0)),
        RelationalQuery(icd_codes=tuple(sources.graph.expand("renal_genitourinary"))),
        RelationalQuery(lab=StructuredFilter(CREATININE, ">", 2.0), icd_codes=tuple(sources.graph.expand("endocrine"))),
        RelationalQuery(lab=StructuredFilter(CREATININE, ">", 99.0)),
    ]


def test_sql_note_counts_match_the_id_list_path(reference):
    for query in queries(reference):
        ids = reference.relational.matching_ids(query)
        expected = (reference.relational.count(query), len(ids), reference.notes.count_notes_for(ids))
        assert reference.relational.count_matches(query) == expected
        assert reference.relational.count_notes(query) == expected[2]


def test_note_metadata_table_has_no_text(reference):
    columns = [r[0] for r in reference.relational.con.execute("DESCRIBE note_meta").fetchall()]
    assert columns == ["note_id", "hadm_id"]


def test_reference_probes_use_one_sql_query(reference):
    question = QuestionSpec(qid="c", text="c", qclass="C", relational_filter=StructuredFilter(CREATININE, ">", 2.0),
                            note_query="dialysis", required_sources=frozenset({"relational", "notes"}))
    probes = run_probes(question, reference, CONFIG)
    assert probes.ran == ("relational_counts_sql", "total_notes", "trial_search")
    assert (probes.relational_rows, probes.matching_admissions, probes.matching_notes) == (4, 3, 3)


def test_unregistered_source_falls_back():
    source = DemoRelationalSource.from_tables(FIXTURES)
    with pytest.raises(NotImplementedError):
        source.count_notes(RelationalQuery(lab=HIGH_CREATININE))
    with pytest.raises(NotImplementedError):
        source.count_matches(RelationalQuery(lab=HIGH_CREATININE))


def test_graph_expand_is_cached():
    graph = IcdGraph([("N179", 10), ("I10", 10)])
    first = graph.expand("renal_genitourinary")
    first.append("mutated")
    assert graph.expand("renal_genitourinary") == ["10:N179"]
    assert "renal_genitourinary" in graph._expanded
