"""Planner decisions on hand-made numbers.

Data: 100 admissions with 10 notes each (1000 notes), one creatinine row per
admission. Config: k=5, alpha=0.1, 10 tokens/row, 200 tokens/chunk,
relational 20 ms, graph 2 ms, scan 0.5 ms per 1000 notes, fetch 2 ms per 1000
candidates, id filter 1 ms per 1000 ids, c 0.1 (search-first may fetch at most
100 of the 1000 notes), beta 1.
"""

import pytest

from clinrag.config import PlannerConfig
from clinrag.planner.execute import execute
from clinrag.planner.planner import CostPlanner
from clinrag.question import QuestionSpec, StructuredFilter
from clinrag.sources.base import Sources
from tests.fakes import CREATININE, FakeGraph, FakeNotes, FakeRelational, creatinine_labs, grid_notes

CONFIG = PlannerConfig(
    top_k=5,
    alpha=0.1,
    avg_tokens_per_row=10,
    avg_tokens_per_chunk=200,
    relational_ms=20,
    graph_ms=2,
    note_scan_ms_per_1k_notes=0.5,
    note_fetch_ms_per_1k_candidates=2.0,
    id_filter_ms_per_1k_ids=1.0,
    c=0.1,
)
HIGH_CREATININE = StructuredFilter(CREATININE, ">", 2.0)


def make_sources(high: set[int], diagnoses=()) -> Sources:
    return Sources(
        relational=FakeRelational(creatinine_labs(100, high), diagnoses),
        notes=FakeNotes(grid_notes(100, 10)),
        graph=FakeGraph({"renal": ["N179", "N186"]}),
    )


def mixed_question() -> QuestionSpec:
    return QuestionSpec(
        qid="q1",
        text="Which patients with creatinine above 2 had dialysis?",
        qclass="C",
        relational_filter=HIGH_CREATININE,
        note_query="dialysis",
        required_sources=frozenset({"relational", "notes"}),
    )


def costs(decision):
    return {s.plan.name: s.cost.total for s in decision.candidates}


def test_selective_filter_picks_filter_first():
    # 4 matching admissions -> 40 notes; search-first needs ceil(5*1000/40) = 125 > 100.
    decision = CostPlanner(make_sources(high={1, 2, 3, 4}), CONFIG).plan(mixed_question())

    assert decision.order.order == "filter_first"
    assert decision.order.candidates == 125
    assert costs(decision) == pytest.approx(
        {
            # id_filter(4) + scan(40) + fetch(5) = 0.004 + 0.02 + 0.01
            "relational+notes:filter_first": 20 + 0.034 + 0.1 * 1000,  # 120.034
            # scan(1000) + fetch(125) = 0.5 + 0.25; pruned by the order rule
            "relational+notes:search_first": 20 + 0.75 + 0.1 * 1000,  # 120.75
            # scan(1000) + fetch(5) = 0.51; tokens 4 rows * 10 + 5 chunks * 200
            "relational+notes:unfiltered": 20 + 0.51 + 0.1 * (40 + 1000),  # 124.51
            "query_all": 20 + 0.51 + 0.1 * (40 + 1000),  # 124.51
        }
    )
    assert decision.plan.name == "relational+notes:filter_first"
    pruned = {s.plan.name for s in decision.candidates if not s.feasible}
    assert pruned == {"relational+notes:search_first"}


def test_loose_filter_picks_search_first():
    # 60 matching admissions -> 600 notes; search-first needs ceil(5000/600) = 9 <= 100.
    decision = CostPlanner(make_sources(high=set(range(1, 61))), CONFIG).plan(mixed_question())

    assert decision.order.order == "search_first"
    # scan(1000) + fetch(9) = 0.5 + 0.018
    assert costs(decision)["relational+notes:search_first"] == pytest.approx(20 + 0.518 + 100)  # 120.518
    assert decision.plan.name == "relational+notes:search_first"
    assert decision.plan.note_search.n_candidates == 9


def test_pruned_order_keeps_its_cost():
    # Same loose filter. The rule picks search_first (9 candidates <= 100), but on
    # this small index the cost model prefers filter_first:
    # id_filter(60) + scan(600) + fetch(5) = 0.06 + 0.3 + 0.01 = 0.37 ms < 0.518 ms.
    decision = CostPlanner(make_sources(high=set(range(1, 61))), CONFIG).plan(mixed_question())

    pruned = [s for s in decision.candidates if not s.feasible]
    assert [s.plan.name for s in pruned] == ["relational+notes:filter_first"]
    assert pruned[0].cost.total == pytest.approx(20 + 0.37 + 100)  # 120.37
    assert decision.order_costs == pytest.approx({"filter_first": 120.37, "search_first": 120.518})
    assert decision.order.order == "search_first"
    assert decision.cost_order == "filter_first"


def test_no_order_costs_without_filtered_note_search():
    question = QuestionSpec(
        qid="q4",
        text="Which notes mention dialysis?",
        qclass="B",
        note_query="dialysis",
        required_sources=frozenset({"notes"}),
    )
    decision = CostPlanner(make_sources(high=set()), CONFIG).plan(question)
    assert decision.order_costs == {}
    assert decision.cost_order is None


def test_chosen_is_cheapest_feasible():
    decision = CostPlanner(make_sources(high=set(range(1, 61))), CONFIG).plan(mixed_question())
    feasible = [s.cost.total for s in decision.candidates if s.feasible]
    assert decision.chosen.cost.total == min(feasible)


def test_overhead_is_recorded():
    decision = CostPlanner(make_sources(high={1}), CONFIG).plan(mixed_question())
    assert decision.probes.probe_ms > 0
    assert decision.planning_overhead_ms >= decision.probes.probe_ms


def test_relational_only_question_never_touches_notes():
    sources = make_sources(high={1, 2})
    question = QuestionSpec(
        qid="q2",
        text="Which admissions had creatinine above 2?",
        qclass="A",
        relational_filter=HIGH_CREATININE,
        required_sources=frozenset({"relational"}),
    )
    decision = CostPlanner(sources, CONFIG).plan(question)
    result = execute(decision.plan, sources, question, CONFIG)

    assert decision.plan.name == "relational_only"
    assert decision.order is None
    assert result.sources_queried == {"relational"}
    # Only the chunk count is read, to cost the query_all baseline.
    assert sources.notes.calls == ["total_notes"]


def test_graph_question_uses_expanded_codes():
    diagnoses = [(1, "N179"), (2, "I10"), (3, "N186")]
    sources = make_sources(high={1, 2, 3}, diagnoses=diagnoses)
    question = QuestionSpec(
        qid="q3",
        text="Renal patients with creatinine above 2 and dialysis notes",
        qclass="D",
        relational_filter=HIGH_CREATININE,
        note_query="dialysis",
        category="renal",
        required_sources=frozenset({"graph", "relational", "notes"}),
    )
    decision = CostPlanner(sources, CONFIG).plan(question)

    # Admissions 1 and 3 have renal codes and high creatinine -> 2 rows, 20 notes.
    assert decision.probes.graph_codes == 2
    assert decision.probes.relational_rows == 2
    assert decision.probes.matching_notes == 20
    assert decision.plan.name == "graph>relational+notes:filter_first"


def test_per_admission_filter_only_returns_that_admission():
    sources = make_sources(high={1, 2, 3})
    question = QuestionSpec(
        qid="q5",
        text="What was the highest creatinine value during admission 2?",
        qclass="A",
        relational_filter=StructuredFilter(CREATININE, hadm_id=2),
        required_sources=frozenset({"relational"}),
    )
    decision = CostPlanner(sources, CONFIG).plan(question)
    # No order choice: the relational count is skipped (lazy probes).
    assert decision.probes.relational_rows is None and sources.relational.calls == []
    result = execute(decision.plan, sources, question, CONFIG)
    assert {item["hadm_id"] for item in result.items} == {2}


def test_structured_filter_any_value_and_validation():
    any_value = StructuredFilter(CREATININE)
    assert any_value.matches(0.1) and not any_value.matches(None)
    assert StructuredFilter(CREATININE, hadm_id=5).applies_to(5)
    assert not StructuredFilter(CREATININE, hadm_id=5).applies_to(6)
    with pytest.raises(ValueError):
        StructuredFilter(CREATININE, ">")


def test_per_admission_note_question_plans_notes_only():
    sources = make_sources(high={1})
    question = QuestionSpec(
        qid="q6",
        text="According to the discharge summary for admission 7, did the patient have sepsis?",
        qclass="B",
        note_query="sepsis",
        note_hadm_id=7,
        required_sources=frozenset({"notes"}),
    )
    decision = CostPlanner(sources, CONFIG).plan(question)
    result = execute(decision.plan, sources, question, CONFIG)

    assert decision.plan.name == "notes_only"
    assert {item["hadm_id"] for item in result.items} == {7}
    assert sources.relational.calls == []


def test_per_admission_note_question_costs_one_admission_scan():
    # Admission 7 has 10 of the 1000 notes. The executor only searches those, so
    # the cost is filter_first over one admission:
    # id_filter(1) + scan(10) + fetch(5) = 0.001 + 0.005 + 0.01 = 0.016 ms,
    # not scan(1000) + fetch(5) = 0.51 ms. Tokens: 5 chunks * 200.
    sources = make_sources(high=set())
    question = QuestionSpec(
        qid="q7",
        text="According to the discharge summary for admission 7, was dialysis performed?",
        qclass="B",
        note_query="dialysis",
        note_hadm_id=7,
        required_sources=frozenset({"notes"}),
    )
    decision = CostPlanner(sources, CONFIG).plan(question)

    assert decision.probes.scoped_notes == 10
    assert "count_notes_for" in sources.notes.calls
    step = decision.chosen.cost.steps[0]
    assert step.est_latency_ms == pytest.approx(0.016)
    assert step.details == {"step": "notes:unfiltered", "scanned": 10, "fetched": 5, "filter_ids": 1}
    assert costs(decision) == pytest.approx({"notes_only": 0.016 + 0.1 * 1000, "query_all": 0.016 + 0.1 * 1000})


def test_unscoped_note_question_has_no_scoped_count():
    sources = make_sources(high=set())
    question = QuestionSpec(
        qid="q8", text="Which notes mention dialysis?", qclass="B", note_query="dialysis",
        required_sources=frozenset({"notes"}),
    )
    decision = CostPlanner(sources, CONFIG).plan(question)
    assert decision.probes.scoped_notes is None
    assert decision.chosen.cost.steps[0].est_latency_ms == pytest.approx(0.51)
