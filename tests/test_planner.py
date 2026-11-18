"""Planner decisions on hand-made numbers.

Data: 100 admissions with 10 notes each (1000 notes), one creatinine row per
admission. Config: k=5, alpha=0.1, 10 tokens/row, 200 tokens/chunk,
relational 20 ms, note search 8 ms per 1000 candidates, max_candidates 100.
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
    note_search_ms_per_1k_candidates=8,
    max_candidates=100,
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
            "relational+notes:filter_first": 20 + 0.32 + 0.1 * 1000,  # 120.32
            "relational+notes:search_first": 20 + 1.0 + 0.1 * 1000,  # 121.0, pruned
            "relational+notes:unfiltered": 20 + 0.04 + 0.1 * (40 + 1000),  # 124.04
            "query_all": 20 + 0.04 + 0.1 * (40 + 1000),  # 124.04
        }
    )
    assert decision.plan.name == "relational+notes:filter_first"
    pruned = {s.plan.name for s in decision.candidates if not s.feasible}
    assert pruned == {"relational+notes:search_first"}


def test_loose_filter_picks_search_first():
    # 60 matching admissions -> 600 notes; search-first needs ceil(5000/600) = 9 <= 100.
    decision = CostPlanner(make_sources(high=set(range(1, 61))), CONFIG).plan(mixed_question())

    assert decision.order.order == "search_first"
    assert costs(decision)["relational+notes:search_first"] == pytest.approx(20 + 0.072 + 100)  # 120.072
    assert costs(decision)["relational+notes:filter_first"] == pytest.approx(20 + 4.8 + 100)  # 124.8, pruned
    assert decision.plan.name == "relational+notes:search_first"
    assert decision.plan.note_search.n_candidates == 9


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
