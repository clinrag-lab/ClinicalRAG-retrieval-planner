"""Analyses on the fake sources, with measured latencies planted so the outcome is hand-computable.

Data and config as in test_planner: 100 admissions x 10 notes, k=5, max_candidates 100.
Creatinine is high on admissions 1-4 (40 notes: the rule picks filter_first, and
the cost model agrees: 120.034 vs 120.75). Potassium is high on 1-60 (600 notes:
the rule picks search_first; the cost model prefers filter_first, 120.37 vs 120.518).
"""

from dataclasses import replace

import pytest

from clinrag.eval import analysis
from clinrag.eval.analysis import cost_model_check, order_check, order_table, overhead_share
from clinrag.eval.metrics import retrieval_score
from clinrag.planner.execute import execute as real_execute
from clinrag.question import QuestionSpec, StructuredFilter
from clinrag.sources.base import Sources
from clinrag.workload.schema import WorkloadQuestion
from tests.fakes import CREATININE, FakeGraph, FakeNotes, FakeRelational, grid_notes
from tests.test_planner import CONFIG

POTASSIUM = 50971


def sources():
    labs = [(h, CREATININE, 3.0 if h <= 4 else 1.0) for h in range(1, 101)]
    labs += [(h, POTASSIUM, 6.0 if h <= 60 else 4.0) for h in range(1, 101)]
    return Sources(relational=FakeRelational(labs), notes=FakeNotes(grid_notes(100, 10)), graph=FakeGraph({}))


def c_question(qid, itemid, selectivity, gold):
    spec = QuestionSpec(
        qid=qid, text=qid, qclass="C", relational_filter=StructuredFilter(itemid, ">", 2.0 if itemid == CREATININE else 5.0),
        note_query="dialysis", required_sources=frozenset({"relational", "notes"}),
    )
    return WorkloadQuestion(spec, {"answer": gold, "template_id": f"C_{qid}", "selectivity": selectivity})


TIGHT = c_question("tight", CREATININE, 0.04, [1, 2])
LOOSE = c_question("loose", POTASSIUM, 0.60, [1, 2, 3, 4, 5, 6, 7])
A = WorkloadQuestion(
    QuestionSpec(qid="a", text="a", qclass="A", relational_filter=StructuredFilter(CREATININE, hadm_id=2),
                 required_sources=frozenset({"relational"})),
    {"answer": 3.0, "template_id": "A_max_creatinine", "hadm_id": 2},
)

PLANTED_MS = {
    "tight": {"filter_first": 2.0, "search_first": 6.0},
    "loose": {"filter_first": 9.0, "search_first": 3.0},
}


@pytest.fixture
def planted(monkeypatch):
    def fake_execute(plan, src, question, config):
        result = real_execute(plan, src, question, config)
        return replace(result, total_latency_ms=PLANTED_MS[question.qid][plan.note_search.mode])

    monkeypatch.setattr(analysis, "execute", fake_execute)


def test_order_check_hand_computed(planted):
    rows = {r["qid"]: r for r in order_check([A, TIGHT, LOOSE], sources(), CONFIG, repeats=2)}
    assert set(rows) == {"tight", "loose"}  # class A has no filtered note search

    tight = rows["tight"]
    assert (tight["faster"], tight["rule_pick"], tight["cost_pick"]) == ("filter_first", "filter_first", "filter_first")
    assert tight["rule_matches_faster"] and tight["cost_matches_faster"]
    assert (tight["filter_first_ms"], tight["search_first_ms"]) == (2.0, 6.0)
    assert tight["matching_notes"] == 40
    # Both orders keep the 5 best notes, all from admission 1: recall 1 / min(5, 2).
    assert tight["filter_first_recall"] == tight["search_first_recall"] == 0.5

    loose = rows["loose"]
    assert (loose["faster"], loose["rule_pick"], loose["cost_pick"]) == ("search_first", "search_first", "filter_first")
    assert loose["rule_matches_faster"] and not loose["cost_matches_faster"]


def test_order_table_hand_computed(planted):
    table = {r["selectivity"]: r for r in order_table(order_check([TIGHT, LOOSE], sources(), CONFIG, repeats=1))}
    assert set(table) == {"0.00-0.05", "0.50-0.75", "all"}
    assert (table["0.00-0.05"]["rule_matches_faster"], table["0.00-0.05"]["cost_matches_faster"]) == (1.0, 1.0)
    assert (table["0.50-0.75"]["rule_matches_faster"], table["0.50-0.75"]["cost_matches_faster"]) == (1.0, 0.0)
    everything = table["all"]
    assert everything["n"] == 2
    assert everything["filter_first_faster"] == 0.5
    assert (everything["rule_matches_faster"], everything["cost_matches_faster"]) == (1.0, 0.5)
    assert everything["rule_picks_filter_first"] == 0.5
    assert everything["mean_filter_first_ms"] == 5.5 and everything["mean_search_first_ms"] == 4.5


def test_cost_model_check_structure():
    summary, rows = cost_model_check([A, TIGHT, LOOSE], sources(), CONFIG)
    # A: relational_only + query_all; C: filter_first, search_first, unfiltered, query_all.
    assert len(rows) == 2 + 4 + 4
    assert {r["qid"] for r in rows if r["pruned"]} == {"tight", "loose"}
    assert summary["overall"]["n_plans"] == 10 and summary["overall"]["questions_with_choice"] == 3
    assert set(summary) == {"overall", "A", "C"}
    assert -1.0 <= summary["overall"]["spearman_cost"] <= 1.0
    # Token estimates rank the plans like the measured tokens.
    assert summary["overall"]["spearman_tokens"] > 0.8
    for r in rows:
        assert r["measured_cost"] == pytest.approx(r["measured_latency_ms"] + CONFIG.alpha * r["measured_tokens"])


def test_overhead_share_hand_computed():
    records = [
        {"router": "x", "planning_overhead_ms": 1.0, "latency_ms": 3.0},
        {"router": "x", "planning_overhead_ms": 3.0, "latency_ms": 1.0},
        {"router": "y", "planning_overhead_ms": 0.0, "latency_ms": 2.0},
    ]
    rows = {r["router"]: r for r in overhead_share(records)}
    assert rows["x"]["mean_overhead_share"] == 0.5 and rows["x"]["overhead_share_of_total_time"] == 0.5
    assert rows["y"]["mean_overhead_share"] == 0.0


def test_retrieval_score_by_question_type():
    lab_rows = [
        {"source": "relational", "hadm_id": 2, "valuenum": 1.0, "text": "x"},
        {"source": "relational", "hadm_id": 2, "valuenum": 3.0, "text": "x"},
    ]
    assert retrieval_score(A, lab_rows, 5).value == 1.0 and retrieval_score(A, lab_rows, 5).metric == "value_row"
    assert retrieval_score(A, lab_rows[:1], 5).value == 0.0

    ever = WorkloadQuestion(
        replace(A.spec, relational_filter=StructuredFilter(CREATININE, ">", 2.0, hadm_id=2)),
        {"answer": 1, "template_id": "A_ever_above_creatinine", "hadm_id": 2},
    )
    assert retrieval_score(ever, lab_rows, 5).value == 1.0
    never = WorkloadQuestion(ever.spec, {**ever.gold, "answer": 0})
    assert retrieval_score(never, [], 5).value == 1.0
    assert retrieval_score(never, lab_rows, 5).value == 0.0

    b_adm = WorkloadQuestion(
        QuestionSpec(qid="b", text="b", qclass="B", note_query="dialysis", note_hadm_id=7, required_sources=frozenset({"notes"})),
        {"answer": 1, "template_id": "B_admission_dialysis", "hadm_id": 7},
    )
    assert retrieval_score(b_adm, [{"source": "notes", "hadm_id": 7}], 5).value == 1.0
    assert retrieval_score(b_adm, [{"source": "notes", "hadm_id": 8}], 5).metric == "gold_note"

    notes = [{"source": "notes", "hadm_id": h} for h in (1, 9, 2)] + [{"source": "relational", "hadm_id": 3}]
    score = retrieval_score(TIGHT, notes, 5)
    assert score.metric == "normalized_recall" and score.value == 1.0
    assert retrieval_score(LOOSE, notes, 2).value == 0.5  # top 2 notes: 1 and 9; 1 is gold
