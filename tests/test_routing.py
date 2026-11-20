"""Routers on the fake sources: oracle, keyword rules (P1) and classifier + cost planner (P2).

Same data and config as test_planner: 100 admissions x 10 notes, one creatinine
row per admission, s 0.1 (filter-first below 10% of the notes).
"""

import pytest

from clinrag.planner.execute import execute
from clinrag.planner.router import (
    ClassifierCostRouter,
    OracleCostRouter,
    Router,
    RoutingResult,
    input_question,
    restrict_question,
)
from clinrag.planner.rule_planner import RulePlanner, rule_sources
from clinrag.question import QuestionSpec, StructuredFilter
from tests.fakes import CREATININE
from tests.test_planner import CONFIG, HIGH_CREATININE, make_sources

POTASSIUM = 50971
ALL = {"relational", "notes", "graph"}


def q(qid, text, qclass, sources, **kw):
    return QuestionSpec(qid=qid, text=text, qclass=qclass, required_sources=frozenset(sources), **kw)


A = q("A1", "What was the highest creatinine value during admission 2?", "A", {"relational"},
      relational_filter=StructuredFilter(CREATININE, hadm_id=2))
B = q("B1", "Which discharge summaries mention dialysis?", "B", {"notes"}, note_query="dialysis")
B_ADM = q("B2", "According to the discharge summary for admission 7, was dialysis performed?", "B", {"notes"},
          note_query="dialysis", note_hadm_id=7)
C_TIGHT = q("C1", "Among admissions with creatinine > 2 mg/dL, which discharge summaries mention dialysis?", "C",
            {"relational", "notes"}, relational_filter=HIGH_CREATININE, note_query="dialysis")
C_LOOSE = q("C2", "Among admissions with creatinine > 2 mg/dL, which discharge summaries mention sepsis?", "C",
            {"relational", "notes"}, relational_filter=HIGH_CREATININE, note_query="sepsis")
D = q("D1", "Among admissions with any acute kidney injury diagnosis, which discharge summaries mention dialysis?",
      "D", ALL, note_query="dialysis", category="renal")
QUESTIONS = [A, B, B_ADM, C_TIGHT, C_LOOSE, D]

DIAGNOSES = [(1, "N179"), (3, "N186"), (5, "N179")]


def sources_for(question):
    # Loose filter (60 admissions) for C_LOOSE so the cost planner picks search_first.
    high = set(range(1, 61)) if question is C_LOOSE else {1, 2, 3, 4}
    return make_sources(high=high, diagnoses=DIAGNOSES)


class PerfectClassifier:
    """Stub: probability 1 for the true sources, 0 for the rest."""

    def __init__(self):
        self.last_latency_ms = 0.0

    def predict_proba(self, question):
        self.last_latency_ms = 0.25
        return {s: float(s in question.required_sources) for s in ALL}


class FixedClassifier:
    def __init__(self, probs):
        self.probs = probs
        self.last_latency_ms = 0.0

    def predict_proba(self, question):
        self.last_latency_ms = 0.1
        return dict(self.probs)


# ---- P1 rules ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("What was the most recent potassium value during admission 21234567?", {"relational"}),
        ("Was hemoglobin ever below 8 during this stay?", {"relational"}),
        ("Which patients had a value greater than 5?", {"relational"}),
        ("Which discharge summaries mention sepsis?", {"notes"}),
        ("Was a fall documented for this patient?", {"notes"}),
        ("Give me a summary of the hospital course.", {"notes"}),
        ("Among admissions with creatinine > 2, which notes mention dialysis?", {"relational", "notes"}),
        ("Among admissions with any heart failure diagnosis, which summaries mention dialysis?", ALL),
        ("List admissions with a COPD diagnosis.", {"graph", "relational"}),
        ("Tell me about this patient.", set()),
    ],
)
def test_rule_sources(text, expected):
    assert rule_sources(text) == expected


def test_rule_planner_plans():
    planner = RulePlanner()
    expected = {
        "A1": "relational_only",
        "B1": "notes_only",
        "B2": "notes_only",
        "C1": "relational+notes:filter_first",
        "C2": "relational+notes:filter_first",
        "D1": "graph>relational+notes:filter_first",
    }
    for question in QUESTIONS:
        decision = planner.decide(question)
        assert decision.plan.name == expected[question.qid], question.qid
        assert decision.predicted_sources == question.required_sources
        assert decision.chosen.cost is None
        assert decision.planning_overhead_ms > 0
        assert decision.candidates[-1].plan.name == "query_all" or decision.plan.sources == {"notes"}


def test_rule_planner_never_uses_search_first():
    # The cost planner picks search_first for the loose filter; P1 still filters first.
    assert OracleCostRouter(sources_for(C_LOOSE), CONFIG).decide(C_LOOSE).plan.name == "relational+notes:search_first"
    planner = RulePlanner()
    for question in QUESTIONS:
        for scored in planner.decide(question).candidates:
            search = scored.plan.note_search
            assert search is None or search.mode != "search_first"


def test_rule_planner_runs_no_probes():
    sources = sources_for(C_TIGHT)
    decision = RulePlanner().decide(C_TIGHT)
    assert sources.notes.calls == [] and sources.relational.calls == []
    result = execute(decision.plan, sources, C_TIGHT, CONFIG)
    assert {item["hadm_id"] for item in result.items} <= {1, 2, 3, 4}


def test_rule_planner_defaults_to_notes():
    question = q("X", "Tell me about this patient.", "C", {"relational", "notes"},
                 relational_filter=HIGH_CREATININE, note_query="dialysis")
    decision = RulePlanner().decide(question)
    assert decision.predicted_sources == frozenset()
    assert decision.plan.name == "notes_only"
    assert [c.plan.name for c in decision.candidates] == ["notes_only", "query_all"]


# ---- P2 ---------------------------------------------------------------------------------


def test_p2_with_perfect_classifier_matches_oracle():
    for question in QUESTIONS:
        sources = sources_for(question)
        oracle = OracleCostRouter(sources, CONFIG).decide(question)
        p2 = ClassifierCostRouter(PerfectClassifier(), sources, CONFIG).decide(question)
        assert p2.plan == oracle.plan, question.qid
        assert p2.chosen.cost.total == pytest.approx(oracle.chosen.cost.total)
        assert [c.plan.name for c in p2.candidates] == [c.plan.name for c in oracle.candidates]
        assert p2.predicted_sources == question.required_sources


def test_p2_overhead_includes_classifier_and_planner():
    sources = sources_for(C_TIGHT)
    router = ClassifierCostRouter(PerfectClassifier(), sources, CONFIG)
    decision = router.decide(C_TIGHT)
    assert router.last_classifier_ms == 0.25
    assert decision.planning_overhead_ms >= decision.probes.probe_ms > 0


def test_p2_tau_controls_predicted_sources():
    probs = {"relational": 0.4, "notes": 0.8, "graph": 0.1}
    sources = sources_for(C_TIGHT)
    low = ClassifierCostRouter(FixedClassifier(probs), sources, CONFIG, tau=0.3).decide(C_TIGHT)
    high = ClassifierCostRouter(FixedClassifier(probs), sources, CONFIG, tau=0.5).decide(C_TIGHT)
    assert low.predicted_sources == {"relational", "notes"}
    assert low.plan.name == "relational+notes:filter_first"
    assert high.predicted_sources == {"notes"}
    assert high.plan.name == "notes_only"


def test_p2_missed_source_keeps_full_query_all():
    # The classifier misses relational; query_all still covers it so a fallback can recover.
    sources = sources_for(C_TIGHT)
    router = ClassifierCostRouter(FixedClassifier({"relational": 0.1, "notes": 0.9, "graph": 0.0}), sources, CONFIG)
    decision = router.decide(C_TIGHT)
    assert decision.plan.name == "notes_only"
    query_all = [c for c in decision.candidates if c.plan.name == "query_all"]
    assert len(query_all) == 1
    assert query_all[0].plan.sources == {"relational", "notes"}
    assert query_all[0].cost.total > decision.chosen.cost.total


def test_p2_empty_prediction_uses_most_probable_executable_source():
    sources = sources_for(A)
    router = ClassifierCostRouter(FixedClassifier({"relational": 0.3, "notes": 0.2, "graph": 0.4}), sources, CONFIG)
    decision = router.decide(A)
    # graph is most probable but A has no category, so relational is next.
    assert decision.predicted_sources == frozenset()
    assert decision.plan.name == "relational_only"


def test_routers_implement_router():
    sources = sources_for(A)
    for router in (OracleCostRouter(sources, CONFIG), RulePlanner(), ClassifierCostRouter(PerfectClassifier(), sources, CONFIG)):
        assert isinstance(router, Router)
        assert router.name


# ---- restrict_question -----------------------------------------------------------------------


def test_restrict_question():
    assert restrict_question(C_TIGHT, C_TIGHT.required_sources) == C_TIGHT
    assert restrict_question(D, ALL) == D

    notes_only = restrict_question(D, {"notes"})
    assert notes_only.required_sources == {"notes"} and notes_only.category is None

    # Notes on a question without a note query search the question text.
    a_notes = restrict_question(A, {"relational", "notes"})
    assert a_notes.required_sources == {"relational", "notes"} and a_notes.note_query == A.text

    # Relational without a filter or graph cannot run and is dropped.
    assert restrict_question(B, {"relational", "notes"}).required_sources == {"notes"}
    assert restrict_question(D, {"relational", "notes"}).required_sources == {"notes"}
    assert restrict_question(D, {"graph"}).required_sources == {"graph", "relational"}

    # note_hadm_id survives only on a notes-only copy.
    assert restrict_question(B_ADM, {"notes"}).note_hadm_id == 7

    # Nothing executable predicted: first executable source in the preference order.
    assert restrict_question(B, set(), preference=("relational", "notes")).required_sources == {"notes"}
    assert input_question(C_TIGHT) == C_TIGHT


# ---- RoutingResult ---------------------------------------------------------------------------


def test_routing_result_from_execution():
    sources = sources_for(C_TIGHT)
    router = OracleCostRouter(sources, CONFIG)
    decision = router.decide(C_TIGHT)
    run = execute(decision.plan, sources, C_TIGHT, CONFIG)
    row = RoutingResult.from_run(router.name, C_TIGHT, decision, run)

    assert row.router == "oracle_cost" and row.qid == "C1" and row.qclass == "C"
    assert row.plan_name == "relational+notes:filter_first"
    assert row.predicted_sources == row.required_sources == {"relational", "notes"}
    assert row.missed_sources == frozenset()
    assert row.est_cost == pytest.approx(decision.chosen.cost.total)
    assert row.tokens == run.tokens and row.latency_ms == run.total_latency_ms
    assert row.sources_queried == {"relational", "notes"}
    assert row.fallbacks_used == 0 and row.executed_plans == ("relational+notes:filter_first",)
    d = row.to_dict()
    assert d["predicted_sources"] == ["notes", "relational"] and d["missed_sources"] == []


def test_routing_result_without_cost_model():
    decision = RulePlanner().decide(B)
    run = execute(decision.plan, sources_for(B), B, CONFIG)
    assert RoutingResult.from_run("rules", B, decision, run).est_cost is None
