"""Evidence check and fallback on the fake sources.

Same data and config as test_planner: 100 admissions x 10 notes; note scores
fall with admission id (admission 1 ~ 1.0, admission 100 ~ 0.0); one creatinine
row per admission; c 0.1 (at most 100 search-first candidates). min_similarity 0.3, min_surviving_notes 1.
"""

from dataclasses import replace

import pytest

from clinrag.planner.enumerate import query_all_plan
from clinrag.planner.execute import execute
from clinrag.planner.fallback import check_evidence, next_plan, run_with_fallback
from clinrag.planner.plan import NoteSearch, Plan, RelationalFilter
from clinrag.planner.planner import PlanDecision, ScoredPlan
from clinrag.planner.probes import Probes
from clinrag.planner.router import ClassifierCostRouter, OracleCostRouter, RoutingResult
from clinrag.planner.rule_planner import RulePlanner
from clinrag.question import QuestionSpec
from clinrag.sources.base import Sources
from tests.fakes import FakeNote, FakeNotes, FakeRelational, creatinine_labs
from tests.test_planner import CONFIG, HIGH_CREATININE, make_sources, mixed_question

FALLBACK_CONFIG = replace(CONFIG, min_similarity=0.3, min_surviving_notes=1, max_fallbacks=2)


def oracle_run(high, config=FALLBACK_CONFIG, **kw):
    sources = make_sources(high=high)
    question = mixed_question()
    decision = OracleCostRouter(sources, config).decide(question)
    return decision, run_with_fallback(decision, sources, question, config, **kw)


def notes_question(**kw):
    return QuestionSpec(
        qid="b", text="Which notes mention dialysis?", qclass="B", note_query="dialysis",
        required_sources=frozenset({"notes"}), **kw,
    )


def test_config_defaults():
    from clinrag.config import PlannerConfig

    config = PlannerConfig()
    assert (config.min_similarity, config.min_surviving_notes, config.max_fallbacks) == (0.3, 1, 2)


def test_fake_notes_carry_scores():
    items = make_sources(high=set()).notes.search("x", 2).items
    assert [item["score"] for item in items] == pytest.approx([1.0, 0.999])


def test_passing_check_runs_one_plan():
    # Admissions 1-4 match: filter_first keeps their notes, scores ~0.96-1.0.
    decision, run = oracle_run(high={1, 2, 3, 4})
    assert run.ok
    assert run.executed_plans == ("relational+notes:filter_first",)
    assert run.fallbacks_used == 0
    assert run.attempts[0].reason == "ok"


def test_empty_result_triggers_next_plan():
    # No admission matches: filter_first returns nothing. The unfiltered variant
    # queries the same sources, so the fallback skips it and runs query_all.
    decision, run = oracle_run(high=set())
    assert decision.plan.name == "relational+notes:filter_first"
    assert run.executed_plans == ("relational+notes:filter_first", "query_all")
    assert not run.attempts[0].ok and run.attempts[0].reason == "no evidence items"
    assert run.ok and run.fallbacks_used == 1
    assert run.plan_name == "query_all" and run.items == run.final.items


def test_low_similarity_triggers_next_plan():
    # Admissions 97-100 match (40 notes, search-first needs 125 > 100: filter_first);
    # their notes score below 0.04.
    decision, run = oracle_run(high={97, 98, 99, 100})
    assert decision.plan.name == "relational+notes:filter_first"
    assert "below min_similarity" in run.attempts[0].reason
    assert run.executed_plans == ("relational+notes:filter_first", "query_all")
    assert run.ok


def test_too_few_surviving_notes_triggers_next_plan():
    # Admissions 1-4 match; filter_first keeps k = 5 notes, fewer than the 6 required.
    config = replace(FALLBACK_CONFIG, min_surviving_notes=6)
    decision, run = oracle_run(high={1, 2, 3, 4}, config=config)
    assert decision.plan.name == "relational+notes:filter_first"
    assert run.attempts[0].reason == "5 notes survived the filter, need 6"
    # query_all's note search is unfiltered, so the surviving-notes rule does not apply.
    assert run.executed_plans == ("relational+notes:filter_first", "query_all") and run.ok


def test_search_first_with_no_survivors_triggers_next_plan():
    # Admissions 96-100 match (50 notes): search_first fetches 100 candidates, all
    # from admissions 1-10, so no note survives, and the relational rows are only
    # a filter: the result is empty.
    decision, run = oracle_run(high={96, 97, 98, 99, 100})
    assert decision.plan.name == "relational+notes:search_first"
    assert run.attempts[0].reason == "no evidence items"
    assert run.executed_plans[-1] == "query_all" and run.ok


def test_costs_add_up_across_attempts():
    sources = make_sources(high=set())
    question = mixed_question()
    decision = OracleCostRouter(sources, FALLBACK_CONFIG).decide(question)
    run = run_with_fallback(decision, sources, question, FALLBACK_CONFIG)

    first = execute(decision.plan, make_sources(high=set()), question, FALLBACK_CONFIG)
    assert len(run.attempts) == 2
    assert run.tokens == sum(a.tokens for a in run.attempts) == first.tokens + run.final.tokens
    assert run.total_latency_ms == pytest.approx(sum(a.latency_ms for a in run.attempts))
    assert run.total_latency_ms > run.final.total_latency_ms
    assert run.sources_queried == {"relational", "notes"}

    row = RoutingResult.from_run("oracle_cost", question, decision, run)
    assert row.tokens == run.tokens and row.latency_ms == run.total_latency_ms
    assert row.fallbacks_used == 1 and row.executed_plans == ("relational+notes:filter_first", "query_all")


def low_score_chain():
    """Every plan fails: notes all score 0.1 and no creatinine row is high."""
    notes = [FakeNote(f"{h}-0", h, "note text", 0.1) for h in range(1, 11)]
    sources = Sources(relational=FakeRelational(creatinine_labs(10, set())), notes=FakeNotes(notes))
    question = mixed_question()
    ff = Plan("relational+notes:filter_first", (RelationalFilter(HIGH_CREATININE, as_evidence=False), NoteSearch("dialysis", "filter_first")))
    notes_only = Plan("notes_only", (NoteSearch("dialysis"),))
    wider = Plan("relational+notes:unfiltered", (RelationalFilter(HIGH_CREATININE), NoteSearch("dialysis")))
    candidates = (ScoredPlan(notes_only, None), ScoredPlan(wider, None), ScoredPlan(query_all_plan(question), None))
    decision = PlanDecision("c", candidates[0], candidates, Probes(), None, 0.0)
    return decision, sources, question, ff


@pytest.mark.parametrize("limit, expected", [(0, 1), (1, 2), (2, 2), (5, 2)])
def test_max_fallbacks_is_respected(limit, expected):
    # notes_only -> relational+notes:unfiltered (wider). query_all has the same
    # steps as the unfiltered plan, so it is skipped and the chain ends at 2.
    decision, sources, question, _ = low_score_chain()
    run = run_with_fallback(decision, sources, question, FALLBACK_CONFIG, max_fallbacks=limit)
    assert len(run.attempts) == expected
    assert not run.ok
    assert run.fallbacks_used == expected - 1


def test_max_fallbacks_defaults_to_config():
    decision, sources, question, _ = low_score_chain()
    run = run_with_fallback(decision, sources, question, replace(FALLBACK_CONFIG, max_fallbacks=0))
    assert len(run.attempts) == 1
    with pytest.raises(ValueError):
        run_with_fallback(decision, sources, question, FALLBACK_CONFIG, max_fallbacks=-1)


def test_next_plan_needs_wider_sources_or_query_all():
    decision, _, question, ff = low_score_chain()
    order = list(decision.candidates)
    notes_only, wider, query_all = (c.plan for c in order)
    assert next_plan(order, notes_only, [notes_only]) == wider
    # From a relational+notes plan, notes_only and the unfiltered plan are not wider; query_all is next.
    assert next_plan(order, ff, [ff]) == query_all
    assert next_plan(order, wider, [notes_only, wider]) is None


def test_scoped_note_search_needs_surviving_notes():
    # Per-admission question on an admission with no notes.
    sources = make_sources(high=set())
    question = notes_question(note_hadm_id=500)
    decision = OracleCostRouter(sources, FALLBACK_CONFIG).decide(question)
    run = run_with_fallback(decision, sources, question, FALLBACK_CONFIG)
    # query_all is the same scoped search, so nothing else is tried.
    assert run.executed_plans == ("notes_only",)
    assert not run.ok and run.attempts[0].reason == "no evidence items"


def test_check_evidence_cases():
    sources = make_sources(high={1})
    question = mixed_question()
    relational_only = QuestionSpec(qid="a", text="t", qclass="A", relational_filter=HIGH_CREATININE,
                                   required_sources=frozenset({"relational"}))
    rows = execute(Plan("relational_only", (RelationalFilter(HIGH_CREATININE),)), sources, relational_only, CONFIG)
    assert check_evidence(rows, relational_only, FALLBACK_CONFIG) == (True, "ok")

    # Relational rows exist but the unfiltered note search scores too low.
    low = Sources(relational=FakeRelational(creatinine_labs(3, {1})), notes=FakeNotes([FakeNote("n", 1, "t", 0.2)]))
    plan = Plan("relational+notes:unfiltered", (RelationalFilter(HIGH_CREATININE), NoteSearch("dialysis")))
    ok, reason = check_evidence(execute(plan, low, question, CONFIG), question, FALLBACK_CONFIG)
    assert not ok and "0.200" in reason
    ok, _ = check_evidence(execute(plan, low, question, CONFIG), question, replace(FALLBACK_CONFIG, min_similarity=0.1))
    assert ok


def test_fallback_recovers_source_missed_by_classifier():
    # P2 predicts notes only on a relational + notes question; the notes all score
    # low, so the fallback runs the full query_all and reaches relational.
    notes = [FakeNote(f"{h}-0", h, "note text", 0.1) for h in range(1, 11)]
    sources = Sources(relational=FakeRelational(creatinine_labs(10, {1, 2})), notes=FakeNotes(notes))

    class NotesOnly:
        last_latency_ms = 0.0

        def predict_proba(self, question):
            return {"relational": 0.1, "notes": 0.9, "graph": 0.0}

    question = mixed_question()
    decision = ClassifierCostRouter(NotesOnly(), sources, FALLBACK_CONFIG).decide(question)
    run = run_with_fallback(decision, sources, question, FALLBACK_CONFIG)
    assert run.executed_plans == ("notes_only", "query_all")
    assert run.sources_queried == {"relational", "notes"}


def test_rule_planner_falls_back_to_query_all():
    # The question text has no note keyword, so P1 routes it to relational only;
    # no admission matches, and the fallback runs query_all, which adds notes.
    sources = make_sources(high=set())
    question = mixed_question()
    decision = RulePlanner().decide(question)
    run = run_with_fallback(decision, sources, question, FALLBACK_CONFIG)
    assert run.executed_plans == ("relational_only", "query_all")
    assert run.sources_queried == {"relational", "notes"}
    assert run.ok


# ---- empty results that answer a predicate ---------------------------------------------


def relational_question(flt):
    return QuestionSpec(qid="a", text="a", qclass="A", relational_filter=flt, required_sources=frozenset({"relational"}))


def test_empty_result_answers_a_predicate_question():
    # "Was creatinine ever above 2 in admission 5?" with no high value: no rows is the answer "no".
    from clinrag.question import StructuredFilter
    from tests.fakes import CREATININE

    sources = make_sources(high=set())
    question = relational_question(StructuredFilter(CREATININE, ">", 2.0, hadm_id=5))
    decision = OracleCostRouter(sources, FALLBACK_CONFIG).decide(question)
    run = run_with_fallback(decision, sources, question, FALLBACK_CONFIG)
    assert run.executed_plans == ("relational_only",)
    assert run.ok and run.attempts[0].reason == "empty result answers the predicate"
    assert run.items == [] and "search" not in sources.notes.calls


def test_empty_result_still_fails_without_a_predicate():
    # "Highest creatinine in admission 500?" needs a value: no rows is missing evidence.
    from clinrag.question import StructuredFilter
    from tests.fakes import CREATININE

    sources = make_sources(high=set())
    question = relational_question(StructuredFilter(CREATININE, hadm_id=500))
    decision = OracleCostRouter(sources, FALLBACK_CONFIG).decide(question)
    run = run_with_fallback(decision, sources, question, FALLBACK_CONFIG)
    assert run.attempts[0].reason == "no evidence items"
    assert run.executed_plans == ("relational_only", "query_all")


def test_empty_predicate_result_with_notes_in_the_question_still_falls_back():
    # The exemption is only for relational-only questions: a C question with an
    # empty filtered note search still widens (see test_empty_result_triggers_next_plan).
    decision, run = oracle_run(high=set())
    assert run.attempts[0].reason == "no evidence items" and run.fallbacks_used == 1
