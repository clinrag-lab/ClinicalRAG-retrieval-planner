"""P1: keyword rules pick the sources, a fixed order picks the plan. No probes, no cost model."""

from __future__ import annotations

import time

from clinrag.planner.classifier import mentions_category, mentions_comparison, mentions_lab, mentions_notes
from clinrag.planner.enumerate import QUERY_ALL, enumerate_plans, query_all_plan
from clinrag.planner.plan import Plan
from clinrag.planner.planner import PlanDecision, ScoredPlan
from clinrag.planner.probes import Probes
from clinrag.planner.router import input_question, restrict_question
from clinrag.question import GRAPH, NOTES, RELATIONAL, QuestionSpec

RULE_PREFERENCE: tuple[str, ...] = (NOTES, RELATIONAL, GRAPH)


def rule_sources(text: str) -> set[str]:
    """Sources the keyword rules assign to a question.

    lab name or comparison                       -> relational
    ICD category name                            -> graph + relational
    "summary", "mention" or "documented"         -> notes
    May be empty; RulePlanner then searches notes.
    """
    sources: set[str] = set()
    if mentions_lab(text) or mentions_comparison(text):
        sources.add(RELATIONAL)
    if mentions_category(text):
        sources |= {GRAPH, RELATIONAL}
    if mentions_notes(text):
        sources.add(NOTES)
    return sources


class RulePlanner:
    """P1: rule_sources, then a fixed plan shape.

    Steps run graph -> relational -> notes. When a relational filter is combined
    with notes the note search is always filter_first (never search_first, never
    unfiltered). The question's parsed inputs are reused as in restrict_question;
    if no rule fires or no fired source is executable, it searches notes.

    The decision holds two uncosted candidates, the rule plan then query_all
    over every source the question has inputs for, so run_with_fallback can
    widen the search. est cost is None; planning_overhead_ms is the wall time
    of applying the rules and building the plan.
    """

    name = "rules"

    def decide(self, question: QuestionSpec) -> PlanDecision:
        start = time.perf_counter()
        predicted = rule_sources(question.text)
        routed = restrict_question(question, predicted, RULE_PREFERENCE)
        plan = self.plan_for(routed)
        baseline = query_all_plan(input_question(question))
        candidates = [ScoredPlan(plan, None)]
        if baseline != plan:
            candidates.append(ScoredPlan(baseline, None))
        return PlanDecision(
            qid=question.qid,
            chosen=candidates[0],
            candidates=tuple(candidates),
            probes=Probes(),
            order=None,
            planning_overhead_ms=(time.perf_counter() - start) * 1000.0,
            predicted_sources=frozenset(predicted),
        )

    @staticmethod
    def plan_for(question: QuestionSpec) -> Plan:
        """The fixed plan for a question's required sources (filter_first when mixed)."""
        targeted = [p for p in enumerate_plans(question) if p.name != QUERY_ALL]
        for plan in targeted:
            search = plan.note_search
            if search is None or search.mode == "filter_first" or not {RELATIONAL, NOTES} <= plan.sources:
                return plan
        raise ValueError(f"{question.qid}: no rule plan for {sorted(question.required_sources)}")
