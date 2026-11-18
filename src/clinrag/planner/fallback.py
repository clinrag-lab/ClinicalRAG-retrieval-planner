"""Evidence check and fallback: if a plan's evidence looks wrong, run a wider plan.

check_evidence fails a result when
  - it has no items at all, unless it is a relational-only question whose filter
    is a predicate (op and threshold set): then an empty result is the answer
    ("was creatinine ever above 2?" -> no rows -> no), not missing evidence;
  - it searched notes but got no notes back, or its best note score is below
    config.min_similarity;
  - its note search was filtered (filter_first, search_first, or scoped to the
    question's note_hadm_id) and fewer than config.min_surviving_notes notes
    survived.

run_with_fallback executes the decision's chosen plan; on failure it executes
the cheapest untried feasible candidate whose sources strictly contain the
failed plan's sources, or query_all (skipping any candidate with the same steps
as a plan already run), and repeats up to max_fallbacks times.
Tokens and latency add up over every attempt.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from clinrag.config import PlannerConfig
from clinrag.planner.enumerate import QUERY_ALL
from clinrag.planner.execute import ExecutionResult, execute
from clinrag.planner.plan import Plan
from clinrag.planner.planner import PlanDecision, ScoredPlan
from clinrag.question import NOTES, RELATIONAL, QuestionSpec
from clinrag.sources.base import Sources

FILTERED_NOTE_STEPS = frozenset({"notes:filter_first", "notes:search_first"})


def empty_answers_predicate(result: ExecutionResult, question: QuestionSpec) -> bool:
    """True if an empty relational-only result is itself the answer to a predicate question."""
    flt = question.relational_filter
    return (
        question.required_sources == {RELATIONAL}
        and flt is not None
        and flt.op is not None
        and flt.threshold is not None
        and bool(result.steps)
        and all(step.source == RELATIONAL for step in result.steps)
    )


def check_evidence(result: ExecutionResult, question: QuestionSpec, config: PlannerConfig) -> tuple[bool, str]:
    """(ok, reason). reason is "ok" when the check passes."""
    if not result.items:
        if empty_answers_predicate(result, question):
            return True, "empty result answers the predicate"
        return False, "no evidence items"

    note_steps = [s for s in result.steps if s.source == NOTES]
    if not note_steps:
        return True, "ok"
    notes = [item for item in result.items if item.get("source") == NOTES]
    filtered = question.note_hadm_id is not None or any(s.step in FILTERED_NOTE_STEPS for s in note_steps)
    if filtered and len(notes) < config.min_surviving_notes:
        return False, f"{len(notes)} notes survived the filter, need {config.min_surviving_notes}"
    if not notes:
        return False, "note search returned no notes"
    scores = [float(item["score"]) for item in notes if item.get("score") is not None]
    if scores and max(scores) < config.min_similarity:
        return False, f"best note similarity {max(scores):.3f} below min_similarity {config.min_similarity}"
    return True, "ok"


@dataclass(frozen=True)
class Attempt:
    """One executed plan and its evidence check."""

    plan_name: str
    ok: bool
    reason: str
    tokens: int
    latency_ms: float
    sources_queried: frozenset[str]


@dataclass(frozen=True)
class FallbackResult:
    """Outcome of run_with_fallback.

    plan_name, items, final: the last plan executed and its evidence.
    tokens, total_latency_ms, sources_queried: summed (or united) over every attempt.
    ok: whether the last attempt passed the evidence check.
    """

    qid: str
    plan_name: str
    items: list[dict[str, Any]]
    tokens: int
    total_latency_ms: float
    sources_queried: frozenset[str]
    attempts: tuple[Attempt, ...]
    final: ExecutionResult

    @property
    def ok(self) -> bool:
        return self.attempts[-1].ok

    @property
    def fallbacks_used(self) -> int:
        return len(self.attempts) - 1

    @property
    def executed_plans(self) -> tuple[str, ...]:
        return tuple(a.plan_name for a in self.attempts)


def fallback_order(decision: PlanDecision) -> list[ScoredPlan]:
    """Feasible candidates, cheapest first; uncosted ones keep their candidate order, after costed ones."""
    indexed = [(i, s) for i, s in enumerate(decision.candidates) if s.feasible]
    indexed.sort(key=lambda pair: (pair[1].cost.total if pair[1].cost is not None else float("inf"), pair[0]))
    return [s for _, s in indexed]


def next_plan(order: Sequence[ScoredPlan], failed: Plan, tried: Sequence[Plan]) -> Plan | None:
    """Cheapest untried candidate that queries every source the failed plan did plus at least one more, or query_all.

    A candidate with the same steps as a tried plan counts as tried (for a
    notes-only question, query_all is the same search as notes_only).
    """
    for scored in order:
        plan = scored.plan
        if any(plan.name == t.name or plan.steps == t.steps for t in tried):
            continue
        if plan.name == QUERY_ALL or plan.sources > failed.sources:
            return plan
    return None


def run_with_fallback(
    decision: PlanDecision,
    sources: Sources,
    question: QuestionSpec,
    config: PlannerConfig,
    max_fallbacks: int | None = None,
) -> FallbackResult:
    """Execute decision.plan; on a failed evidence check, try wider plans.

    max_fallbacks defaults to config.max_fallbacks. Stops at the first passing
    check, after max_fallbacks extra plans, or when no wider candidate is left.
    """
    limit = config.max_fallbacks if max_fallbacks is None else max_fallbacks
    if limit < 0:
        raise ValueError("max_fallbacks must be non-negative")
    order = fallback_order(decision)
    plan = decision.plan
    tried: list[Plan] = []
    attempts: list[Attempt] = []
    results: list[ExecutionResult] = []

    while True:
        tried.append(plan)
        result = execute(plan, sources, question, config)
        ok, reason = check_evidence(result, question, config)
        results.append(result)
        attempts.append(
            Attempt(plan.name, ok, reason, result.tokens, result.total_latency_ms, result.sources_queried)
        )
        if ok or len(attempts) > limit:
            break
        wider = next_plan(order, plan, tried)
        if wider is None:
            break
        plan = wider

    final = results[-1]
    return FallbackResult(
        qid=question.qid,
        plan_name=final.plan_name,
        items=final.items,
        tokens=sum(r.tokens for r in results),
        total_latency_ms=sum(r.total_latency_ms for r in results),
        sources_queried=frozenset().union(*(r.sources_queried for r in results)),
        attempts=tuple(attempts),
        final=final,
    )
