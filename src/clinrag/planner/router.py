"""Router interface shared by every planner the experiments compare, and the per-question log record.

A router turns a question into a PlanDecision. Routers differ in how they
decide which sources a question needs (true labels, keyword rules, a learned
classifier) and in whether they cost plans. The experiment runner executes the
decision and logs one RoutingResult per (router, question).
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol, runtime_checkable

from clinrag.config import PlannerConfig
from clinrag.planner.classifier import threshold
from clinrag.planner.cost import estimate_cost
from clinrag.planner.enumerate import QUERY_ALL, query_all_plan
from clinrag.planner.planner import CostPlanner, PlanDecision, ScoredPlan, cheapest_feasible
from clinrag.planner.probes import run_probes
from clinrag.question import ALL_SOURCES, GRAPH, NOTES, RELATIONAL, QuestionSpec
from clinrag.sources.base import Sources


@runtime_checkable
class Router(Protocol):
    """Decides a retrieval plan for a question.

    name: short id used in logs and plots.
    decide: returns the chosen plan, the candidates it considered (fallback
    picks from these) and planning_overhead_ms covering all decision time.
    """

    name: str

    def decide(self, question: QuestionSpec) -> PlanDecision: ...


class ExecutedRun(Protocol):
    """What RoutingResult needs from an execution: an ExecutionResult or a fallback run."""

    plan_name: str
    tokens: int
    total_latency_ms: float
    sources_queried: frozenset[str]


@dataclass(frozen=True)
class RoutingResult:
    """One logged row per (router, question).

    Decision fields:
      router, qid, qclass
      plan_name:            plan the router chose
      predicted_sources:    sources the router predicted the question needs
      required_sources:     the question's true sources, for scoring the prediction
      planning_overhead_ms: time spent deciding (probes, classifier, rules)
      est_cost:             estimated total cost of the chosen plan; None for
                            routers without a cost model
    Execution fields (summed over every attempt when fallback ran):
      tokens, latency_ms, sources_queried
      fallbacks_used:       plans executed after the first one
      executed_plans:       every plan executed, in order
    """

    router: str
    qid: str
    qclass: str
    plan_name: str
    predicted_sources: frozenset[str]
    required_sources: frozenset[str]
    planning_overhead_ms: float
    est_cost: float | None
    tokens: int
    latency_ms: float
    sources_queried: frozenset[str]
    fallbacks_used: int
    executed_plans: tuple[str, ...]

    @property
    def missed_sources(self) -> frozenset[str]:
        """Needed sources the router did not predict."""
        return self.required_sources - self.predicted_sources

    @classmethod
    def from_run(
        cls,
        router: str,
        question: QuestionSpec,
        decision: PlanDecision,
        run: ExecutedRun,
    ) -> RoutingResult:
        """Build the log row from a decision and its execution.

        run is an ExecutionResult (no fallback) or a FallbackResult, which also
        carries fallbacks_used and executed_plans.
        """
        cost = decision.chosen.cost
        return cls(
            router=router,
            qid=question.qid,
            qclass=question.qclass,
            plan_name=decision.plan.name,
            predicted_sources=decision.predicted_sources,
            required_sources=question.required_sources,
            planning_overhead_ms=decision.planning_overhead_ms,
            est_cost=None if cost is None else cost.total,
            tokens=run.tokens,
            latency_ms=run.total_latency_ms,
            sources_queried=run.sources_queried,
            fallbacks_used=int(getattr(run, "fallbacks_used", 0)),
            executed_plans=tuple(getattr(run, "executed_plans", (run.plan_name,))),
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready dict; source sets become sorted lists."""
        return {
            "router": self.router,
            "qid": self.qid,
            "qclass": self.qclass,
            "plan_name": self.plan_name,
            "predicted_sources": sorted(self.predicted_sources),
            "required_sources": sorted(self.required_sources),
            "missed_sources": sorted(self.missed_sources),
            "planning_overhead_ms": self.planning_overhead_ms,
            "est_cost": self.est_cost,
            "tokens": self.tokens,
            "latency_ms": self.latency_ms,
            "sources_queried": sorted(self.sources_queried),
            "fallbacks_used": self.fallbacks_used,
            "executed_plans": list(self.executed_plans),
        }


class OracleCostRouter:
    """CostPlanner given the question's true required_sources.

    UPPER BOUND FOR ANALYSIS, NOT A CONTENDER: no deployable router knows the
    true sources. Use it to measure how much cost the other routers lose to
    wrong source predictions.
    """

    name = "oracle_cost"

    def __init__(self, sources: Sources, config: PlannerConfig) -> None:
        self.planner = CostPlanner(sources, config)

    def decide(self, question: QuestionSpec) -> PlanDecision:
        return self.planner.plan(question)


# ---- routing a question to predicted sources ---------------------------------------

DEFAULT_PREFERENCE: tuple[str, ...] = (NOTES, RELATIONAL, GRAPH)


def restrict_question(
    question: QuestionSpec,
    sources: Iterable[str],
    preference: Sequence[str] = DEFAULT_PREFERENCE,
) -> QuestionSpec:
    """Copy of question that requires only the predicted sources it can actually run.

    The question's parsed inputs (filter, category, note query) stay as given;
    only the source set changes. A predicted source is kept when it is executable:
    - graph:      needs a category; also keeps relational, which consumes the codes.
    - relational: needs a relational_filter, or a category with graph kept.
    - notes:      always; searches note_query, or the question text when there is none.
    Inputs of dropped sources are cleared (category without graph, filter without
    relational). note_hadm_id is kept only on a notes-only copy.

    If nothing predicted is executable, the first source in preference that is
    executable on its own is used (notes always is).
    """
    keep = _executable(question, set(sources))
    for source in preference:
        if keep:
            break
        keep = _executable(question, {source})
    if not keep:
        raise ValueError(f"{question.qid}: no executable source among the prediction or {list(preference)}")

    category = question.category if GRAPH in keep else None
    relational_filter = question.relational_filter if RELATIONAL in keep else None
    note_query = question.note_query
    if NOTES in keep and not note_query:
        note_query = question.text
    return replace(
        question,
        required_sources=frozenset(keep),
        category=category,
        relational_filter=relational_filter,
        note_query=note_query,
        note_hadm_id=question.note_hadm_id if keep == {NOTES} else None,
    )


def _executable(question: QuestionSpec, predicted: set[str]) -> set[str]:
    keep: set[str] = set()
    if GRAPH in predicted and question.category is not None:
        keep |= {GRAPH, RELATIONAL}
    if RELATIONAL in predicted and question.relational_filter is not None:
        keep.add(RELATIONAL)
    if NOTES in predicted:
        keep.add(NOTES)
    return keep


def input_question(question: QuestionSpec) -> QuestionSpec:
    """Copy requiring every source the question gives an input for (the query_all scope)."""
    return restrict_question(question, ALL_SOURCES)


class SourcePredictor(Protocol):
    """What ClassifierCostRouter needs from a classifier (SourceClassifier or a test stub)."""

    last_latency_ms: float

    def predict_proba(self, question: QuestionSpec) -> dict[str, float]: ...


class ClassifierCostRouter:
    """P2: predict sources with a classifier, then let CostPlanner choose among them.

    decide():
    1. predict_proba on the question text; keep sources with p >= tau.
    2. restrict_question to those sources (most probable executable source if none).
    3. CostPlanner.plan on the restricted copy.
    4. Replace its query_all candidate with the query_all over every source the
       question has inputs for (costed with probes of that wider scope), so the
       fallback can still reach sources the classifier missed.
    planning_overhead_ms is the wall time of all four steps: classifier time
    plus planner time (plus the extra probes when step 4 applies).
    """

    def __init__(
        self,
        classifier: SourcePredictor,
        sources: Sources,
        config: PlannerConfig,
        tau: float = 0.5,
        name: str = "classifier_cost",
    ) -> None:
        if not 0.0 <= tau <= 1.0:
            raise ValueError("tau must be in [0, 1]")
        self.classifier = classifier
        self.sources = sources
        self.config = config
        self.tau = tau
        self.name = name
        self.planner = CostPlanner(sources, config)
        self.last_classifier_ms = 0.0

    def decide(self, question: QuestionSpec) -> PlanDecision:
        start = time.perf_counter()
        probs = self.classifier.predict_proba(question)
        self.last_classifier_ms = self.classifier.last_latency_ms
        predicted = threshold(probs, self.tau)
        preference = sorted(probs, key=lambda s: -probs[s])
        routed = restrict_question(question, predicted, preference)

        decision = self.planner.plan(routed)
        decision = self._with_input_query_all(question, decision)
        return replace(
            decision,
            qid=question.qid,
            predicted_sources=frozenset(predicted),
            planning_overhead_ms=(time.perf_counter() - start) * 1000.0,
        )

    def _with_input_query_all(self, question: QuestionSpec, decision: PlanDecision) -> PlanDecision:
        full = input_question(question)
        baseline = query_all_plan(full)
        if any(c.plan == baseline for c in decision.candidates):
            return decision
        probes = run_probes(full, self.sources, self.config)
        scored = ScoredPlan(baseline, estimate_cost(baseline, probes, self.config))
        candidates = tuple(scored if c.plan.name == QUERY_ALL else c for c in decision.candidates)
        return replace(decision, candidates=candidates, chosen=cheapest_feasible(candidates))
