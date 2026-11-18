"""Router interface shared by every planner the experiments compare, and the per-question log record.

A router turns a question into a PlanDecision. Routers differ in how they
decide which sources a question needs (true labels, keyword rules, a learned
classifier) and in whether they cost plans. The experiment runner executes the
decision and logs one RoutingResult per (router, question).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from clinrag.config import PlannerConfig
from clinrag.planner.planner import CostPlanner, PlanDecision
from clinrag.question import QuestionSpec
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
