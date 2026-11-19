"""Cost-based planner: probe, enumerate, cost, apply the order rule, pick the cheapest."""

from __future__ import annotations

import time
from dataclasses import dataclass

from clinrag.config import PlannerConfig
from clinrag.planner.cost import CostEstimate, estimate_cost
from clinrag.planner.enumerate import enumerate_plans
from clinrag.planner.order import OrderDecision, candidates_to_fetch, choose_order
from clinrag.planner.plan import Plan
from clinrag.planner.probes import Probes, run_probes
from clinrag.question import NOTES, RELATIONAL, QuestionSpec
from clinrag.sources.base import Sources


@dataclass(frozen=True)
class ScoredPlan:
    """A candidate plan with its estimated cost.

    pruned_reason is set when the order rule ruled the plan out; pruned plans
    keep their cost for logging but are never chosen.
    """

    plan: Plan
    cost: CostEstimate
    pruned_reason: str | None = None

    @property
    def feasible(self) -> bool:
        return self.pruned_reason is None


@dataclass(frozen=True)
class PlanDecision:
    """Everything the planner decided for one question."""

    qid: str
    chosen: ScoredPlan
    candidates: tuple[ScoredPlan, ...]
    probes: Probes
    order: OrderDecision | None
    planning_overhead_ms: float

    @property
    def plan(self) -> Plan:
        return self.chosen.plan

    @property
    def order_costs(self) -> dict[str, float]:
        """Estimated total cost of each filter-order variant, including the pruned one.

        Example: {"filter_first": 120.0, "search_first": 121.5}. Empty when the
        question has no filtered note search.
        """
        costs: dict[str, float] = {}
        for scored in self.candidates:
            search = scored.plan.note_search
            if search is not None and scored.plan.is_mixed:
                costs[search.mode] = scored.cost.total
        return costs

    @property
    def cost_order(self) -> str | None:
        """Order the cost model alone would pick, to compare with order.order (the rule's pick)."""
        costs = self.order_costs
        return min(costs, key=costs.__getitem__) if costs else None


class CostPlanner:
    """Chooses the cheapest plan for a question.

    Steps: run probes; enumerate candidate plans; if the question mixes a
    relational filter with note search, apply choose_order and prune the
    filter_first/search_first variants it rules out; cost every plan; pick the
    cheapest feasible one (first in enumeration order on ties).
    """

    def __init__(self, sources: Sources, config: PlannerConfig) -> None:
        self.sources = sources
        self.config = config

    def plan(self, question: QuestionSpec) -> PlanDecision:
        start = time.perf_counter()
        probes = run_probes(question, self.sources, self.config)
        order = self._order(question, probes)

        scored: list[ScoredPlan] = []
        for plan in enumerate_plans(question):
            plan = self._fill_candidates(plan, probes)
            cost = estimate_cost(plan, probes, self.config)
            scored.append(ScoredPlan(plan, cost, self._pruned_reason(plan, order)))

        feasible = [s for s in scored if s.feasible]
        chosen = min(feasible, key=lambda s: s.cost.total)
        return PlanDecision(
            qid=question.qid,
            chosen=chosen,
            candidates=tuple(scored),
            probes=probes,
            order=order,
            planning_overhead_ms=(time.perf_counter() - start) * 1000.0,
        )

    def _order(self, question: QuestionSpec, probes: Probes) -> OrderDecision | None:
        if not {RELATIONAL, NOTES} <= question.required_sources:
            return None
        assert probes.total_notes is not None and probes.matching_notes is not None
        return choose_order(
            k=self.config.top_k,
            total_notes=probes.total_notes,
            matching_notes=probes.matching_notes,
            max_candidates=self.config.max_candidates,
        )

    def _fill_candidates(self, plan: Plan, probes: Probes) -> Plan:
        search = plan.note_search
        if search is None or search.mode != "search_first":
            return plan
        assert probes.total_notes is not None and probes.matching_notes is not None
        n = candidates_to_fetch(self.config.top_k, probes.total_notes, probes.matching_notes)
        return plan.with_candidates(n)

    @staticmethod
    def _pruned_reason(plan: Plan, order: OrderDecision | None) -> str | None:
        search = plan.note_search
        if order is None or search is None or not plan.is_mixed or search.mode == order.order:
            return None
        return f"order rule chose {order.order} (search-first candidates {order.candidates}, max {order.max_candidates})"
