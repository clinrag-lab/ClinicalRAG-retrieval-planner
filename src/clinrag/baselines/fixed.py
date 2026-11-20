"""Fixed-order baselines as routers.

Each baseline uses the question's true required sources, exactly like
OracleCostRouter, and runs the same probes and cost model. Only the final pick
is forced, so the comparison isolates the effect of order or coverage:

  QueryAllRouter            always query_all: every source with an input, notes unfiltered
  AlwaysFilterFirstRouter   filter_first whenever a structured filter feeds a note search
  AlwaysSearchFirstRouter   search_first whenever a structured filter feeds a note search

For questions without a filtered note search (classes A and B) the two order
baselines pick what the oracle picks. A forced plan may be one the order rule
pruned; it is executed anyway, and its candidate list (used by the fallback)
is unchanged. The LLM router and vector-only RAG baselines are left to the team.
"""

from __future__ import annotations

from dataclasses import replace

from clinrag.config import PlannerConfig
from clinrag.planner.enumerate import QUERY_ALL
from clinrag.planner.planner import CostPlanner, PlanDecision, ScoredPlan
from clinrag.question import QuestionSpec
from clinrag.sources.base import Sources


class _ForcedPlanRouter:
    name = "forced"

    def __init__(self, sources: Sources, config: PlannerConfig) -> None:
        self.planner = CostPlanner(sources, config)

    def pick(self, decision: PlanDecision) -> ScoredPlan | None:
        raise NotImplementedError

    def decide(self, question: QuestionSpec) -> PlanDecision:
        decision = self.planner.plan(question)
        forced = self.pick(decision)
        if forced is None or forced is decision.chosen:
            return decision
        # Executed as chosen even if the order rule pruned it.
        return replace(decision, chosen=replace(forced, pruned_reason=None))


class QueryAllRouter(_ForcedPlanRouter):
    """Always the query_all baseline plan."""

    name = "query_all"

    def pick(self, decision: PlanDecision) -> ScoredPlan | None:
        return next(c for c in decision.candidates if c.plan.name == QUERY_ALL)


class _ForcedOrderRouter(_ForcedPlanRouter):
    mode = ""

    def pick(self, decision: PlanDecision) -> ScoredPlan | None:
        for scored in decision.candidates:
            search = scored.plan.note_search
            if scored.plan.is_mixed and search is not None and search.mode == self.mode:
                return scored
        return None  # no filtered note search: keep the oracle's pick


class AlwaysFilterFirstRouter(_ForcedOrderRouter):
    """filter_first for every filtered note search, regardless of the order rule or cost."""

    name = "always_filter_first"
    mode = "filter_first"


class AlwaysSearchFirstRouter(_ForcedOrderRouter):
    """search_first for every filtered note search, regardless of the order rule or cost."""

    name = "always_search_first"
    mode = "search_first"
