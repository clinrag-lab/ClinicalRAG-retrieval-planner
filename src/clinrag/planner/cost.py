"""Plan cost model. All cost formulas live in this module.

    cost(plan) = est_latency_ms + alpha * est_tokens

summed over the plan's steps. Each step's estimate comes from the probes and the
constants in PlannerConfig; nothing is executed.
"""

from __future__ import annotations

from dataclasses import dataclass

from clinrag.config import PlannerConfig
from clinrag.planner.order import candidates_to_fetch
from clinrag.planner.plan import GraphExpand, NoteSearch, Plan, RelationalFilter, Step
from clinrag.planner.probes import Probes
from clinrag.sources.base import ProbeEstimate


@dataclass(frozen=True)
class CostEstimate:
    """Estimated cost of one plan, with the per-step breakdown."""

    est_latency_ms: float
    est_tokens: float
    total: float
    steps: tuple[ProbeEstimate, ...]


def estimate_cost(plan: Plan, probes: Probes, config: PlannerConfig) -> CostEstimate:
    """Estimate a plan's cost without running it.

    est_latency_ms = sum of step latencies
    est_tokens     = sum of step tokens
    total          = est_latency_ms + alpha * est_tokens
    """
    steps = tuple(_estimate_step(step, probes, config) for step in plan.steps)
    latency = sum(s.est_latency_ms for s in steps)
    tokens = sum(s.est_tokens for s in steps)
    return CostEstimate(
        est_latency_ms=latency,
        est_tokens=tokens,
        total=latency + config.alpha * tokens,
        steps=steps,
    )


def _estimate_step(step: Step, probes: Probes, config: PlannerConfig) -> ProbeEstimate:
    if isinstance(step, GraphExpand):
        return graph_step_cost(probes, config)
    if isinstance(step, RelationalFilter):
        return relational_step_cost(step, probes, config)
    if isinstance(step, NoteSearch):
        return note_step_cost(step, probes, config)
    raise TypeError(f"Unknown step type {type(step).__name__}")


def graph_step_cost(probes: Probes, config: PlannerConfig) -> ProbeEstimate:
    """Graph expansion.

    latency = graph_ms
    rows    = graph_codes
    tokens  = 0  (codes feed the relational filter; they are not sent to the LLM)
    """
    rows = _need(probes.graph_codes, "graph_codes")
    return ProbeEstimate(est_rows=rows, est_tokens=0.0, est_latency_ms=config.graph_ms, details={"step": "graph"})


def relational_step_cost(step: RelationalFilter, probes: Probes, config: PlannerConfig) -> ProbeEstimate:
    """Relational query.

    latency = relational_ms
    rows    = relational_rows
    tokens  = rows * avg_tokens_per_row if the rows are evidence, else 0
    """
    rows = _need(probes.relational_rows, "relational_rows")
    tokens = rows * config.avg_tokens_per_row if step.as_evidence else 0.0
    return ProbeEstimate(
        est_rows=rows,
        est_tokens=tokens,
        est_latency_ms=config.relational_ms,
        details={"step": step.label, "as_evidence": step.as_evidence},
    )


def note_step_cost(step: NoteSearch, probes: Probes, config: PlannerConfig) -> ProbeEstimate:
    """Note search. Latency scales with the number of candidates scored.

    candidates:
      unfiltered    min(k, total_notes)            index returns the top k
      filter_first  matching_notes                 restricted search scores every matching note
      search_first  min(ceil(k * total / matching), total)   (total if matching = 0)
    latency = note_search_ms_per_1k_candidates * candidates / 1000
    rows    = min(k, total_notes) for unfiltered, min(k, matching_notes) otherwise
    tokens  = rows * avg_tokens_per_chunk
    """
    k = config.top_k
    total = _need(probes.total_notes, "total_notes")
    if step.mode == "unfiltered":
        candidates = min(k, total)
        rows = min(k, total)
    else:
        matching = _need(probes.matching_notes, "matching_notes")
        if step.mode == "filter_first":
            candidates = matching
        else:
            candidates = candidates_to_fetch(k, total, matching)
        rows = min(k, matching)
    return ProbeEstimate(
        est_rows=rows,
        est_tokens=rows * config.avg_tokens_per_chunk,
        est_latency_ms=config.note_search_ms_per_1k_candidates * candidates / 1000.0,
        details={"step": step.label, "candidates": candidates},
    )


def _need(value: int | None, name: str) -> int:
    if value is None:
        raise ValueError(f"Probes are missing {name}, which this plan needs")
    return value
