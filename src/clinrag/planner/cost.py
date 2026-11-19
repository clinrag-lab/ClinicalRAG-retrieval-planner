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
    """Note search over a flat vector index.

    A flat index scores every note it searches, so unfiltered and search-first
    queries pay for all notes regardless of k; filter-first scores only the
    matching notes but pays to build and apply the admission id filter.

    scan(n)      = note_scan_ms_per_1k_notes * n / 1000
    fetch(n)     = note_fetch_ms_per_1k_candidates * n / 1000
    id_filter(n) = id_filter_ms_per_1k_ids * n / 1000

    latency:
      unfiltered    scan(total_notes) + fetch(k)
      unfiltered, scoped to one admission (question.note_hadm_id; probes.scoped_notes set)
                    id_filter(1) + scan(scoped_notes) + fetch(k)
      search_first  scan(total_notes) + fetch(min(ceil(k * total_notes / matching_notes), total_notes))
                    (fetch(total_notes) when matching_notes = 0)
      filter_first  id_filter(matching_admissions) + scan(matching_notes) + fetch(k)
    rows   = min(k, total_notes) for unfiltered (min(k, scoped_notes) when scoped),
             min(k, matching_notes) otherwise
    tokens = rows * avg_tokens_per_chunk
    """
    k = config.top_k
    total = _need(probes.total_notes, "total_notes")
    filter_ids = 0
    if step.mode == "unfiltered" and probes.scoped_notes is not None:
        # The executor restricts unfiltered searches to note_hadm_id: filter_first over one admission.
        filter_ids = 1
        scanned, fetched = probes.scoped_notes, k
        rows = min(k, probes.scoped_notes)
    elif step.mode == "unfiltered":
        scanned, fetched = total, k
        rows = min(k, total)
    else:
        matching = _need(probes.matching_notes, "matching_notes")
        if step.mode == "filter_first":
            filter_ids = _need(probes.matching_admissions, "matching_admissions")
            scanned, fetched = matching, k
        else:
            scanned, fetched = total, candidates_to_fetch(k, total, matching)
        rows = min(k, matching)

    latency = (
        _id_filter_ms(filter_ids, config) + _scan_ms(scanned, config) + _fetch_ms(fetched, config)
    )
    return ProbeEstimate(
        est_rows=rows,
        est_tokens=rows * config.avg_tokens_per_chunk,
        est_latency_ms=latency,
        details={"step": step.label, "scanned": scanned, "fetched": fetched, "filter_ids": filter_ids},
    )


def _scan_ms(n: int, config: PlannerConfig) -> float:
    return config.note_scan_ms_per_1k_notes * n / 1000.0


def _fetch_ms(n: int, config: PlannerConfig) -> float:
    return config.note_fetch_ms_per_1k_candidates * n / 1000.0


def _id_filter_ms(n: int, config: PlannerConfig) -> float:
    return config.id_filter_ms_per_1k_ids * n / 1000.0


def _need(value: int | None, name: str) -> int:
    if value is None:
        raise ValueError(f"Probes are missing {name}, which this plan needs")
    return value
