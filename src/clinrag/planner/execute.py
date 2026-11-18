"""Run a plan step by step and measure what it actually cost."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from clinrag.config import PlannerConfig
from clinrag.planner.plan import GraphExpand, NoteSearch, Plan, RelationalFilter
from clinrag.question import QuestionSpec
from clinrag.sources.base import RelationalQuery, Sources
from clinrag.sources.tokens import count_tokens


@dataclass(frozen=True)
class StepRecord:
    """Measured outcome of one executed step. tokens counts only evidence it added."""

    step: str
    source: str
    latency_ms: float
    rows: int
    tokens: int


@dataclass(frozen=True)
class ExecutionResult:
    """Evidence for the LLM plus measured cost.

    items:  evidence in step order; each has "hadm_id", "text" and "source".
    tokens: count_tokens summed over the evidence text.
    """

    qid: str
    plan_name: str
    items: list[dict[str, Any]]
    tokens: int
    steps: list[StepRecord]
    total_latency_ms: float
    sources_queried: frozenset[str]


def execute(plan: Plan, sources: Sources, question: QuestionSpec, config: PlannerConfig) -> ExecutionResult:
    """Execute plan steps in order, passing ICD codes and admission ids forward."""
    start = time.perf_counter()
    codes: tuple[str, ...] = ()
    matching_ids: list[int] | None = None
    items: list[dict[str, Any]] = []
    records: list[StepRecord] = []

    for step in plan.steps:
        step_start = time.perf_counter()
        added: list[dict[str, Any]] = []

        if isinstance(step, GraphExpand):
            codes = tuple(sources.require_graph().expand(step.category))
            rows = len(codes)

        elif isinstance(step, RelationalFilter):
            query = RelationalQuery(lab=step.filter, icd_codes=codes if step.use_graph_codes else ())
            result = sources.require_relational().run(query)
            matching_ids = result.ids
            rows = len(result.items)
            if step.as_evidence:
                added = result.items

        elif isinstance(step, NoteSearch):
            added = _run_note_search(step, sources, matching_ids, config.top_k)
            rows = len(added)

        else:
            raise TypeError(f"Unknown step type {type(step).__name__}")

        tagged = [{**item, "source": step.source} for item in added]
        items.extend(tagged)
        records.append(
            StepRecord(
                step=step.label,
                source=step.source,
                latency_ms=(time.perf_counter() - step_start) * 1000.0,
                rows=rows,
                tokens=sum(count_tokens(item["text"]) for item in tagged),
            )
        )

    return ExecutionResult(
        qid=question.qid,
        plan_name=plan.name,
        items=items,
        tokens=sum(r.tokens for r in records),
        steps=records,
        total_latency_ms=(time.perf_counter() - start) * 1000.0,
        sources_queried=frozenset(r.source for r in records),
    )


def _run_note_search(
    step: NoteSearch, sources: Sources, matching_ids: list[int] | None, k: int
) -> list[dict[str, Any]]:
    notes = sources.require_notes()
    if step.mode == "unfiltered":
        return notes.search(step.query, k).items

    if matching_ids is None:
        raise ValueError(f"{step.mode} note search needs ids from an earlier relational step")
    if step.mode == "filter_first":
        return notes.search(step.query, k, hadm_ids=matching_ids).items

    if step.n_candidates is None:
        raise ValueError("search_first step has no n_candidates; plan it with CostPlanner first")
    allowed = set(matching_ids)
    candidates = notes.search(step.query, step.n_candidates).items
    return [item for item in candidates if item["hadm_id"] in allowed][:k]
