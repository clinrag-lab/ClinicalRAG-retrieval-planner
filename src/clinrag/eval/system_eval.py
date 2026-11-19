"""ClinicalRAG end-to-end evaluation runner.

Copy to src/clinrag/eval/system_eval.py.

This module deliberately does not invent an LLM or infer final answers from
retrieved IDs. Provide a callable answer_fn(question, execution) -> prediction.
Without one, answer scores are null, while planning/retrieval metrics are saved.
"""
from __future__ import annotations

import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

from clinrag.config import PlannerConfig
from clinrag.eval.metrics import score_answer
from clinrag.planner.execute import ExecutionResult, execute
from clinrag.sources.base import Sources
from clinrag.workload.schema import WorkloadQuestion, read_jsonl

AnswerFn = Callable[[WorkloadQuestion, ExecutionResult], Any]


def evaluate(
    questions: Iterable[WorkloadQuestion],
    planner: Any,
    sources: Sources,
    config: PlannerConfig,
    *,
    answer_fn: AnswerFn | None = None,
    out_csv: str | Path = "data/eval/system_results.csv",
    out_json: str | Path = "data/eval/system_summary.json",
) -> dict[str, Any]:
    """Evaluate selected plans, executed evidence, and optional final answers.

    planner must implement plan(question.spec) -> PlanDecision.
    The callback must return a final prediction in the format expected by
    score_answer (number, yes/no, or list of admission IDs).
    """
    rows: list[dict[str, Any]] = []
    for question in questions:
        q = question.spec
        row: dict[str, Any] = {
            "qid": q.qid,
            "qclass": q.qclass,
            "question": q.text,
            "status": "ok",
            "error": "",
            "plan": "",
            "sources_queried": "",
            "predicted_sources": "",
            "planning_ms": None,
            "retrieval_ms": None,
            "generation_ms": None,
            "end_to_end_ms": None,
            "evidence_tokens": None,
            "evidence_items": None,
            "answer_metric": "",
            "answer_score": None,
            "prediction": "",
            "gold_answer": json.dumps(question.gold["answer"]),
        }
        started = time.perf_counter()
        try:
            decision = planner.plan(q)
            row["plan"] = decision.plan.name
            row["planning_ms"] = decision.planning_overhead_ms
            row["predicted_sources"] = ",".join(sorted(decision.predicted_sources))

            execution = execute(decision.plan, sources, q, config)
            row["retrieval_ms"] = execution.total_latency_ms
            row["sources_queried"] = ",".join(sorted(execution.sources_queried))
            row["evidence_tokens"] = execution.tokens
            row["evidence_items"] = len(execution.items)

            if answer_fn is not None:
                gen_start = time.perf_counter()
                prediction = answer_fn(question, execution)
                row["generation_ms"] = (time.perf_counter() - gen_start) * 1000
                score = score_answer(question, prediction)
                row["answer_metric"] = score.metric
                row["answer_score"] = score.value
                row["prediction"] = json.dumps(prediction, default=str)
        except Exception as exc:
            row["status"] = "error"
            row["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            row["end_to_end_ms"] = (time.perf_counter() - started) * 1000
            rows.append(row)

    out_csv, out_json = Path(out_csv), Path(out_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)