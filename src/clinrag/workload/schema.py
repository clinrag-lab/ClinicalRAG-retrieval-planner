"""Workload question records and their JSONL format."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from clinrag.question import QuestionSpec, StructuredFilter


@dataclass(frozen=True)
class WorkloadQuestion:
    """A QuestionSpec plus its gold answer.

    gold keys: "answer" (a number for class A, a sorted list of hadm_ids for
    B, C and D), "template_id", and for C and D "selectivity" (fraction of
    admissions passing the structured filter or category). Templates may add
    more keys (unit, threshold, hadm_id).
    """

    spec: QuestionSpec
    gold: dict[str, Any]

    @property
    def template_id(self) -> str:
        return str(self.gold["template_id"])

    def with_qid(self, qid: str) -> WorkloadQuestion:
        return WorkloadQuestion(replace(self.spec, qid=qid), self.gold)


def question_to_dict(q: WorkloadQuestion) -> dict[str, Any]:
    s = q.spec
    f = s.relational_filter
    return {
        "qid": s.qid,
        "text": s.text,
        "qclass": s.qclass,
        "relational_filter": None
        if f is None
        else {"itemid": f.itemid, "op": f.op, "threshold": f.threshold, "hadm_id": f.hadm_id, "agg": f.agg},
        "note_query": s.note_query,
        "category": s.category,
        "note_hadm_id": s.note_hadm_id,
        "required_sources": sorted(s.required_sources),
        "gold": q.gold,
    }


def template_agg(template_id: str) -> str | None:
    """The aggregation a class A template asks for: latest, max, or None (ever above T is a predicate)."""
    for agg in ("latest", "max"):
        if template_id.startswith(f"A_{agg}_"):
            return agg
    return None


def question_from_dict(d: dict[str, Any]) -> WorkloadQuestion:
    """Workload files written before filters had agg get it from the template id."""
    f = d.get("relational_filter")
    if f is not None and "agg" not in f:
        f = {**f, "agg": template_agg(str(d["gold"].get("template_id", "")))}
    spec = QuestionSpec(
        qid=d["qid"],
        text=d["text"],
        qclass=d["qclass"],
        relational_filter=None if f is None else StructuredFilter(**f),
        note_query=d.get("note_query"),
        category=d.get("category"),
        note_hadm_id=d.get("note_hadm_id"),
        required_sources=frozenset(d["required_sources"]),
    )
    return WorkloadQuestion(spec, d["gold"])


def write_jsonl(path: str | Path, questions: Iterable[WorkloadQuestion]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for q in questions:
            f.write(json.dumps(question_to_dict(q), sort_keys=True) + "\n")


def read_jsonl(path: str | Path) -> list[WorkloadQuestion]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [question_from_dict(json.loads(line)) for line in f if line.strip()]
