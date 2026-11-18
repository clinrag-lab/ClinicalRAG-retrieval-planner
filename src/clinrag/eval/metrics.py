"""Shared scoring for retrieval and answers.

Retrieval:  normalized_recall_at_k
Answers:    set_f1 (lists of admissions), numeric_match (class A values),
            yes_no_match (1/0 questions), and score_answer to pick one by question.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from clinrag.workload.schema import WorkloadQuestion

_YES = {"yes", "y", "true", "1"}
_NO = {"no", "n", "false", "0"}


def normalized_recall_at_k(retrieved_hadm_ids: Sequence[int], gold_hadm_ids: Iterable[int], k: int) -> float:
    """|top k retrieved ∩ gold| / min(k, |gold|).

    The top k are the first k retrieved items; an admission retrieved twice in
    the top k counts once. Dividing by min(k, |gold|) means a perfect top k
    scores 1.0 even when gold has more than k admissions. With empty gold, the
    score is 1.0 if nothing was retrieved and 0.0 otherwise.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    gold = set(gold_hadm_ids)
    if not gold:
        return 1.0 if len(retrieved_hadm_ids) == 0 else 0.0
    top = set(list(retrieved_hadm_ids)[:k])
    return len(top & gold) / min(k, len(gold))


def set_f1(predicted_ids: Iterable[int], gold_ids: Iterable[int]) -> float:
    """F1 between predicted and gold admission sets. Both empty scores 1.0."""
    predicted, gold = set(predicted_ids), set(gold_ids)
    if not predicted and not gold:
        return 1.0
    hits = len(predicted & gold)
    if hits == 0:
        return 0.0
    precision, recall = hits / len(predicted), hits / len(gold)
    return 2 * precision * recall / (precision + recall)


def numeric_match(predicted: Any, gold: float, rel_tol: float = 0.01) -> bool:
    """True if predicted is within rel_tol of gold (relative). Unparseable predictions never match."""
    value = _as_float(predicted)
    if value is None:
        return False
    return math.isclose(value, float(gold), rel_tol=rel_tol)


def yes_no_match(predicted: Any, gold: int) -> bool:
    """Exact match for yes/no answers. Accepts 1/0, True/False, or "yes"/"no" text."""
    value = _as_yes_no(predicted)
    return value is not None and value == int(gold)


@dataclass(frozen=True)
class Score:
    metric: str  # "numeric_match", "yes_no_match" or "set_f1"
    value: float


def score_answer(question: WorkloadQuestion, predicted: Any) -> Score:
    """Score an answer with the metric for the question's class and template.

    A, "ever above" templates   yes_no_match
    A, latest / max templates   numeric_match
    B, per-admission template   yes_no_match
    B list, C, D                set_f1 over predicted hadm_ids
    """
    qclass, template = question.spec.qclass, question.template_id
    gold = question.gold["answer"]
    if qclass == "A":
        if template.startswith("A_ever_above_"):
            return Score("yes_no_match", float(yes_no_match(predicted, gold)))
        return Score("numeric_match", float(numeric_match(predicted, gold)))
    if qclass == "B" and template.startswith("B_admission_"):
        return Score("yes_no_match", float(yes_no_match(predicted, gold)))
    if not isinstance(gold, list):
        raise ValueError(f"{question.spec.qid}: expected a list gold answer for template {template}")
    return Score("set_f1", set_f1(_as_ids(predicted), gold))


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_yes_no(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and value in (0, 1):
        return int(value)
    if isinstance(value, str):
        text = value.strip().lower().rstrip(".")
        if text in _YES:
            return 1
        if text in _NO:
            return 0
    return None


def _as_ids(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise TypeError("set answers must be an iterable of hadm_ids")
    return [int(v) for v in value]


# ---- retrieval scoring for the experiment runner -------------------------------------


def retrieval_score(question: WorkloadQuestion, items: Sequence[dict[str, Any]], k: int) -> Score:
    """How well retrieved evidence covers the gold answer, by question type.

    items: evidence in retrieval order, each with "hadm_id" and "source"
    ("relational" lab rows also carry "valuenum").

    A, latest / max     "value_row":   1.0 if a lab row of the gold admission with
                                       the gold value came back
    A, ever above       "value_row":   1.0 if whether any returned row passes the
                                       filter agrees with the gold yes/no
    B, per-admission    "gold_note":   1.0 if a note of the gold admission came back
    B list, C, D        "normalized_recall": normalized_recall_at_k over the hadm_ids
                                       of the note items, in rank order
    """
    spec, template = question.spec, question.template_id
    gold = question.gold["answer"]
    if spec.qclass == "A":
        flt = spec.relational_filter
        rows = [
            i for i in items
            if i.get("source") == "relational" and i.get("valuenum") is not None
            and (flt is None or flt.applies_to(int(i["hadm_id"])))
        ]
        if template.startswith("A_ever_above_"):
            passed = any(flt is not None and flt.matches(float(i["valuenum"])) for i in rows)
            return Score("value_row", float(passed == bool(gold)))
        return Score("value_row", float(any(math.isclose(float(i["valuenum"]), float(gold), rel_tol=1e-9) for i in rows)))
    notes = [int(i["hadm_id"]) for i in items if i.get("source") == "notes"]
    if spec.qclass == "B" and template.startswith("B_admission_"):
        return Score("gold_note", float(int(question.gold["hadm_id"]) in notes))
    return Score("normalized_recall", normalized_recall_at_k(notes, gold, k))
