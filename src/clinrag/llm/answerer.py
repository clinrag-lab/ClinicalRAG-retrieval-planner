"""Answer step slot: turns retrieved evidence into an answer.

The experiment runner calls an Answerer after retrieval. NullAnswerer returns
no answer, so the runner reports retrieval metrics only. The team's local LLM
(Llama-3.1-8B-Instruct) plugs in by implementing Answerer; nothing here loads
a model.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from clinrag.question import QuestionSpec


class Answerer(Protocol):
    """answer(question, evidence_items) -> (answer, prompt_tokens, latency_ms).

    answer: a number (class A latest/max), 1/0 or "yes"/"no" (yes/no questions),
            or a list of hadm_ids (list questions), as scored by
            clinrag.eval.metrics.score_answer; None when no answer was produced.
    """

    name: str

    def answer(self, question: QuestionSpec, evidence_items: Sequence[dict[str, Any]]) -> tuple[Any, int, float]: ...


class NullAnswerer:
    """No answer step: answer None, 0 prompt tokens, 0 ms."""

    name = "none"

    def answer(self, question: QuestionSpec, evidence_items: Sequence[dict[str, Any]]) -> tuple[Any, int, float]:
        return None, 0, 0.0
