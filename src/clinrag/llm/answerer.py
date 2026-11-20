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
    """answer(question, evidence_items, template_id) -> (answer, prompt_tokens, latency_ms).

    template_id: the question's workload template; it fixes the expected answer
            type (clinrag.eval.parse.answer_type). None when the caller has none.
    answer: the model's raw output text. The runner parses it with
            clinrag.eval.parse.parse_answer into a number (class A latest/max),
            1/0 (yes/no questions) or a list of hadm_ids (list questions) and
            scores that with clinrag.eval.metrics.score_answer. An answerer may
            also return such a value directly; None when no answer was produced.
    """

    name: str

    def answer(
        self, question: QuestionSpec, evidence_items: Sequence[dict[str, Any]], template_id: str | None = None
    ) -> tuple[Any, int, float]: ...


class NullAnswerer:
    """No answer step: answer None, 0 prompt tokens, 0 ms."""

    name = "none"

    def answer(
        self, question: QuestionSpec, evidence_items: Sequence[dict[str, Any]], template_id: str | None = None
    ) -> tuple[Any, int, float]:
        return None, 0, 0.0
