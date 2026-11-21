"""Turn an answerer's raw output into a value that clinrag.eval.metrics can score.

Shared by every answerer: the runner parses the text an answerer returns, so
all models are parsed and scored the same way.

    answer_type(template_id)        "number", "yesno" or "list"
    parse_answer(output, type)      ParsedAnswer(value, failed)
    score_parsed(question, parsed)  0.0 for no value, else score_answer

Parsing takes the first JSON object anywhere in the output ({"answer": ...},
so text around it is ignored) and falls back to a bare value. Values are
coerced by type:

    number  12, "1,250.5 mg/dL", "about 4.9"  -> float (units and commas dropped)
    yesno   true, "Yes.", "NO!", 1, "no, ..."  -> 1 or 0
    list    [1, "Admission 2"], "Admissions 1 and 2"  -> [1, 2] (ids in order, no duplicates)

A null answer ({"answer": null}, or a bare null/none) is the model declining:
value None, failed False. Output that yields no value of the expected type is
value None, failed True. Both score 0.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from clinrag.eval.metrics import score_answer
from clinrag.workload.schema import WorkloadQuestion

ANSWER_TYPES = ("number", "yesno", "list")

_NUMBER = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?|[-+]?\.\d+")
_INTEGER = re.compile(r"\d+")
_YES_NO = re.compile(r"\b(yes|no|true|false)\b")
_NULL = {"null", "none", "n/a", "na", "unknown"}
_YES = {"yes", "true"}


@dataclass(frozen=True)
class ParsedAnswer:
    """value: float, 1/0, list of hadm_ids, or None. failed: the output had no value of the expected type."""

    value: Any
    failed: bool


_DECLINED = ParsedAnswer(None, False)
_FAILED = ParsedAnswer(None, True)


def answer_type(template_id: str) -> str:
    """Expected answer type of a workload template, the same split score_answer uses.

    A_latest_*, A_max_*         number
    A_ever_above_*, B_admission_*  yesno
    B_list_*, C_*, D_*          list
    """
    if template_id.startswith(("A_latest_", "A_max_")):
        return "number"
    if template_id.startswith(("A_ever_above_", "B_admission_")):
        return "yesno"
    if template_id.startswith(("B_list_", "C_", "D_")):
        return "list"
    raise ValueError(f"no answer type for template {template_id!r}")


def first_json_object(text: str) -> dict[str, Any] | None:
    """The first JSON object that decodes, starting at any "{" in text."""
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            value, _ = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            value = None
        if isinstance(value, dict):
            return value
        start = text.find("{", start + 1)
    return None


def parse_answer(output: Any, expected: str) -> ParsedAnswer:
    """Parse an answerer's output as the expected type. See the module docstring."""
    if expected not in ANSWER_TYPES:
        raise ValueError(f"unknown answer type {expected!r}; expected one of {ANSWER_TYPES}")
    if output is None:
        return _DECLINED
    if not isinstance(output, str):
        return _coerce(output, expected)  # an answerer that already returns a value
    text = output.strip()
    if not text:
        return _FAILED
    obj = first_json_object(text)
    if obj is not None and "answer" in obj:
        return _coerce(obj["answer"], expected)
    return _coerce(_bare(text, expected), expected)


def _bare(text: str, expected: str) -> Any:
    """A bare value: a JSON array or scalar if the text is one, else the text itself."""
    if expected == "list":
        start = text.find("[")
        if start != -1:
            try:
                value, _ = json.JSONDecoder().raw_decode(text, start)
                return value
            except json.JSONDecodeError:
                pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _coerce(value: Any, expected: str) -> ParsedAnswer:
    if value is None:
        return _DECLINED
    if isinstance(value, str) and value.strip().strip(".!").lower() in _NULL:
        return _DECLINED
    parsed = {"number": _number, "yesno": _yes_no, "list": _ids}[expected](value)
    return _FAILED if parsed is None else ParsedAnswer(parsed, False)


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = _NUMBER.search(value)
        if match:
            return float(match.group().replace(",", ""))
    return None


def _yes_no(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value) if value in (0, 1) else None
    if isinstance(value, str):
        text = value.strip().lower()
        bare = text.strip(" .!?\"'")
        if bare in ("1", "0"):
            return int(bare)
        if bare in ("y", "n"):
            return int(bare == "y")
        match = _YES_NO.search(text)
        if match:
            return int(match.group(1) in _YES)
    return None


def _ids(value: Any) -> list[int] | None:
    """hadm_ids from a list of numbers or strings ("Admission 123"), or from one string naming them."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        value = [value]
    if isinstance(value, str):
        found = [int(m) for m in _INTEGER.findall(value)]
        return _unique(found) if found else None
    if not isinstance(value, (list, tuple)):
        return None
    out: list[int] = []
    for element in value:
        if isinstance(element, bool):
            return None
        if isinstance(element, int):
            out.append(element)
        elif isinstance(element, float) and element.is_integer():
            out.append(int(element))
        elif isinstance(element, str) and _INTEGER.search(element):
            out.extend(int(m) for m in _INTEGER.findall(element))
        else:
            return None
    return _unique(out)


def _unique(ids: list[int]) -> list[int]:
    return list(dict.fromkeys(ids))


def score_parsed(question: WorkloadQuestion, parsed: ParsedAnswer) -> float:
    """Answer score when an answerer was used: 0.0 for a null or unparseable answer, else score_answer."""
    if parsed.value is None:
        return 0.0
    return score_answer(question, parsed.value).value
