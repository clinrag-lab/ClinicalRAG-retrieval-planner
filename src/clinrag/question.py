"""Question specification shared by the workload, planner and sources."""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Callable, Literal

QClass = Literal["A", "B", "C", "D"]
Op = Literal["<", "<=", ">", ">=", "=", "!="]

RELATIONAL = "relational"
NOTES = "notes"
GRAPH = "graph"
ALL_SOURCES: frozenset[str] = frozenset({RELATIONAL, NOTES, GRAPH})

QCLASSES: frozenset[str] = frozenset({"A", "B", "C", "D"})

_OPS: dict[str, Callable[[float, float], bool]] = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "=": operator.eq,
    "!=": operator.ne,
}


@dataclass(frozen=True)
class StructuredFilter:
    """A condition on one lab item, e.g. creatinine (itemid 50912) > 2.0.

    op and threshold: both set for a comparison, or both None to match any
    value of the item.
    hadm_id: restrict to one admission (used by per-admission questions).
    """

    itemid: int
    op: Op | None = None
    threshold: float | None = None
    hadm_id: int | None = None

    def __post_init__(self) -> None:
        if (self.op is None) != (self.threshold is None):
            raise ValueError("op and threshold must be given together")
        if self.op is not None and self.op not in _OPS:
            raise ValueError(f"Unsupported op {self.op!r}; expected one of {sorted(_OPS)}")

    def matches(self, value: float | None) -> bool:
        """True if a lab value satisfies the comparison. Missing values never match."""
        if value is None:
            return False
        if self.op is None or self.threshold is None:
            return True
        return _OPS[self.op](value, self.threshold)

    def applies_to(self, hadm_id: int) -> bool:
        """True if rows from this admission are in scope."""
        return self.hadm_id is None or self.hadm_id == hadm_id


@dataclass(frozen=True, kw_only=True)
class QuestionSpec:
    """One clinical question and the inputs each source needs to answer it.

    Consistency rules (checked on construction):
    - "graph" is required exactly when a category is given.
    - "graph" needs "relational": graph output is ICD codes, used to filter admissions.
    - "relational" needs a relational_filter or a category.
    - "notes" needs a note_query.
    - A relational_filter is only allowed when "relational" is required.
    """

    qid: str
    text: str
    qclass: QClass
    relational_filter: StructuredFilter | None = None
    note_query: str | None = None
    category: str | None = None
    required_sources: frozenset[str]

    def __post_init__(self) -> None:
        sources = frozenset(self.required_sources)
        object.__setattr__(self, "required_sources", sources)

        if self.qclass not in QCLASSES:
            raise ValueError(f"{self.qid}: qclass must be one of {sorted(QCLASSES)}")
        if not sources:
            raise ValueError(f"{self.qid}: required_sources is empty")
        unknown = sources - ALL_SOURCES
        if unknown:
            raise ValueError(f"{self.qid}: unknown sources {sorted(unknown)}")
        if (GRAPH in sources) != (self.category is not None):
            raise ValueError(f"{self.qid}: 'graph' must be required exactly when a category is set")
        if GRAPH in sources and RELATIONAL not in sources:
            raise ValueError(f"{self.qid}: 'graph' requires 'relational'")
        if RELATIONAL in sources and self.relational_filter is None and self.category is None:
            raise ValueError(f"{self.qid}: 'relational' needs a relational_filter or a category")
        if self.relational_filter is not None and RELATIONAL not in sources:
            raise ValueError(f"{self.qid}: relational_filter given but 'relational' not required")
        if NOTES in sources and not self.note_query:
            raise ValueError(f"{self.qid}: 'notes' needs a note_query")
