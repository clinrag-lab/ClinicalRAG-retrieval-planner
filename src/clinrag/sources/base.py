"""Interfaces every retrieval source implements, plus the shared result types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from clinrag.question import StructuredFilter


@dataclass(frozen=True)
class ProbeEstimate:
    """Estimated rows, tokens and latency for one step, computed without running it."""

    est_rows: float
    est_tokens: float
    est_latency_ms: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class SourceResult:
    """What a source returns when actually queried.

    ids:   admission ids (hadm_id) in the result, without duplicates.
    items: evidence records; each has at least "hadm_id" and "text".
    """

    ids: list[int]
    items: list[dict[str, Any]]
    tokens: int
    latency_ms: float


@dataclass(frozen=True)
class RelationalQuery:
    """Structured query against the hospital tables.

    Semantics implementations must follow:
    - lab set: one result row per lab event matching the filter (itemid, the
      comparison if any, and lab.hadm_id if set); if icd_codes is also set,
      only admissions with at least one of those diagnosis codes.
    - lab unset, icd_codes set: one result row per diagnosis row with one of the codes.
    - neither set: invalid.
    icd_codes come from GraphSource.expand. The real ICD graph returns
    version-qualified codes ("10:I214", "9:4019"; see
    clinrag.workload.icd_categories.qualify), which must be matched on
    (icd_version, icd_code).
    """

    lab: StructuredFilter | None = None
    icd_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.lab is None and not self.icd_codes:
            raise ValueError("RelationalQuery needs a lab filter, ICD codes, or both")


class RelationalSource(ABC):
    """Structured hospital tables (DuckDB in the real system)."""

    @abstractmethod
    def count(self, query: RelationalQuery) -> int:
        """Number of rows run(query) would return, without fetching them."""

    @abstractmethod
    def matching_ids(self, query: RelationalQuery) -> list[int]:
        """Distinct hadm_ids that run(query) would return. Ids only, no row text."""

    @abstractmethod
    def run(self, query: RelationalQuery) -> SourceResult:
        """Execute the query and return its rows as evidence items."""


class NotesSource(ABC):
    """Vector search over discharge note chunks (FAISS in the real system)."""

    @abstractmethod
    def total_notes(self) -> int:
        """Number of note chunks in the index."""

    @abstractmethod
    def count_notes_for(self, hadm_ids: Sequence[int]) -> int:
        """Number of note chunks that belong to the given admissions."""

    @abstractmethod
    def probe_search(self, query: str, k_probe: int) -> list[float]:
        """Similarity scores of the top k_probe chunks, highest first. No text returned."""

    @abstractmethod
    def search(self, query: str, k: int, hadm_ids: Sequence[int] | None = None) -> SourceResult:
        """Top-k chunks for the query, restricted to hadm_ids when given.

        Items are ordered by similarity, highest first.
        """


class GraphSource(ABC):
    """ICD diagnosis-code hierarchy (in memory)."""

    @abstractmethod
    def expand(self, category: str) -> list[str]:
        """All ICD codes under a category such as "circulatory" or "heart_failure".

        Codes may be version-qualified ("10:I214"); see RelationalQuery.
        """

    @abstractmethod
    def descendant_count(self, category: str) -> int:
        """Number of codes expand(category) would return."""


@dataclass
class Sources:
    """The sources available to the planner and executor."""

    relational: RelationalSource | None = None
    notes: NotesSource | None = None
    graph: GraphSource | None = None

    def require_relational(self) -> RelationalSource:
        if self.relational is None:
            raise ValueError("This question or plan needs a relational source")
        return self.relational

    def require_notes(self) -> NotesSource:
        if self.notes is None:
            raise ValueError("This question or plan needs a notes source")
        return self.notes

    def require_graph(self) -> GraphSource:
        if self.graph is None:
            raise ValueError("This question or plan needs a graph source")
        return self.graph
