"""Plan representation: an ordered list of steps over the three sources."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import ClassVar, Literal, Union

from clinrag.question import GRAPH, NOTES, RELATIONAL, StructuredFilter

SearchMode = Literal["unfiltered", "filter_first", "search_first"]
SEARCH_MODES: frozenset[str] = frozenset({"unfiltered", "filter_first", "search_first"})


@dataclass(frozen=True)
class GraphExpand:
    """Expand a diagnosis category into ICD codes for a later relational step."""

    category: str
    source: ClassVar[str] = GRAPH

    @property
    def label(self) -> str:
        return "graph"


@dataclass(frozen=True)
class RelationalFilter:
    """Run the structured query.

    use_graph_codes: restrict to the ICD codes produced by an earlier GraphExpand.
    as_evidence:     send the rows to the LLM. False when the step only serves
                     as a filter for a later note search.
    """

    filter: StructuredFilter | None
    use_graph_codes: bool = False
    as_evidence: bool = True
    source: ClassVar[str] = RELATIONAL

    @property
    def label(self) -> str:
        return "relational"


@dataclass(frozen=True)
class NoteSearch:
    """Vector search over note chunks.

    mode:
      unfiltered    search all notes, keep the top k.
      filter_first  search only notes of admissions from the earlier relational step.
      search_first  fetch n_candidates from all notes, drop non-matching admissions,
                    keep the top k.
    n_candidates: only for search_first; filled in by the planner.
    """

    query: str
    mode: SearchMode = "unfiltered"
    n_candidates: int | None = None
    source: ClassVar[str] = NOTES

    def __post_init__(self) -> None:
        if self.mode not in SEARCH_MODES:
            raise ValueError(f"Unknown search mode {self.mode!r}")
        if self.n_candidates is not None and self.mode != "search_first":
            raise ValueError("n_candidates only applies to search_first")

    @property
    def label(self) -> str:
        return f"notes:{self.mode}"


Step = Union[GraphExpand, RelationalFilter, NoteSearch]


@dataclass(frozen=True)
class Plan:
    """A candidate retrieval plan.

    name:  short readable id used in logs and plots.
    steps: executed in order; ids and codes flow from earlier steps to later ones.
    """

    name: str
    steps: tuple[Step, ...]

    def __post_init__(self) -> None:
        if not self.steps:
            raise ValueError(f"Plan {self.name!r} has no steps")
        seen_graph = seen_relational = False
        for step in self.steps:
            if isinstance(step, GraphExpand):
                seen_graph = True
            elif isinstance(step, RelationalFilter):
                if step.use_graph_codes and not seen_graph:
                    raise ValueError(f"Plan {self.name!r}: use_graph_codes before any GraphExpand")
                seen_relational = True
            elif isinstance(step, NoteSearch):
                if step.mode != "unfiltered" and not seen_relational:
                    raise ValueError(f"Plan {self.name!r}: {step.mode} needs an earlier RelationalFilter")
            else:
                raise TypeError(f"Unknown step type {type(step).__name__}")

    @property
    def sources(self) -> frozenset[str]:
        """Sources this plan queries."""
        return frozenset(step.source for step in self.steps)

    @property
    def note_search(self) -> NoteSearch | None:
        for step in self.steps:
            if isinstance(step, NoteSearch):
                return step
        return None

    @property
    def is_mixed(self) -> bool:
        """True if a structured filter restricts the note search (filter_first or search_first)."""
        search = self.note_search
        return search is not None and search.mode != "unfiltered"

    def with_candidates(self, n_candidates: int) -> Plan:
        """Copy of a search_first plan with its candidate count set."""
        search = self.note_search
        if search is None or search.mode != "search_first":
            raise ValueError(f"Plan {self.name!r} has no search_first step")
        steps = tuple(
            replace(step, n_candidates=n_candidates) if step is search else step for step in self.steps
        )
        return Plan(name=self.name, steps=steps)
