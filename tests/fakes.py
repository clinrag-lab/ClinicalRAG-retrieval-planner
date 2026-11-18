"""Deterministic in-memory sources for planner tests.

No embeddings or randomness: every note has a fixed similarity score, and search
ignores the query text. Each fake records the operations called on it in .calls.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from clinrag.sources.base import GraphSource, NotesSource, RelationalQuery, RelationalSource, SourceResult
from clinrag.sources.tokens import count_tokens

CREATININE = 50912


class FakeRelational(RelationalSource):
    """labs: (hadm_id, itemid, valuenum) rows. diagnoses: (hadm_id, icd_code) rows."""

    def __init__(self, labs: Sequence[tuple[int, int, float]], diagnoses: Sequence[tuple[int, str]] = ()) -> None:
        self.labs = list(labs)
        self.diagnoses = list(diagnoses)
        self.calls: list[str] = []

    def _rows(self, query: RelationalQuery) -> list[dict]:
        codes = set(query.icd_codes)
        coded = {h for h, c in self.diagnoses if c in codes}
        if query.lab is not None:
            lab = query.lab
            return [
                {"hadm_id": h, "text": f"lab itemid {i} value {v}"}
                for h, i, v in self.labs
                if i == lab.itemid and lab.matches(v) and (not codes or h in coded)
            ]
        return [{"hadm_id": h, "text": f"diagnosis {c}"} for h, c in self.diagnoses if c in codes]

    def count(self, query: RelationalQuery) -> int:
        self.calls.append("count")
        return len(self._rows(query))

    def matching_ids(self, query: RelationalQuery) -> list[int]:
        self.calls.append("matching_ids")
        return sorted({r["hadm_id"] for r in self._rows(query)})

    def run(self, query: RelationalQuery) -> SourceResult:
        self.calls.append("run")
        rows = self._rows(query)
        return SourceResult(
            ids=sorted({r["hadm_id"] for r in rows}),
            items=rows,
            tokens=sum(count_tokens(r["text"]) for r in rows),
            latency_ms=0.0,
        )


@dataclass(frozen=True)
class FakeNote:
    note_id: str
    hadm_id: int
    text: str
    score: float


class FakeNotes(NotesSource):
    """Notes ranked by their fixed score (ties broken by note_id)."""

    def __init__(self, notes: Sequence[FakeNote]) -> None:
        self.notes = sorted(notes, key=lambda n: (-n.score, n.note_id))
        self.calls: list[str] = []
        self.search_args: list[tuple[int, tuple[int, ...] | None]] = []

    def total_notes(self) -> int:
        self.calls.append("total_notes")
        return len(self.notes)

    def count_notes_for(self, hadm_ids: Sequence[int]) -> int:
        self.calls.append("count_notes_for")
        wanted = set(hadm_ids)
        return sum(1 for n in self.notes if n.hadm_id in wanted)

    def probe_search(self, query: str, k_probe: int) -> list[float]:
        self.calls.append("probe_search")
        return [n.score for n in self.notes[:k_probe]]

    def search(self, query: str, k: int, hadm_ids: Sequence[int] | None = None) -> SourceResult:
        self.calls.append("search")
        self.search_args.append((k, None if hadm_ids is None else tuple(hadm_ids)))
        pool = self.notes if hadm_ids is None else [n for n in self.notes if n.hadm_id in set(hadm_ids)]
        hits = pool[:k]
        items = [{"hadm_id": n.hadm_id, "note_id": n.note_id, "text": n.text} for n in hits]
        return SourceResult(
            ids=sorted({n.hadm_id for n in hits}),
            items=items,
            tokens=sum(count_tokens(n.text) for n in hits),
            latency_ms=0.0,
        )


class FakeGraph(GraphSource):
    def __init__(self, hierarchy: dict[str, list[str]]) -> None:
        self.hierarchy = hierarchy
        self.calls: list[str] = []

    def expand(self, category: str) -> list[str]:
        self.calls.append("expand")
        return list(self.hierarchy[category])

    def descendant_count(self, category: str) -> int:
        self.calls.append("descendant_count")
        return len(self.hierarchy[category])


def grid_notes(n_admissions: int, notes_per_admission: int, text: str = "note text here") -> list[FakeNote]:
    """notes_per_admission notes for hadm_ids 1..n_admissions.

    Scores fall with position, so note (h=1, j=0) ranks first and the last
    admission's last note ranks last.
    """
    notes = []
    total = n_admissions * notes_per_admission
    for h in range(1, n_admissions + 1):
        for j in range(notes_per_admission):
            index = (h - 1) * notes_per_admission + j
            notes.append(FakeNote(f"{h}-{j}", h, text, score=1.0 - index / total))
    return notes


def creatinine_labs(n_admissions: int, high: set[int]) -> list[tuple[int, int, float]]:
    """One creatinine row per admission: 3.0 for admissions in high, 1.0 otherwise."""
    return [(h, CREATININE, 3.0 if h in high else 1.0) for h in range(1, n_admissions + 1)]
