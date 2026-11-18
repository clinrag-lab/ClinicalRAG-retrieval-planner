"""Filter-order rule: restrict the note search first, or search first and filter after."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Order = Literal["filter_first", "search_first"]


@dataclass(frozen=True)
class OrderDecision:
    """Result of choose_order, with its inputs kept for logging and plots.

    candidates: search-first candidates needed to keep about k notes after
                filtering; None when no notes match.
    """

    order: Order
    candidates: int | None
    k: int
    total_notes: int
    matching_notes: int
    max_candidates: int


def search_first_candidates(k: int, total_notes: int, matching_notes: int) -> int | None:
    """ceil(k * total_notes / matching_notes), or None when matching_notes is 0.

    If a fraction matching_notes / total_notes of all notes pass the filter,
    fetching this many unfiltered candidates keeps about k after filtering.
    """
    _check_counts(k, total_notes, matching_notes)
    if matching_notes == 0:
        return None
    return -(-k * total_notes // matching_notes)


def candidates_to_fetch(k: int, total_notes: int, matching_notes: int) -> int:
    """Candidates a search-first step actually fetches: capped at total_notes.

    With no matching notes, search-first would scan everything and keep nothing.
    """
    needed = search_first_candidates(k, total_notes, matching_notes)
    return total_notes if needed is None else min(needed, total_notes)


def choose_order(k: int, total_notes: int, matching_notes: int, max_candidates: int) -> OrderDecision:
    """Pick filter_first or search_first for a filtered note search.

    search_first if it needs at most max_candidates candidates; filter_first if it
    would need more, or if no notes match the filter.
    """
    if max_candidates < 1:
        raise ValueError("max_candidates must be at least 1")
    needed = search_first_candidates(k, total_notes, matching_notes)
    order: Order = "filter_first" if needed is None or needed > max_candidates else "search_first"
    return OrderDecision(
        order=order,
        candidates=needed,
        k=k,
        total_notes=total_notes,
        matching_notes=matching_notes,
        max_candidates=max_candidates,
    )


def _check_counts(k: int, total_notes: int, matching_notes: int) -> None:
    if k < 1:
        raise ValueError("k must be at least 1")
    if total_notes < 0 or matching_notes < 0:
        raise ValueError("note counts must be non-negative")
    if matching_notes > total_notes:
        raise ValueError("matching_notes cannot exceed total_notes")
