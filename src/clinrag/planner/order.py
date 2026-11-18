"""Filter-order rule: restrict the note search first, or search first and filter after.

Relative rule with a correlation correction:

    candidates = ceil(beta * k * N / f)        N = notes in the index, f = matching notes
    filter_first  if f == 0 or candidates > c * N
    search_first  otherwise

k * N / f is how many unfiltered candidates keep about k matching notes if
matching notes were spread evenly through the ranking. beta >= 1 corrects for
the filter and the query being correlated (relevant notes cluster outside the
cohort), so search-first fetches more. c is the largest fraction of the index
search-first may fetch before filter-first is preferred. Both are PlannerConfig
keys; python -m clinrag.eval.tune_order fits them on dev.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

Order = Literal["filter_first", "search_first"]


@dataclass(frozen=True)
class OrderDecision:
    """Result of choose_order, with its inputs kept for logging and plots.

    candidates: beta-corrected search-first candidates needed to keep about k
                notes after filtering; None when no notes match.
    limit:      c * total_notes, the most candidates search-first may fetch.
    """

    order: Order
    candidates: int | None
    k: int
    total_notes: int
    matching_notes: int
    c: float
    beta: float

    @property
    def limit(self) -> float:
        return self.c * self.total_notes


def search_first_candidates(k: int, total_notes: int, matching_notes: int, beta: float = 1.0) -> int | None:
    """ceil(beta * k * total_notes / matching_notes), or None when matching_notes is 0.

    If a fraction matching_notes / total_notes of all notes pass the filter,
    fetching k * total / matching unfiltered candidates keeps about k after
    filtering; beta scales that up for correlated filters.
    """
    _check_counts(k, total_notes, matching_notes)
    if beta <= 0:
        raise ValueError("beta must be positive")
    if matching_notes == 0:
        return None
    # Integer arithmetic when beta is whole, so exact boundaries are not lost to float rounding.
    if float(beta).is_integer():
        return -(-int(beta) * k * total_notes // matching_notes)
    return math.ceil(beta * k * total_notes / matching_notes)


def candidates_to_fetch(k: int, total_notes: int, matching_notes: int, beta: float = 1.0) -> int:
    """Candidates a search-first step actually fetches: beta-corrected, capped at total_notes.

    With no matching notes, search-first would scan everything and keep nothing.
    """
    needed = search_first_candidates(k, total_notes, matching_notes, beta)
    return total_notes if needed is None else min(needed, total_notes)


def choose_order(k: int, total_notes: int, matching_notes: int, c: float = 0.25, beta: float = 1.0) -> OrderDecision:
    """Pick filter_first or search_first for a filtered note search.

    filter_first if no notes match or search-first needs more than c * total_notes
    candidates; search_first otherwise (the exact boundary counts as search_first).
    """
    if c <= 0:
        raise ValueError("c must be positive")
    needed = search_first_candidates(k, total_notes, matching_notes, beta)
    order: Order = "filter_first" if needed is None or needed > c * total_notes else "search_first"
    return OrderDecision(
        order=order,
        candidates=needed,
        k=k,
        total_notes=total_notes,
        matching_notes=matching_notes,
        c=c,
        beta=beta,
    )


def switch_matching_notes(k: int, c: float, beta: float = 1.0) -> float:
    """Matching-note count at which the rule switches: search_first iff f >= beta * k / c."""
    return beta * k / c


def _check_counts(k: int, total_notes: int, matching_notes: int) -> None:
    if k < 1:
        raise ValueError("k must be at least 1")
    if total_notes < 0 or matching_notes < 0:
        raise ValueError("note counts must be non-negative")
    if matching_notes > total_notes:
        raise ValueError("matching_notes cannot exceed total_notes")
