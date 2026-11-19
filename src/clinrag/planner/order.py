"""Filter-order rule: restrict the note search first, or search first and filter after.

Selectivity rule:

    filter_first  if f == 0 or f / N < s
    search_first  otherwise

N = notes in the index, f = notes of the admissions passing the filter, s =
the selectivity threshold. beta is only the search-first over-fetch factor:
search-first fetches ceil(beta * k * N / f) candidates, since relevant notes
cluster outside the cohort when filter and query are correlated. Both are
PlannerConfig keys; python -m clinrag.eval.tune_order fits them on dev.

The previous rule (filter_first when beta * k * N / f > c * N) cancels N and
reduces to "f < beta * k / c": a fixed number of matching notes, so on a
larger index it chose search_first for ever more selective filters. f / N is
a fraction of the index and means the same at any index size.
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
    """

    order: Order
    candidates: int | None
    k: int
    total_notes: int
    matching_notes: int
    s: float
    beta: float

    @property
    def selectivity(self) -> float:
        """f / N, the fraction of the index that passes the filter."""
        return self.matching_notes / self.total_notes if self.total_notes else 0.0


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


def choose_order(k: int, total_notes: int, matching_notes: int, s: float = 0.1, beta: float = 1.0) -> OrderDecision:
    """filter_first if no notes match or the filter keeps less than a fraction s of the index, else search_first.

    The exact boundary f / N == s counts as search_first. The returned decision
    carries the beta-corrected search-first candidate count either way.
    """
    if not 0 < s <= 1:
        raise ValueError("s must be in (0, 1]")
    needed = search_first_candidates(k, total_notes, matching_notes, beta)
    selective = matching_notes == 0 or matching_notes < s * total_notes
    return OrderDecision(
        order="filter_first" if selective else "search_first",
        candidates=needed,
        k=k,
        total_notes=total_notes,
        matching_notes=matching_notes,
        s=s,
        beta=beta,
    )


def _check_counts(k: int, total_notes: int, matching_notes: int) -> None:
    if k < 1:
        raise ValueError("k must be at least 1")
    if total_notes < 0 or matching_notes < 0:
        raise ValueError("note counts must be non-negative")
    if matching_notes > total_notes:
        raise ValueError("matching_notes cannot exceed total_notes")
