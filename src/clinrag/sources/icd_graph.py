"""In-memory ICD hierarchy graph source: chapter -> block -> codes present in the data."""

from __future__ import annotations

from collections.abc import Iterable

import duckdb

from clinrag.sources.base import GraphSource
from clinrag.workload.icd_categories import CATEGORIES, IcdCategory, children, qualify


class IcdGraph(GraphSource):
    """Expands chapter or block categories into the ICD codes present in the data.

    expand() returns version-qualified codes ("10:I214", "9:4019") because the
    same code string can mean different things in ICD-9 and ICD-10. Relational
    sources match them on (icd_version, icd_code).

    Every category is expanded once at construction, so expand() is a list copy
    and descendant_count(), the planner's graph probe, is a dict lookup.
    """

    def __init__(self, codes: Iterable[tuple[str, int]], categories: dict[str, IcdCategory] = CATEGORIES) -> None:
        self.codes = sorted({(str(c).strip().upper(), int(v)) for c, v in codes})
        self.categories = categories
        self._pairs: dict[str, list[tuple[str, int]]] = {
            name: [(c, v) for c, v in self.codes if cat.contains(c, v)] for name, cat in categories.items()
        }
        self._expanded: dict[str, list[str]] = {
            name: [qualify(c, v) for c, v in pairs] for name, pairs in self._pairs.items()
        }
        self._counts: dict[str, int] = {name: len(pairs) for name, pairs in self._pairs.items()}

    @classmethod
    def from_connection(cls, con: duckdb.DuckDBPyConnection, table: str = "diagnoses_icd") -> IcdGraph:
        """Build from the distinct (icd_code, icd_version) pairs in a loaded table."""
        rows = con.execute(f"SELECT DISTINCT icd_code, icd_version FROM {table}").fetchall()
        return cls((str(code), int(version)) for code, version in rows)

    def category(self, name: str) -> IcdCategory:
        if name not in self.categories:
            raise KeyError(f"Unknown ICD category {name!r}; known: {sorted(self.categories)}")
        return self.categories[name]

    def expand_pairs(self, category: str) -> list[tuple[str, int]]:
        """(icd_code, icd_version) pairs in the category, sorted by (code, version)."""
        self.category(category)
        return list(self._pairs[category])

    def expand(self, category: str) -> list[str]:
        """Version-qualified codes in the category, precomputed at construction."""
        self.category(category)
        return list(self._expanded[category])

    def descendant_count(self, category: str) -> int:
        """Number of codes expand(category) returns, in O(1)."""
        self.category(category)
        return self._counts[category]

    def parent(self, category: str) -> str | None:
        return self.category(category).parent

    def children(self, category: str) -> list[str]:
        """Blocks under a chapter; empty for a block."""
        return children(category) if self.category(category).parent is None else []
