"""Reference source implementations over the open MIMIC-IV demo and the generated notes.

These are reference implementations for smoke tests and as a fallback. The
team's DuckDB relational store and FAISS note index replace them; anything
that implements the interfaces in sources/base.py plugs into the planner and
the experiment runner the same way.

- DemoRelationalSource: DuckDB over the hosp tables (loaded with workload/tables.py).
- TfidfNotesSource:     TF-IDF (scikit-learn) over the discharge notes, one chunk
                        per note, brute-force cosine scoring.
- IcdGraph:             the existing ICD hierarchy (sources/icd_graph.py).

Usage:
    sources = build_reference_sources("data/mimic-iv-demo", "data/generated")
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from clinrag.sources.base import NotesSource, RelationalQuery, RelationalSource, SourceResult, Sources
from clinrag.sources.icd_graph import IcdGraph
from clinrag.sources.tokens import count_tokens
from clinrag.workload.icd_categories import unqualify
from clinrag.workload.tables import connect_tables, has_table, table_path

_SQL_OPS = {"<": "<", "<=": "<=", ">": ">", ">=": ">=", "=": "=", "!=": "<>"}

# Row order of a lab query by StructuredFilter.agg. An aggregate keeps one row.
_LAB_ORDER = {
    None: "ORDER BY l.hadm_id, l.charttime, l.labevent_id",
    "latest": "ORDER BY l.charttime DESC, l.labevent_id DESC LIMIT 1",
    "max": "ORDER BY l.valuenum DESC, l.charttime DESC, l.labevent_id DESC LIMIT 1",
}


class DemoRelationalSource(RelationalSource):
    """RelationalQuery over the hosp tables in a DuckDB connection.

    Semantics match how the workload's gold answers were computed:
    - lab rows: labevents rows of admissions in the admissions table with a
      non-null hadm_id and numeric valuenum, matching itemid, the comparison on
      valuenum (any row of an admission can pass, as in class C gold), and
      lab.hadm_id when set;
    - lab.agg: of those rows, exactly one (none if nothing matches). "latest" is
      the row with the greatest charttime, "max" the row with the highest
      valuenum; ties go to the later charttime, then the higher labevent_id,
      as in class A gold (workload/tables.py lab_summary). count() is then at
      most 1; matching_ids and the note counts are unchanged, since agg needs
      lab.hadm_id and the row comes from that admission;
    - icd_codes: version-qualified codes ("10:I214"), matched on
      (icd_version, icd_code) via unqualify();
    - lab and codes: lab rows of admissions with at least one of the codes;
    - codes only: one row per matching diagnoses_icd row.

    Lab items carry hadm_id, text, itemid, label, valuenum, unit, charttime;
    diagnosis items carry hadm_id, text, icd_code, icd_version.
    """

    def __init__(self, con: duckdb.DuckDBPyConnection) -> None:
        self.con = con
        self._lab_labels = self._labels("d_labitems", "itemid", "label")
        self._dx_titles = self._dx_titles_map()

    @classmethod
    def from_tables(cls, tables_dir: str | Path) -> DemoRelationalSource:
        con = connect_tables(
            tables_dir,
            required=("admissions", "diagnoses_icd", "labevents"),
            optional=("d_labitems", "d_icd_diagnoses"),
        )
        return cls(con)

    def _labels(self, table: str, key: str, value: str) -> dict[Any, str]:
        if not has_table(self.con, table):
            return {}
        return {k: str(v) for k, v in self.con.execute(f"SELECT {key}, {value} FROM {table}").fetchall()}

    def _dx_titles_map(self) -> dict[tuple[str, int], str]:
        if not has_table(self.con, "d_icd_diagnoses"):
            return {}
        rows = self.con.execute("SELECT icd_code, icd_version, long_title FROM d_icd_diagnoses").fetchall()
        return {(str(c).strip().upper(), int(v)): str(t) for c, v, t in rows}

    @staticmethod
    def _codes(query: RelationalQuery) -> list[str]:
        """Codes as "version:CODE" keys, the same form as qualify()."""
        keys = []
        for qualified in query.icd_codes:
            code, version = unqualify(qualified)
            keys.append(f"{version}:{code}")
        return keys

    def _from_where(self, query: RelationalQuery) -> tuple[str, list[Any], bool]:
        """(FROM ... WHERE ..., params, is_lab)."""
        params: list[Any] = []
        dx_key = "CAST(d.icd_version AS VARCHAR) || ':' || upper(trim(d.icd_code))"
        if query.lab is not None:
            lab = query.lab
            sql = (
                "FROM labevents l JOIN admissions a ON a.hadm_id = l.hadm_id "
                "WHERE l.hadm_id IS NOT NULL AND l.valuenum IS NOT NULL AND l.itemid = ?"
            )
            params.append(lab.itemid)
            if lab.hadm_id is not None:
                sql += " AND l.hadm_id = ?"
                params.append(lab.hadm_id)
            if lab.op is not None:
                sql += f" AND l.valuenum {_SQL_OPS[lab.op]} ?"
                params.append(lab.threshold)
            if query.icd_codes:
                sql += f" AND l.hadm_id IN (SELECT d.hadm_id FROM diagnoses_icd d WHERE list_contains(?, {dx_key}))"
                params.append(self._codes(query))
            return sql, params, True
        sql = f"FROM diagnoses_icd d JOIN admissions a ON a.hadm_id = d.hadm_id WHERE list_contains(?, {dx_key})"
        return sql, [self._codes(query)], False

    def count(self, query: RelationalQuery) -> int:
        sql, params, _ = self._from_where(query)
        n = int(self.con.execute(f"SELECT count(*) {sql}", params).fetchone()[0])
        return min(n, 1) if query.lab is not None and query.lab.agg is not None else n

    def matching_ids(self, query: RelationalQuery) -> list[int]:
        sql, params, is_lab = self._from_where(query)
        col = "l.hadm_id" if is_lab else "d.hadm_id"
        return [int(r[0]) for r in self.con.execute(f"SELECT DISTINCT {col} {sql} ORDER BY 1", params).fetchall()]

    # ---- note counts in SQL (optional probes) ----

    def register_notes(self, note_ids: Sequence[str], hadm_ids: Sequence[int]) -> None:
        """Load the note index's metadata (note_id, hadm_id; never text) as DuckDB tables.

        note_meta has one row per indexed chunk; note_counts has chunks per admission.
        """
        frame = pd.DataFrame({"note_id": [str(n) for n in note_ids], "hadm_id": [int(h) for h in hadm_ids]})
        self.con.register("_note_meta_frame", frame)
        self.con.execute("CREATE OR REPLACE TABLE note_meta AS SELECT note_id, hadm_id FROM _note_meta_frame")
        self.con.unregister("_note_meta_frame")
        self.con.execute(
            "CREATE OR REPLACE TABLE note_counts AS SELECT hadm_id, count(*) AS n FROM note_meta GROUP BY hadm_id"
        )
        self._notes_registered = True

    def _require_notes(self) -> None:
        if not getattr(self, "_notes_registered", False):
            raise NotImplementedError("no note metadata registered; call register_notes first")

    def count_notes(self, query: RelationalQuery) -> int:
        self._require_notes()
        sql, params, is_lab = self._from_where(query)
        col = "l.hadm_id" if is_lab else "d.hadm_id"
        row = self.con.execute(
            f"SELECT coalesce(sum(c.n), 0) FROM note_counts c WHERE c.hadm_id IN (SELECT {col} {sql})", params
        ).fetchone()
        return int(row[0])

    def count_matches(self, query: RelationalQuery) -> tuple[int, int, int]:
        self._require_notes()
        sql, params, is_lab = self._from_where(query)
        col = "l.hadm_id" if is_lab else "d.hadm_id"
        row = self.con.execute(
            f"""
            WITH m AS MATERIALIZED (SELECT {col} AS h {sql}),
                 a AS (SELECT DISTINCT h FROM m)
            SELECT (SELECT count(*) FROM m),
                   (SELECT count(*) FROM a),
                   (SELECT coalesce(sum(c.n), 0) FROM note_counts c JOIN a ON c.hadm_id = a.h)
            """,
            params,
        ).fetchone()
        return int(row[0]), int(row[1]), int(row[2])

    def run(self, query: RelationalQuery) -> SourceResult:
        sql, params, is_lab = self._from_where(query)
        items: list[dict[str, Any]] = []
        if is_lab:
            assert query.lab is not None
            rows = self.con.execute(
                f"SELECT l.hadm_id, l.itemid, l.valuenum, l.valueuom, l.charttime {sql} {_LAB_ORDER[query.lab.agg]}",
                params,
            ).fetchall()
            for hadm, itemid, value, uom, charttime in rows:
                label = self._lab_labels.get(itemid, f"item {itemid}")
                unit = f" {uom}" if uom else ""
                items.append({
                    "hadm_id": int(hadm),
                    "itemid": int(itemid),
                    "label": label,
                    "valuenum": float(value),
                    "unit": str(uom) if uom else "",
                    "charttime": str(charttime),
                    "text": f"Admission {hadm}: {label} {value:g}{unit} at {charttime}",
                })
        else:
            rows = self.con.execute(
                f"SELECT d.hadm_id, d.icd_code, d.icd_version {sql} ORDER BY d.hadm_id, d.seq_num", params
            ).fetchall()
            for hadm, code, version in rows:
                key = (str(code).strip().upper(), int(version))
                title = self._dx_titles.get(key, "")
                items.append({
                    "hadm_id": int(hadm),
                    "icd_code": key[0],
                    "icd_version": key[1],
                    "text": f"Admission {hadm}: diagnosis ICD-{version} {key[0]} {title}".rstrip(),
                })
        ids = sorted({item["hadm_id"] for item in items})
        return SourceResult(ids=ids, items=items, tokens=sum(count_tokens(i["text"]) for i in items), latency_ms=0.0)


class TfidfNotesSource(NotesSource):
    """TF-IDF vectors of the discharge notes, one chunk per note, cosine similarity.

    search(query, k) scores every note (brute force); search(query, k, hadm_ids)
    first builds the row list of the given admissions (the id filter), then
    scores only those notes. Items carry hadm_id, note_id, text and score
    (cosine similarity, highest first).
    """

    def __init__(self, notes: pd.DataFrame) -> None:
        notes = notes.reset_index(drop=True)
        self.note_ids = notes["note_id"].astype(str).tolist()
        self.hadm_ids = notes["hadm_id"].astype(int).to_numpy()
        self.texts = notes["text"].astype(str).tolist()
        self.vectorizer = TfidfVectorizer(sublinear_tf=True, stop_words="english")
        self.matrix = self.vectorizer.fit_transform(self.texts).tocsr()  # rows L2-normalized
        self._rows_by_hadm: dict[int, list[int]] = {}
        for row, hadm in enumerate(self.hadm_ids):
            self._rows_by_hadm.setdefault(int(hadm), []).append(row)

    @classmethod
    def from_dir(cls, notes_dir: str | Path) -> TfidfNotesSource:
        path = table_path(Path(notes_dir), "discharge")
        if path is None:
            raise FileNotFoundError(f"discharge.csv(.gz) not found in {notes_dir}")
        return cls(pd.read_csv(path))

    def total_notes(self) -> int:
        return len(self.texts)

    def count_notes_for(self, hadm_ids: Sequence[int]) -> int:
        return sum(len(self._rows_by_hadm.get(int(h), ())) for h in set(hadm_ids))

    def _scores(self, query: str, rows: np.ndarray | None) -> np.ndarray:
        q = self.vectorizer.transform([query])
        matrix = self.matrix if rows is None else self.matrix[rows]
        return (matrix @ q.T).toarray().ravel()

    @staticmethod
    def _top(scores: np.ndarray, k: int) -> np.ndarray:
        """Positions of the k highest scores, highest first (ties by position)."""
        k = min(k, len(scores))
        if k <= 0:
            return np.array([], dtype=int)
        part = np.argpartition(-scores, k - 1)[:k] if k < len(scores) else np.arange(len(scores))
        return part[np.lexsort((part, -scores[part]))]

    def probe_search(self, query: str, k_probe: int) -> list[float]:
        scores = self._scores(query, None)
        return [float(scores[i]) for i in self._top(scores, k_probe)]

    def search(self, query: str, k: int, hadm_ids: Sequence[int] | None = None) -> SourceResult:
        if hadm_ids is None:
            rows = None
        else:
            selected = [r for h in dict.fromkeys(int(h) for h in hadm_ids) for r in self._rows_by_hadm.get(h, ())]
            rows = np.array(sorted(selected), dtype=int)
            if len(rows) == 0:
                return SourceResult(ids=[], items=[], tokens=0, latency_ms=0.0)
        scores = self._scores(query, rows)
        items = []
        for pos in self._top(scores, k):
            row = int(pos) if rows is None else int(rows[pos])
            items.append({
                "hadm_id": int(self.hadm_ids[row]),
                "note_id": self.note_ids[row],
                "text": self.texts[row],
                "score": float(scores[pos]),
            })
        return SourceResult(
            ids=sorted({i["hadm_id"] for i in items}),
            items=items,
            tokens=sum(count_tokens(i["text"]) for i in items),
            latency_ms=0.0,
        )


def build_reference_sources(tables_dir: str | Path, notes_dir: str | Path) -> Sources:
    """Relational, notes and graph reference sources over one dataset."""
    relational = DemoRelationalSource.from_tables(tables_dir)
    graph = IcdGraph.from_connection(relational.con)
    notes = TfidfNotesSource.from_dir(notes_dir)
    relational.register_notes(notes.note_ids, notes.hadm_ids.tolist())
    return Sources(relational=relational, notes=notes, graph=graph)
