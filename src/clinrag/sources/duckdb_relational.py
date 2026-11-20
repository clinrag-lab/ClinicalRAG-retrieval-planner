"""Relational source over a persistent DuckDB file built once from the hosp tables.

Build the file once (scripts/build_duckdb.py), then open it read-only:

    python scripts/build_duckdb.py --tables-dir data/mimic-iv-demo --out data/clinrag.duckdb
    sources = build_duckdb_sources("data/clinrag.duckdb", "data/generated")

DuckDBRelationalSource answers RelationalQuery exactly as DemoRelationalSource
does (same rows, same items), but the per-query work moves to build time:

- explicit types: icd_code VARCHAR (ICD-9 leading zeros survive), hadm_id BIGINT;
- icd_code trimmed and uppercased once, and a precomputed dx_key column
  ("version:CODE", the form of qualify()), so queries compare keys instead of
  building a string per row;
- labevents keeps only rows any lab query can return (an admission in the
  admissions table, a non-null hadm_id and valuenum), sorted by (itemid, hadm_id)
  so DuckDB's zone maps skip every row group of other items; the admissions
  join therefore happens once, at build time;
- diagnoses keeps only rows of admissions in the admissions table, sorted by dx_key;
- primary keys on the small lookup tables; only the columns the source reads.

The file is attached READ_ONLY to an in-memory connection, so several
processes can share it. Note metadata (register_notes) lives in the in-memory
database and never touches the file.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import duckdb

from clinrag.sources.base import RelationalQuery, RelationalSource, SourceResult, Sources
from clinrag.sources.icd_graph import IcdGraph
from clinrag.sources.tokens import count_tokens
from clinrag.workload.icd_categories import unqualify
from clinrag.workload.tables import find_tables_dir, table_path

SCHEMA_VERSION = "1"
DB_ALIAS = "rel"
# Distinct (icd_code, icd_version) of every diagnoses_icd row, for IcdGraph.from_connection.
GRAPH_TABLE = f"{DB_ALIAS}.dx_codes"

_SQL_OPS = {"<": "<", "<=": "<=", ">": ">", ">=": ">=", "=": "=", "!=": "<>"}

# Row order of a lab query by StructuredFilter.agg, as in DemoRelationalSource. An aggregate keeps one row.
_LAB_ORDER = {
    None: "ORDER BY l.hadm_id, l.charttime, l.labevent_id",
    "latest": "ORDER BY l.charttime DESC, l.labevent_id DESC LIMIT 1",
    "max": "ORDER BY l.valuenum DESC, l.charttime DESC, l.labevent_id DESC LIMIT 1",
}

# Column types of the CSV columns the build reads; every other column is dropped.
_CSV_TYPES = {
    "admissions": {"subject_id": "BIGINT", "hadm_id": "BIGINT"},
    "labevents": {
        "labevent_id": "BIGINT", "hadm_id": "BIGINT", "itemid": "BIGINT",
        "charttime": "TIMESTAMP", "valuenum": "DOUBLE", "valueuom": "VARCHAR",
    },
    "diagnoses_icd": {"hadm_id": "BIGINT", "seq_num": "BIGINT", "icd_code": "VARCHAR", "icd_version": "INTEGER"},
    "d_labitems": {"itemid": "BIGINT", "label": "VARCHAR"},
    "d_icd_diagnoses": {"icd_code": "VARCHAR", "icd_version": "INTEGER", "long_title": "VARCHAR"},
}


def _literal(path: Path) -> str:
    return "'" + path.as_posix().replace("'", "''") + "'"


def _read_csv(path: Path, table: str) -> str:
    """read_csv(...) of only the columns the source needs, with explicit types."""
    types = _CSV_TYPES[table]
    struct = ", ".join(f"'{col}': '{typ}'" for col, typ in types.items())
    columns = ", ".join(types)
    return f"(SELECT {columns} FROM read_csv({_literal(path)}, header = true, types = {{{struct}}}))"


def _select(table: str, relation: str) -> str:
    """The needed columns of an already loaded table, cast to the build's types."""
    columns = ", ".join(f"CAST({col} AS {typ}) AS {col}" for col, typ in _CSV_TYPES[table].items())
    return f"(SELECT {columns} FROM {relation})"


_REQUIRED = ("admissions", "diagnoses_icd", "labevents")


def build_database(tables_dir: str | Path, out: str | Path, overwrite: bool = False) -> Path:
    """Build the relational DuckDB file from the hosp tables (.csv or .csv.gz).

    Writes to a temporary file next to out and renames it on success, so a
    failed build never leaves a half-written database. Refuses to replace an
    existing file unless overwrite is True.
    """
    directory = find_tables_dir(tables_dir)
    paths = {name: table_path(directory, name) for name in _CSV_TYPES}
    for name in _REQUIRED:
        if paths[name] is None:
            raise FileNotFoundError(f"Missing table {name} in {directory}")
    relations = {name: _read_csv(path, name) if path else None for name, path in paths.items()}
    con = duckdb.connect(":memory:")
    try:
        return _write_database(con, relations, Path(out), overwrite, source=str(directory))
    finally:
        con.close()


def build_database_from_connection(
    con: duckdb.DuckDBPyConnection, out: str | Path, overwrite: bool = False, source: str = "connection"
) -> Path:
    """Build the file from hosp tables already loaded in an in-memory connection
    (as workload.tables.connect_tables loads them); used for scaled copies."""
    names = {r[0] for r in con.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_catalog = 'memory' AND table_schema = 'main'"
    ).fetchall()}
    for name in _REQUIRED:
        if name not in names:
            raise ValueError(f"connection has no table {name}")
    relations = {name: _select(name, f"memory.main.{name}") if name in names else None for name in _CSV_TYPES}
    return _write_database(con, relations, Path(out), overwrite, source)


def _write_database(
    con: duckdb.DuckDBPyConnection, relations: dict[str, str | None], out: Path, overwrite: bool, source: str
) -> Path:
    if out.exists() and not overwrite:
        raise FileExistsError(f"{out} exists; pass overwrite=True (--force) to replace it")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".building")
    for stale in (tmp, tmp.with_name(tmp.name + ".wal")):
        stale.unlink(missing_ok=True)
    con.execute(f"ATTACH {_literal(tmp)} AS _build").fetchall()
    try:
        con.execute("USE _build").fetchall()
        _create_tables(con, relations, source)
    finally:
        con.execute("USE memory").fetchall()
        con.execute("DETACH _build").fetchall()
    os.replace(tmp, out)
    return out


def _create_tables(con: duckdb.DuckDBPyConnection, relations: dict[str, str | None], source: str) -> None:
    """Create every table in the current default catalog from the source relations."""

    def run(sql: str) -> list[tuple]:
        return con.execute(sql).fetchall()

    run("CREATE TABLE admissions (hadm_id BIGINT PRIMARY KEY, subject_id BIGINT)")
    src = relations["admissions"]
    duplicates = run(f"SELECT count(*) - count(DISTINCT hadm_id) FROM {src} WHERE hadm_id IS NOT NULL")[0][0]
    if duplicates:
        # The source drops the admissions join from lab queries, which is exact only for unique hadm_ids.
        raise ValueError(f"admissions has {duplicates} duplicate hadm_id values")
    run(f"INSERT INTO admissions SELECT hadm_id, subject_id FROM {src} WHERE hadm_id IS NOT NULL ORDER BY hadm_id")

    # Only rows a lab query can return: DemoRelationalSource joins admissions and
    # requires a non-null hadm_id and valuenum on every lab query.
    src = relations["labevents"]
    run(
        f"""
        CREATE TABLE labevents AS
        SELECT labevent_id, hadm_id, itemid, charttime, valuenum, valueuom
        FROM {src} l
        WHERE l.hadm_id IS NOT NULL AND l.valuenum IS NOT NULL
          AND l.hadm_id IN (SELECT hadm_id FROM admissions)
        ORDER BY itemid, hadm_id, charttime, labevent_id
        """
    )

    # Codes normalized once; dx_key in the "version:CODE" form of qualify().
    src = relations["diagnoses_icd"]
    run(
        f"""
        CREATE TABLE diagnoses_all AS
        SELECT hadm_id, seq_num, upper(trim(icd_code)) AS icd_code, icd_version,
               CAST(icd_version AS VARCHAR) || ':' || upper(trim(icd_code)) AS dx_key
        FROM {src}
        """
    )
    # Codes for the ICD graph come from every diagnosis row, as IcdGraph.from_connection
    # reads them from the reference's diagnoses_icd table.
    run("CREATE TABLE dx_codes AS SELECT DISTINCT icd_code, icd_version FROM diagnoses_all ORDER BY ALL")
    run(
        """
        CREATE TABLE diagnoses AS
        SELECT hadm_id, seq_num, icd_code, icd_version, dx_key
        FROM diagnoses_all
        WHERE dx_key IS NOT NULL AND hadm_id IN (SELECT hadm_id FROM admissions)
        ORDER BY dx_key, hadm_id, seq_num
        """
    )
    run("DROP TABLE diagnoses_all")

    run("CREATE TABLE d_labitems (itemid BIGINT PRIMARY KEY, label VARCHAR)")
    if relations["d_labitems"] is not None:
        src = relations["d_labitems"]
        # The reference builds a dict from the rows in file order, so the last duplicate wins.
        run(
            f"""
            INSERT INTO d_labitems
            SELECT itemid, arg_max(label, rn) FROM (SELECT *, row_number() OVER () AS rn FROM {src})
            WHERE itemid IS NOT NULL GROUP BY itemid
            """
        )

    # Titles only for codes that occur in diagnoses; keyed like the reference's title dict.
    run(
        "CREATE TABLE d_icd_diagnoses (icd_code VARCHAR, icd_version INTEGER, long_title VARCHAR, "
        "PRIMARY KEY (icd_code, icd_version))"
    )
    if relations["d_icd_diagnoses"] is not None:
        src = relations["d_icd_diagnoses"]
        run(
            f"""
            INSERT INTO d_icd_diagnoses
            SELECT code, icd_version, arg_max(long_title, rn)
            FROM (SELECT upper(trim(icd_code)) AS code, icd_version, long_title,
                         row_number() OVER () AS rn FROM {src}) t
            WHERE code IS NOT NULL AND icd_version IS NOT NULL
              AND EXISTS (SELECT 1 FROM diagnoses d WHERE d.icd_code = t.code AND d.icd_version = t.icd_version)
            GROUP BY code, icd_version
            """
        )

    run("CREATE TABLE build_info (key VARCHAR PRIMARY KEY, value VARCHAR)")
    con.execute(
        "INSERT INTO build_info VALUES ('schema_version', ?), ('source', ?)",
        [SCHEMA_VERSION, source],
    ).fetchall()


class DuckDBRelationalSource(RelationalSource):
    """RelationalQuery over a DuckDB file from build_database().

    Same semantics and items as DemoRelationalSource:
    - lab rows: one row per lab event of an admission in the admissions table
      with a non-null valuenum, matching itemid, the comparison if any, and
      lab.hadm_id when set;
    - icd_codes: version-qualified ("10:I509"), matched on (icd_version, icd_code)
      through the stored dx_key;
    - lab.agg: of those rows, exactly one (none if nothing matches): "latest"
      the row with the greatest charttime, "max" the row with the highest
      valuenum; ties go to the later charttime, then the higher labevent_id.
      count() is then at most 1;
    - lab and codes: lab rows of admissions with at least one of the codes;
    - codes only: one row per matching diagnosis row.

    Lab items carry hadm_id, text, itemid, label, valuenum, unit, charttime;
    diagnosis items carry hadm_id, text, icd_code, icd_version.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"no DuckDB file at {self.db_path}; build it with scripts/build_duckdb.py")
        self.con = duckdb.connect(":memory:")
        self.con.execute(f"ATTACH {_literal(self.db_path)} AS {DB_ALIAS} (READ_ONLY)").fetchall()
        info = dict(self.con.execute(f"SELECT key, value FROM {DB_ALIAS}.build_info").fetchall())
        if info.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"{self.db_path} has schema version {info.get('schema_version')!r}, expected "
                f"{SCHEMA_VERSION!r}; rebuild it with scripts/build_duckdb.py --force"
            )
        self._lab_labels = {
            k: str(v) for k, v in self.con.execute(f"SELECT itemid, label FROM {DB_ALIAS}.d_labitems").fetchall()
        }
        self._dx_titles = {
            (c, int(v)): str(t)
            for c, v, t in self.con.execute(
                f"SELECT icd_code, icd_version, long_title FROM {DB_ALIAS}.d_icd_diagnoses"
            ).fetchall()
        }
        self._notes_registered = False

    def icd_graph(self) -> IcdGraph:
        """The ICD graph over the codes present in the data, as the reference builds it."""
        return IcdGraph.from_connection(self.con, table=GRAPH_TABLE)

    @staticmethod
    def _keys(query: RelationalQuery) -> list[str]:
        """dx_key values for the query's codes; raises ValueError on an unqualified code."""
        keys = []
        for qualified in query.icd_codes:
            code, version = unqualify(qualified)
            keys.append(f"{version}:{code}")
        return keys

    def _from_where(self, query: RelationalQuery) -> tuple[str, list[Any], bool]:
        """(FROM ... WHERE ..., params, is_lab)."""
        if query.lab is None:
            sql = f"FROM {DB_ALIAS}.diagnoses d WHERE d.dx_key IN (SELECT unnest(?::VARCHAR[]))"
            return sql, [self._keys(query)], False
        lab = query.lab
        sql = f"FROM {DB_ALIAS}.labevents l WHERE l.itemid = ?"
        params: list[Any] = [lab.itemid]
        if lab.hadm_id is not None:
            sql += " AND l.hadm_id = ?"
            params.append(lab.hadm_id)
        if lab.op is not None:
            sql += f" AND l.valuenum {_SQL_OPS[lab.op]} ?"
            params.append(lab.threshold)
        if query.icd_codes:
            sql += (
                f" AND l.hadm_id IN (SELECT d.hadm_id FROM {DB_ALIAS}.diagnoses d "
                "WHERE d.dx_key IN (SELECT unnest(?::VARCHAR[])))"
            )
            params.append(self._keys(query))
        return sql, params, True

    def _rows(self, sql: str, params: Sequence[Any]) -> list[tuple]:
        return self.con.execute(sql, list(params)).fetchall()

    def count(self, query: RelationalQuery) -> int:
        sql, params, _ = self._from_where(query)
        n = int(self._rows(f"SELECT count(*) {sql}", params)[0][0])
        return min(n, 1) if query.lab is not None and query.lab.agg is not None else n

    def matching_ids(self, query: RelationalQuery) -> list[int]:
        sql, params, is_lab = self._from_where(query)
        col = "l.hadm_id" if is_lab else "d.hadm_id"
        return [int(r[0]) for r in self._rows(f"SELECT DISTINCT {col} {sql} ORDER BY 1", params)]

    # ---- note counts in SQL (optional probes) ----

    def register_notes(self, note_ids: Sequence[str], hadm_ids: Sequence[int]) -> None:
        """Load the note index's metadata (note_id, hadm_id; never text) into the in-memory database.

        note_meta has one row per indexed chunk; note_counts has chunks per admission.
        Calling it again replaces the previous metadata.
        """
        note_ids = [str(n) for n in note_ids]
        hadm_ids = [int(h) for h in hadm_ids]
        if len(note_ids) != len(hadm_ids):
            raise ValueError("note_ids and hadm_ids must have the same length")
        self.con.execute("CREATE OR REPLACE TABLE note_meta (note_id VARCHAR, hadm_id BIGINT)").fetchall()
        if note_ids:
            self.con.execute(
                "INSERT INTO note_meta SELECT unnest(?::VARCHAR[]), unnest(?::BIGINT[])", [note_ids, hadm_ids]
            ).fetchall()
        self.con.execute(
            "CREATE OR REPLACE TABLE note_counts (hadm_id BIGINT PRIMARY KEY, n BIGINT NOT NULL)"
        ).fetchall()
        self.con.execute(
            "INSERT INTO note_counts SELECT hadm_id, count(*) FROM note_meta GROUP BY hadm_id"
        ).fetchall()
        self._notes_registered = True

    def _require_notes(self) -> None:
        if not self._notes_registered:
            raise NotImplementedError("no note metadata registered; call register_notes first")

    def count_notes(self, query: RelationalQuery) -> int:
        self._require_notes()
        sql, params, is_lab = self._from_where(query)
        col = "l.hadm_id" if is_lab else "d.hadm_id"
        rows = self._rows(
            f"SELECT coalesce(sum(c.n), 0) FROM note_counts c WHERE c.hadm_id IN (SELECT {col} {sql})", params
        )
        return int(rows[0][0])

    def count_matches(self, query: RelationalQuery) -> tuple[int, int, int]:
        self._require_notes()
        sql, params, is_lab = self._from_where(query)
        col = "l.hadm_id" if is_lab else "d.hadm_id"
        rows = self._rows(
            f"""
            WITH m AS MATERIALIZED (SELECT {col} AS h {sql}),
                 a AS (SELECT DISTINCT h FROM m)
            SELECT (SELECT count(*) FROM m),
                   (SELECT count(*) FROM a),
                   (SELECT coalesce(sum(c.n), 0) FROM note_counts c JOIN a ON c.hadm_id = a.h)
            """,
            params,
        )
        return int(rows[0][0]), int(rows[0][1]), int(rows[0][2])

    def run(self, query: RelationalQuery) -> SourceResult:
        sql, params, is_lab = self._from_where(query)
        items: list[dict[str, Any]] = []
        if is_lab:
            assert query.lab is not None
            rows = self._rows(
                f"SELECT l.hadm_id, l.itemid, l.valuenum, l.valueuom, l.charttime {sql} {_LAB_ORDER[query.lab.agg]}",
                params,
            )
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
            rows = self._rows(
                f"SELECT d.hadm_id, d.icd_code, d.icd_version {sql} ORDER BY d.hadm_id, d.seq_num, d.dx_key",
                params,
            )
            for hadm, code, version in rows:
                title = self._dx_titles.get((code, int(version)), "")
                items.append({
                    "hadm_id": int(hadm),
                    "icd_code": code,
                    "icd_version": int(version),
                    "text": f"Admission {hadm}: diagnosis ICD-{version} {code} {title}".rstrip(),
                })
        ids = sorted({item["hadm_id"] for item in items})
        return SourceResult(ids=ids, items=items, tokens=sum(count_tokens(i["text"]) for i in items), latency_ms=0.0)

    def close(self) -> None:
        self.con.close()


def build_duckdb_sources(db_path: str | Path, notes_dir: str | Path) -> Sources:
    """DuckDBRelationalSource + the reference TF-IDF notes + IcdGraph, as build_reference_sources."""
    from clinrag.sources.reference import TfidfNotesSource

    relational = DuckDBRelationalSource(db_path)
    graph = relational.icd_graph()
    notes = TfidfNotesSource.from_dir(notes_dir)
    relational.register_notes(notes.note_ids, notes.hadm_ids.tolist())
    return Sources(relational=relational, notes=notes, graph=graph)
