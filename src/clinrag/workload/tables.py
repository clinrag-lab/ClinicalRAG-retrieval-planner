"""Load MIMIC-IV hosp tables (.csv or .csv.gz) into an in-memory DuckDB."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import duckdb

# Columns that must stay text even when every value looks numeric (ICD-9 codes).
_TYPES = {
    "diagnoses_icd": {"icd_code": "VARCHAR"},
    "d_icd_diagnoses": {"icd_code": "VARCHAR"},
}


def table_path(tables_dir: Path, name: str) -> Path | None:
    for suffix in (".csv.gz", ".csv"):
        path = tables_dir / f"{name}{suffix}"
        if path.exists():
            return path
    return None


def find_tables_dir(path: str | Path) -> Path:
    """Return path if it holds admissions.csv(.gz), else the hosp/ folder below it that does."""
    root = Path(path)
    if table_path(root, "admissions") is not None:
        return root
    for candidate in sorted(root.rglob("hosp")):
        if candidate.is_dir() and table_path(candidate, "admissions") is not None:
            return candidate
    raise FileNotFoundError(f"No admissions table found in {root} or any hosp/ folder below it")


def load_csv(con: duckdb.DuckDBPyConnection, table: str, path: Path, types: dict[str, str] | None = None) -> None:
    """Create table from a CSV file, reading the whole file to infer types."""
    literal = path.as_posix().replace("'", "''")
    options = "header = true, sample_size = -1"
    if types:
        struct = ", ".join(f"'{col}': '{typ}'" for col, typ in types.items())
        options += f", types = {{{struct}}}"
    con.execute(f"CREATE TABLE {table} AS SELECT * FROM read_csv('{literal}', {options})")


def connect_tables(
    tables_dir: str | Path,
    required: Iterable[str],
    optional: Iterable[str] = (),
    con: duckdb.DuckDBPyConnection | None = None,
) -> duckdb.DuckDBPyConnection:
    """Load the named tables into a DuckDB connection. Missing optional tables are skipped."""
    directory = find_tables_dir(tables_dir)
    con = con or duckdb.connect(":memory:")
    for name, needed in [(n, True) for n in required] + [(n, False) for n in optional]:
        path = table_path(directory, name)
        if path is None:
            if needed:
                raise FileNotFoundError(f"Missing table {name} in {directory}")
            continue
        load_csv(con, name, path, _TYPES.get(name))
    return con


def lab_summary(con: duckdb.DuckDBPyConnection, itemids: Iterable[int]) -> list[tuple[int, int, float, float]]:
    """(hadm_id, itemid, max_value, latest_value) per admission and lab item.

    Only rows with an hadm_id and a numeric value count. latest_value is the
    value with the greatest charttime (ties broken by labevent_id).
    """
    ids = sorted(set(itemids))
    if not ids:
        return []
    placeholders = ", ".join("?" for _ in ids)
    return con.execute(
        f"""
        WITH rows AS (
            SELECT hadm_id, itemid, valuenum,
                   row_number() OVER (
                       PARTITION BY hadm_id, itemid ORDER BY charttime DESC, labevent_id DESC
                   ) AS recency
            FROM labevents
            WHERE hadm_id IS NOT NULL AND valuenum IS NOT NULL AND itemid IN ({placeholders})
        )
        SELECT hadm_id, itemid, max(valuenum) AS max_value,
               max(valuenum) FILTER (WHERE recency = 1) AS latest_value
        FROM rows
        GROUP BY hadm_id, itemid
        ORDER BY hadm_id, itemid
        """,
        ids,
    ).fetchall()


def has_table(con: duckdb.DuckDBPyConnection, name: str) -> bool:
    rows = con.execute("SELECT 1 FROM information_schema.tables WHERE table_name = ?", [name]).fetchall()
    return bool(rows)
