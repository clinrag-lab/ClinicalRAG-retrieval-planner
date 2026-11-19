from pathlib import Path

import duckdb
import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures"

FIXTURE_TABLES = [
    "patients",
    "admissions",
    "diagnoses_icd",
    "d_icd_diagnoses",
    "labevents",
    "d_labitems",
    "discharge",
]


@pytest.fixture
def fixture_db():
    """In-memory DuckDB connection with every fixture CSV loaded as a table."""
    con = duckdb.connect(":memory:")
    for table in FIXTURE_TABLES:
        path = (FIXTURE_DIR / f"{table}.csv").as_posix()
        con.execute(
            f"CREATE TABLE {table} AS SELECT * FROM read_csv_auto(?, header = true)",
            [path],
        )
    yield con
    con.close()
