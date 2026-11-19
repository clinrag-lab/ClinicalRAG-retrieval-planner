import pytest

EXPECTED_ROW_COUNTS = {
    "patients": 5,
    "admissions": 8,
    "diagnoses_icd": 19,
    "d_icd_diagnoses": 11,
    "labevents": 20,
    "d_labitems": 3,
    "discharge": 8,
}


@pytest.mark.parametrize("table,expected", EXPECTED_ROW_COUNTS.items())
def test_row_counts(fixture_db, table, expected):
    (count,) = fixture_db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    assert count == expected


def test_discharge_hadm_ids_exist_in_admissions(fixture_db):
    orphans = fixture_db.execute(
        """
        SELECT d.hadm_id
        FROM discharge d
        LEFT JOIN admissions a ON d.hadm_id = a.hadm_id
        WHERE a.hadm_id IS NULL
        """
    ).fetchall()
    assert orphans == []
