from pathlib import Path

import pytest

from clinrag.sources.icd_graph import IcdGraph
from clinrag.workload.icd_categories import BLOCKS, CATEGORIES, CHAPTERS, children, qualify, unqualify
from clinrag.workload.tables import connect_tables

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize(
    "block,code,version,expected",
    [
        ("heart_failure", "I509", 10, True),
        ("heart_failure", "4280", 9, True),
        ("ischemic_heart_disease", "I214", 10, True),
        ("ischemic_heart_disease", "I2510", 10, True),
        ("ischemic_heart_disease", "41401", 9, True),
        ("ischemic_heart_disease", "I10", 10, False),
        ("atrial_fibrillation", "I480", 10, True),
        ("atrial_fibrillation", "42731", 9, True),
        ("atrial_fibrillation", "42789", 9, False),
        ("chronic_kidney_disease", "N186", 10, True),
        ("chronic_kidney_disease", "5856", 9, True),
        ("acute_kidney_injury", "N179", 10, True),
        ("acute_kidney_injury", "5849", 9, True),
        ("acute_kidney_injury", "N186", 10, False),
        ("pneumonia", "J189", 10, True),
        ("pneumonia", "J1282", 10, True),
        ("pneumonia", "486", 9, True),
        ("pneumonia", "J449", 10, False),
        ("copd", "J449", 10, True),
        ("copd", "4912", 9, True),
        ("copd", "496", 9, True),
        ("copd", "4930", 9, False),  # asthma
        ("diabetes", "E119", 10, True),
        ("diabetes", "E0800", 10, True),
        ("diabetes", "25000", 9, True),
        ("diabetes", "E039", 10, False),
        ("hypothyroidism", "E039", 10, True),
        ("hypothyroidism", "2449", 9, True),
    ],
)
def test_block_membership(block, code, version, expected):
    assert CATEGORIES[block].contains(code, version) is expected


def test_hierarchy_is_explicit():
    chapter_names = {c.name for c in CHAPTERS}
    assert all(c.level == "chapter" for c in CHAPTERS)
    assert all(b.level == "block" and b.parent in chapter_names for b in BLOCKS)
    assert set(CATEGORIES) == chapter_names | {b.name for b in BLOCKS}
    assert children("circulatory") == ["heart_failure", "ischemic_heart_disease", "atrial_fibrillation"]
    assert children("endocrine") == ["diabetes", "hypothyroidism"]


# One representative code per block and version; each must also be in its chapter.
BLOCK_EXAMPLES = [
    ("heart_failure", "I509", 10), ("heart_failure", "4280", 9),
    ("ischemic_heart_disease", "I214", 10), ("ischemic_heart_disease", "4109", 9),
    ("atrial_fibrillation", "I480", 10), ("atrial_fibrillation", "42731", 9),
    ("chronic_kidney_disease", "N183", 10), ("chronic_kidney_disease", "5853", 9),
    ("acute_kidney_injury", "N170", 10), ("acute_kidney_injury", "5845", 9),
    ("pneumonia", "J13", 10), ("pneumonia", "4829", 9),
    ("copd", "J441", 10), ("copd", "4920", 9),
    ("diabetes", "E1165", 10), ("diabetes", "25060", 9),
    ("hypothyroidism", "E038", 10), ("hypothyroidism", "2448", 9),
]


@pytest.mark.parametrize("block,code,version", BLOCK_EXAMPLES)
def test_block_codes_are_inside_their_chapter(block, code, version):
    category = CATEGORIES[block]
    assert category.contains(code, version)
    assert CATEGORIES[category.parent].contains(code, version)


def test_expand_works_for_chapters_and_blocks():
    graph = IcdGraph([("I509", 10), ("I214", 10), ("I10", 10), ("4280", 9), ("N179", 10), ("4019", 9)])
    # Sorted by (code, version): ICD-9 numeric codes sort before ICD-10 letter codes.
    assert graph.expand("circulatory") == ["9:4019", "9:4280", "10:I10", "10:I214", "10:I509"]
    assert graph.expand("heart_failure") == ["9:4280", "10:I509"]
    assert graph.descendant_count("heart_failure") == 2
    assert set(graph.expand("heart_failure")) <= set(graph.expand("circulatory"))
    assert graph.parent("heart_failure") == "circulatory"
    assert graph.parent("circulatory") is None
    assert graph.children("renal_genitourinary") == ["chronic_kidney_disease", "acute_kidney_injury"]
    assert graph.children("copd") == []


def test_expand_keeps_versions_apart():
    # "E8881" is metabolic syndrome in ICD-10 (endocrine) but a fall external-cause code in ICD-9.
    graph = IcdGraph([("E8881", 10), ("E8881", 9)])
    assert graph.expand("endocrine") == ["10:E8881"]


def test_expand_only_returns_codes_present_in_data():
    con = connect_tables(FIXTURES, required=("diagnoses_icd",))
    graph = IcdGraph.from_connection(con)
    assert graph.expand("renal_genitourinary") == ["10:N179", "10:N185", "10:N186"]
    assert graph.expand("chronic_kidney_disease") == ["10:N185", "10:N186"]
    assert graph.expand("hypothyroidism") == []


def test_unknown_category_raises():
    with pytest.raises(KeyError, match="Unknown ICD category"):
        IcdGraph([]).expand("dermatology")


def test_qualify_round_trip():
    assert qualify("i214 ", 10) == "10:I214"
    assert unqualify("9:4019") == ("4019", 9)
    with pytest.raises(ValueError):
        unqualify("I214")
