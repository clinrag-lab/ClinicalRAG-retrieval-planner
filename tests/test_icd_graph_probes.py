"""IcdGraph probes: precomputed expansions and O(1) descendant counts."""

from dataclasses import dataclass
from pathlib import Path

import pytest

from clinrag.sources.icd_graph import IcdGraph
from clinrag.workload.icd_categories import CATEGORIES, IcdCategory, qualify
from clinrag.workload.tables import connect_tables

FIXTURES = Path(__file__).parent / "fixtures"

# A mix of versions, a leading-zero ICD-9 code, V and E codes, and unnormalized input.
CODES = [
    ("I509", 10), ("4280", 9), ("I214", 10), ("41401", 9), ("I480", 10), ("42731", 9), ("N186", 10),
    ("5856", 9), ("N179", 10), ("5849", 9), ("J189", 10), ("486", 9), ("J449", 10), ("496", 9),
    ("E119", 10), ("25000", 9), ("E039", 10), ("2449", 9), ("0389", 9), ("V4581", 9), ("E8881", 9),
    (" i10 ", 10), ("I10", 10), ("K219", 10),
]


def brute_force(codes, category):
    normalized = sorted({(c.strip().upper(), int(v)) for c, v in codes})
    return [qualify(c, v) for c, v in normalized if category.contains(c, v)]


@pytest.fixture(scope="module")
def graph():
    return IcdGraph(CODES)


@pytest.mark.parametrize("name", sorted(CATEGORIES))
def test_counts_and_expansions_match_brute_force(graph, name):
    expected = brute_force(CODES, CATEGORIES[name])
    assert graph.expand(name) == expected
    assert graph.descendant_count(name) == len(expected) == len(graph.expand_pairs(name))


def test_expected_counts(graph):
    counts = {name: graph.descendant_count(name) for name in CATEGORIES}
    assert counts == {
        "circulatory": 7, "renal_genitourinary": 4, "respiratory": 4, "endocrine": 4,
        "heart_failure": 2, "ischemic_heart_disease": 2, "atrial_fibrillation": 2,
        "chronic_kidney_disease": 2, "acute_kidney_injury": 2, "pneumonia": 2, "copd": 2,
        "diabetes": 2, "hypothyroidism": 2,
    }
    # A chapter counts every code of its blocks, plus codes in no block.
    for chapter in ("circulatory", "renal_genitourinary", "respiratory", "endocrine"):
        blocks = [b for b, c in CATEGORIES.items() if c.parent == chapter]
        assert counts[chapter] >= sum(counts[b] for b in blocks)


def test_counts_on_fixture_data():
    con = connect_tables(FIXTURES, required=("diagnoses_icd",))
    graph = IcdGraph.from_connection(con)
    assert graph.descendant_count("renal_genitourinary") == 3  # N179, N185, N186
    assert graph.descendant_count("heart_failure") == 1
    assert graph.descendant_count("copd") == 0
    assert graph.expand("copd") == []


@dataclass(frozen=True)
class CountingCategory(IcdCategory):
    calls: list = None

    def contains(self, icd_code, icd_version):
        self.calls.append(1)
        return super().contains(icd_code, icd_version)


def test_descendant_count_does_no_work_after_construction():
    calls = []
    categories = {"hf": CountingCategory("hf", "heart failure", ("I50",), ((428, 428),), calls=calls)}
    graph = IcdGraph(CODES, categories)
    built = len(calls)
    assert built == len({(c.strip().upper(), v) for c, v in CODES})  # one membership test per code
    for _ in range(100):
        assert graph.descendant_count("hf") == 2
        graph.expand("hf")
    assert len(calls) == built


def test_expand_returns_a_copy(graph):
    codes = graph.expand("heart_failure")
    codes.append("10:X")
    graph.expand_pairs("heart_failure").clear()
    assert graph.expand("heart_failure") == ["9:4280", "10:I509"]
    assert graph.descendant_count("heart_failure") == 2


def test_unknown_category_raises_for_every_probe(graph):
    for probe in (graph.expand, graph.expand_pairs, graph.descendant_count):
        with pytest.raises(KeyError, match="Unknown ICD category"):
            probe("not_a_category")
