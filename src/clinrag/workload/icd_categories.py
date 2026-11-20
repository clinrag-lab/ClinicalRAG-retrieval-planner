"""Diagnosis categories as ICD-10 prefixes and ICD-9 numeric ranges, in a two-level hierarchy.

chapter (e.g. circulatory) -> block (e.g. heart_failure) -> codes present in the data

MIMIC-IV stores ICD codes without dots (e.g. "I214", "4019"). ICD-9 ranges
compare the first three digits, so V and E supplementary codes never fall in a
numeric range; list them as ICD-9 prefixes instead. The same code string can
mean different things in ICD-9 and ICD-10 (e.g. "E8881"), so codes passed
between sources are version-qualified with qualify().
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class IcdCategory:
    name: str
    label: str
    icd10_prefixes: tuple[str, ...] = ()
    icd9_ranges: tuple[tuple[int, int], ...] = ()  # inclusive, on the first 3 digits
    icd9_prefixes: tuple[str, ...] = ()
    parent: str | None = None  # chapter name for blocks, None for chapters

    @property
    def level(self) -> str:
        return "chapter" if self.parent is None else "block"

    def contains(self, icd_code: str, icd_version: int) -> bool:
        code = icd_code.strip().upper()
        if int(icd_version) == 10:
            return code.startswith(self.icd10_prefixes)
        if code.startswith(self.icd9_prefixes):  # False for an empty tuple
            return True
        head = code[:3]
        if not head.isdigit():
            return False
        number = int(head)
        return any(lo <= number <= hi for lo, hi in self.icd9_ranges)


def _prefix_range(letter: str, lo: int, hi: int) -> tuple[str, ...]:
    """ICD-10 prefixes letter+lo .. letter+hi, e.g. ("I20", ..., "I25")."""
    return tuple(f"{letter}{n:02d}" for n in range(lo, hi + 1))


CHAPTERS: tuple[IcdCategory, ...] = (
    IcdCategory("circulatory", "circulatory", ("I",), ((390, 459),)),
    IcdCategory("renal_genitourinary", "renal or genitourinary", ("N",), ((580, 629),)),
    IcdCategory("respiratory", "respiratory", ("J",), ((460, 519),)),
    IcdCategory("endocrine", "endocrine", ("E",), ((240, 279),)),
)

BLOCKS: tuple[IcdCategory, ...] = (
    IcdCategory("heart_failure", "heart failure", ("I50",), ((428, 428),), parent="circulatory"),
    IcdCategory(
        "ischemic_heart_disease", "ischemic heart disease", _prefix_range("I", 20, 25), ((410, 414),),
        parent="circulatory",
    ),
    IcdCategory("atrial_fibrillation", "atrial fibrillation", ("I48",), (), ("4273",), parent="circulatory"),
    IcdCategory("chronic_kidney_disease", "chronic kidney disease", ("N18",), ((585, 585),), parent="renal_genitourinary"),
    IcdCategory("acute_kidney_injury", "acute kidney injury", ("N17",), ((584, 584),), parent="renal_genitourinary"),
    IcdCategory("pneumonia", "pneumonia", _prefix_range("J", 12, 18), ((480, 486),), parent="respiratory"),
    # Chronic bronchitis and emphysema in both versions: ICD-9 491-492 map to ICD-10 J41-J43.
    IcdCategory("copd", "COPD", _prefix_range("J", 41, 44), ((491, 492), (496, 496)), parent="respiratory"),
    # Secondary diabetes in both versions: ICD-10 E08-E09 map to ICD-9 249.
    IcdCategory("diabetes", "diabetes", _prefix_range("E", 8, 13), ((249, 250),), parent="endocrine"),
    # Congenital, iodine-deficiency and postprocedural hypothyroidism in both versions:
    # ICD-10 E03.0-E03.1 map to ICD-9 243, and ICD-9 244.0-244.2 to ICD-10 E02 and E89.0.
    IcdCategory("hypothyroidism", "hypothyroidism", ("E02", "E03", "E890"), ((243, 244),), parent="endocrine"),
)

# Every category the graph can expand: chapters first, then blocks.
CATEGORIES: dict[str, IcdCategory] = {c.name: c for c in CHAPTERS + BLOCKS}


def children(chapter: str) -> list[str]:
    """Block names under a chapter."""
    return [b.name for b in BLOCKS if b.parent == chapter]


# Narrower groups used only to drive synthetic note content.
RENAL_FAILURE = IcdCategory("renal_failure", "renal failure", ("N17", "N18", "N19"), ((584, 586),))
DIABETES = CATEGORIES["diabetes"]
SEPSIS = IcdCategory("sepsis", "sepsis", ("A40", "A41", "R652"), ((38, 38),), ("99591", "99592", "78552"))
FALL = IcdCategory("fall", "fall", tuple(f"W{n:02d}" for n in range(0, 20)), (), tuple(f"E88{n}" for n in range(0, 9)))


def qualify(icd_code: str, icd_version: int) -> str:
    """Version-qualified code string, e.g. ("I214", 10) -> "10:I214"."""
    return f"{int(icd_version)}:{icd_code.strip().upper()}"


def unqualify(qualified: str) -> tuple[str, int]:
    """Inverse of qualify: "10:I214" -> ("I214", 10)."""
    version, _, code = qualified.partition(":")
    if not code or version not in ("9", "10"):
        raise ValueError(f"Not a version-qualified ICD code: {qualified!r}")
    return code, int(version)


def codes_in_category(category: IcdCategory, codes: Iterable[tuple[str, int]]) -> list[tuple[str, int]]:
    """The (icd_code, icd_version) pairs that belong to a category, in input order."""
    return [(code, version) for code, version in codes if category.contains(code, version)]
