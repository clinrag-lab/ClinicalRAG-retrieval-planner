"""Diagnosis categories as ICD-10 letter prefixes and ICD-9 numeric ranges.

MIMIC-IV stores ICD codes without dots (e.g. "I214", "4019"). ICD-9 ranges
compare the first three digits, so V and E supplementary codes never fall in a
numeric range; list them as ICD-9 prefixes instead.
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


# Categories used by class D questions (the graph source's categories).
CATEGORIES: dict[str, IcdCategory] = {
    c.name: c
    for c in (
        IcdCategory("circulatory", "circulatory", ("I",), ((390, 459),)),
        IcdCategory("renal_genitourinary", "renal or genitourinary", ("N",), ((580, 629),)),
        IcdCategory("respiratory", "respiratory", ("J",), ((460, 519),)),
        IcdCategory("endocrine", "endocrine", ("E",), ((240, 279),)),
    )
}

# Narrower groups used only to drive synthetic note content.
RENAL_FAILURE = IcdCategory("renal_failure", "renal failure", ("N17", "N18", "N19"), ((584, 586),))
DIABETES = IcdCategory(
    "diabetes", "diabetes", ("E08", "E09", "E10", "E11", "E13"), ((250, 250),)
)
SEPSIS = IcdCategory("sepsis", "sepsis", ("A40", "A41", "R652"), ((38, 38),), ("99591", "99592", "78552"))
FALL = IcdCategory("fall", "fall", tuple(f"W{n:02d}" for n in range(0, 20)), (), tuple(f"E88{n}" for n in range(0, 9)))


def codes_in_category(category: IcdCategory, codes: Iterable[tuple[str, int]]) -> list[tuple[str, int]]:
    """The (icd_code, icd_version) pairs that belong to a category, in input order."""
    return [(code, version) for code, version in codes if category.contains(code, version)]
