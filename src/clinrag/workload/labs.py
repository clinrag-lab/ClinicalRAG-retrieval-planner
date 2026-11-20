"""Lab items used by the note generator and the question templates."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Lab:
    name: str
    itemid: int  # MIMIC-IV d_labitems itemid (blood chemistry / hematology)
    unit: str
    above_threshold: float  # threshold for "was X ever above T" questions
    decimals: int = 1  # rounding for thresholds and note text


LABS: dict[str, Lab] = {
    lab.name: lab
    for lab in (
        Lab("creatinine", 50912, "mg/dL", 2.0),
        Lab("potassium", 50971, "mEq/L", 5.0),
        Lab("hemoglobin", 51222, "g/dL", 12.0),
    )
}
