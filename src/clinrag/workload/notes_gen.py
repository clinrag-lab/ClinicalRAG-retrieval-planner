"""Generate synthetic discharge notes from structured MIMIC-IV hosp tables.

The open MIMIC-IV demo has no notes, so each admission gets one templated note
built from its own diagnoses, labs, prescriptions and disposition. Which
concepts a note mentions depends on that admission's data, so note content
correlates with structured filters. note_concepts records exactly which
concepts each note mentions; negated mentions ("no chest pain") are recorded
separately in note_negations and are never tagged.

Usage:
    python -m clinrag.workload.notes_gen --tables-dir data/mimic-iv-demo --out-dir data/generated --seed 0
    python -m clinrag.workload.notes_gen --tables-dir data/mimic-iv-demo --out-dir data/generated --scale 10

--scale S writes S notes per admission (note_seq 1..S) to <out-dir>/scale_S/.
Note 1 is the scale-1 note; notes 2..S have independently seeded phrasing but
mention exactly the same tagged and negated concepts, so every per-admission
gold answer is unchanged. Scaled notes exist to grow the index for the
filter-first vs search-first experiments.
"""

from __future__ import annotations

import argparse
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd

from clinrag.workload.icd_categories import CATEGORIES, DIABETES, FALL, RENAL_FAILURE, SEPSIS
from clinrag.workload.labs import LABS
from clinrag.workload.tables import connect_tables, has_table, lab_summary

# Chance that an admission's note mentions each concept. "base" applies to every
# admission; each other key is a trigger (see admission_triggers) that adds its
# own chance when present. They combine as a noisy-OR:
#     p = 1 - (1 - base) * prod(1 - p_trigger for each trigger the admission has)
# "negated" is the chance that a note NOT tagged with the concept gets a negated
# mention of it, as a retrieval distractor.
CONCEPT_PROBS: dict[str, dict[str, float]] = {
    "dialysis": {"base": 0.02, "renal_failure_dx": 0.50, "creatinine_above_2": 0.35, "negated": 0.10},
    "chest pain": {"base": 0.04, "circulatory_dx": 0.35, "negated": 0.15},
    "shortness of breath": {"base": 0.06, "respiratory_dx": 0.40, "circulatory_dx": 0.15, "negated": 0.12},
    "anticoagulation": {"base": 0.04, "circulatory_dx": 0.25, "negated": 0.08},
    "insulin": {"base": 0.03, "diabetes_dx": 0.60, "negated": 0.08},
    "sepsis": {"base": 0.02, "sepsis_dx": 0.85, "negated": 0.08},
    "fall": {"base": 0.03, "fall_dx": 0.80, "age_75_plus": 0.10, "negated": 0.06},
    "intubation": {"base": 0.02, "respiratory_dx": 0.15, "sepsis_dx": 0.25, "negated": 0.06},
}
CONCEPTS: tuple[str, ...] = tuple(CONCEPT_PROBS)

# How a mention of each concept is recognised in text (positive or negated).
CONCEPT_PATTERNS: dict[str, re.Pattern[str]] = {
    "dialysis": re.compile(r"dialysis", re.IGNORECASE),
    "chest pain": re.compile(r"chest pain", re.IGNORECASE),
    "shortness of breath": re.compile(r"shortness of breath", re.IGNORECASE),
    "anticoagulation": re.compile(r"anticoagulation", re.IGNORECASE),
    "insulin": re.compile(r"\binsulin\b", re.IGNORECASE),
    "sepsis": re.compile(r"\bsepsis\b", re.IGNORECASE),
    "fall": re.compile(r"\bfalls?\b", re.IGNORECASE),
    "intubation": re.compile(r"intubation", re.IGNORECASE),
}

POSITIVE_PHRASES: dict[str, tuple[str, ...]] = {
    "dialysis": (
        "Hemodialysis was initiated for volume overload and uremia.",
        "The patient received dialysis during the admission and tolerated sessions well.",
        "Nephrology was consulted and dialysis was continued three times weekly.",
    ),
    "chest pain": (
        "The patient presented with substernal chest pain radiating to the left arm.",
        "Intermittent chest pain at rest was reported on arrival.",
        "The chest pain resolved after sublingual nitroglycerin.",
    ),
    "shortness of breath": (
        "Progressive shortness of breath was noted over several days before admission.",
        "Shortness of breath improved with diuresis.",
        "The patient described shortness of breath with minimal exertion.",
    ),
    "anticoagulation": (
        "Therapeutic anticoagulation was started with heparin and bridged to warfarin.",
        "Anticoagulation was continued at discharge with close follow-up.",
        "Systemic anticoagulation was resumed once bleeding risk was acceptable.",
    ),
    "insulin": (
        "Blood glucose was managed with an insulin sliding scale.",
        "Basal insulin was titrated during the stay.",
        "The patient was discharged on insulin with home glucose monitoring.",
    ),
    "sepsis": (
        "The course was complicated by sepsis treated with broad-spectrum antibiotics.",
        "Sepsis was suspected on arrival and blood cultures were drawn.",
        "Hemodynamics improved as the sepsis was treated.",
    ),
    "fall": (
        "The patient was admitted after a fall at home.",
        "A fall on the ward was witnessed without injury.",
        "Physical therapy addressed gait instability after a recent fall.",
    ),
    "intubation": (
        "Intubation was required for hypoxemic respiratory failure.",
        "The patient underwent intubation in the emergency department.",
        "Extubation was achieved two days after intubation.",
    ),
}

NEGATED_PHRASES: dict[str, tuple[str, ...]] = {
    "dialysis": ("Dialysis was not required.", "There was no indication for dialysis."),
    "chest pain": ("The patient denied chest pain.", "No chest pain was reported."),
    "shortness of breath": ("There was no shortness of breath at rest.", "The patient denied shortness of breath."),
    "anticoagulation": ("Anticoagulation was not started.", "No anticoagulation was indicated."),
    "insulin": ("Insulin was not needed during the stay.", "No insulin was required."),
    "sepsis": ("There was no evidence of sepsis.", "Sepsis was ruled out."),
    "fall": ("No falls occurred during the admission.", "The patient denied any recent fall."),
    "intubation": ("Intubation was not required.", "No intubation was needed."),
}

# Neutral sentences used to bring short notes up to MIN_WORDS. No concept terms.
FILLER_SENTENCES: tuple[str, ...] = (
    "Physical therapy evaluated the patient and recommended continued mobilization.",
    "Medications were reconciled with the patient prior to discharge.",
    "The patient tolerated a regular diet before discharge.",
    "Vital signs remained stable over the final twenty-four hours.",
    "Pain was controlled with oral analgesics.",
    "The patient was ambulating independently at the time of discharge.",
    "Follow-up was arranged with the primary care physician within one week.",
    "Return precautions were reviewed and the patient verbalized understanding.",
    "Case management assisted with discharge planning.",
    "Telemetry monitoring showed no significant events.",
    "Nutrition was consulted and dietary recommendations were provided.",
    "Home medications were resumed unless otherwise noted below.",
    "The patient and family were updated on the plan of care daily.",
    "Pending results at discharge will be followed up by the primary team.",
    "Skin was intact on examination at discharge.",
    "The patient was alert and oriented on the day of discharge.",
)

MIN_WORDS = 150
MAX_WORDS = 400
FILLER_TARGET_MAX = 260  # each note is padded to a seeded target in [MIN_WORDS, FILLER_TARGET_MAX]
MAX_DIAGNOSES = 6
MAX_MEDICATIONS = 8

_SECONDARY_LEADS = (
    "Additional problems addressed during the stay included",
    "Other active diagnoses were",
    "The stay also addressed",
)
_LAB_VERBS = ("peaked at", "reached a maximum of", "was as high as")


@dataclass(frozen=True)
class GeneratedNotes:
    discharge: pd.DataFrame  # note_id, subject_id, hadm_id, note_type, note_seq, charttime, text
    concepts: pd.DataFrame  # note_id, hadm_id, concept (positive mentions only)
    negations: pd.DataFrame  # note_id, hadm_id, concept (negated mentions, never tagged)


def mentioned_concepts(text: str) -> set[str]:
    """Concepts whose term appears anywhere in text, positive or negated."""
    return {c for c, pattern in CONCEPT_PATTERNS.items() if pattern.search(text)}


def concept_probability(concept: str, triggers: set[str]) -> float:
    probs = CONCEPT_PROBS[concept]
    keep = 1.0 - probs["base"]
    for trigger, p in probs.items():
        if trigger not in ("base", "negated") and trigger in triggers:
            keep *= 1.0 - p
    return 1.0 - keep


def admission_triggers(codes: list[tuple[str, int]], lab_max: dict[str, float], age: int | None) -> set[str]:
    """Data-driven triggers for one admission (keys used in CONCEPT_PROBS)."""
    groups = {
        "circulatory_dx": CATEGORIES["circulatory"],
        "respiratory_dx": CATEGORIES["respiratory"],
        "renal_failure_dx": RENAL_FAILURE,
        "diabetes_dx": DIABETES,
        "sepsis_dx": SEPSIS,
        "fall_dx": FALL,
    }
    triggers = {name for name, group in groups.items() if any(group.contains(c, v) for c, v in codes)}
    if lab_max.get("creatinine", 0.0) > 2.0:
        triggers.add("creatinine_above_2")
    if age is not None and age >= 75:
        triggers.add("age_75_plus")
    return triggers


def generate_notes(tables_dir: str | Path, seed: int, scale: int = 1) -> GeneratedNotes:
    """scale synthetic discharge notes per admission (default one), plus concept and negation tables.

    With scale 1 the output is the original one-note-per-admission set. With
    scale S > 1 each admission gets S notes: note 1 is the scale-1 note, notes
    2..S re-phrase it with their own seeds and the same tagged and negated
    concepts; note_id is "<subject>-DS-<admission seq>-<s>" and note_seq is s.
    """
    if scale < 1:
        raise ValueError("scale must be at least 1")
    con = connect_tables(
        tables_dir,
        required=("admissions", "patients", "diagnoses_icd", "d_icd_diagnoses", "labevents"),
        optional=("prescriptions",),
    )
    data = _AdmissionData.load(con)

    notes, concepts, negations = [], [], []
    seq_by_subject: dict[int, int] = defaultdict(int)
    for adm in data.admissions:
        seq_by_subject[adm["subject_id"]] += 1
        note_seq = seq_by_subject[adm["subject_id"]]
        note_id = f"{adm['subject_id']}-DS-{note_seq}"
        text, tagged, negated = _compose_note(adm, data, seed)
        variants = [(note_id, note_seq, text)]
        if scale > 1:
            variants = [(f"{note_id}-1", 1, text)]
            for s in range(2, scale + 1):
                variant_text, _, _ = _compose_note(adm, data, seed, variant=s, fixed=(tagged, negated))
                variants.append((f"{note_id}-{s}", s, variant_text))
        for vid, vseq, vtext in variants:
            notes.append(
                {
                    "note_id": vid,
                    "subject_id": adm["subject_id"],
                    "hadm_id": adm["hadm_id"],
                    "note_type": "DS",
                    "note_seq": vseq,
                    "charttime": adm["dischtime"],
                    "text": vtext,
                }
            )
            concepts += [{"note_id": vid, "hadm_id": adm["hadm_id"], "concept": c} for c in tagged]
            negations += [{"note_id": vid, "hadm_id": adm["hadm_id"], "concept": c} for c in negated]

    columns = ["note_id", "hadm_id", "concept"]
    return GeneratedNotes(
        discharge=pd.DataFrame(notes, columns=["note_id", "subject_id", "hadm_id", "note_type", "note_seq", "charttime", "text"]),
        concepts=pd.DataFrame(concepts, columns=columns),
        negations=pd.DataFrame(negations, columns=columns),
    )


@dataclass
class _AdmissionData:
    admissions: list[dict]
    diagnoses: dict[int, list[tuple[str, int, str]]]  # hadm -> (code, version, long_title) by seq_num
    labs: dict[int, dict[str, tuple[float, float]]]  # hadm -> lab name -> (max, latest)
    medications: dict[int, list[str]]

    @classmethod
    def load(cls, con: duckdb.DuckDBPyConnection) -> _AdmissionData:
        has_location = "discharge_location" in _columns(con, "admissions")
        location = "a.discharge_location" if has_location else "NULL"
        admissions = [
            {
                "subject_id": int(r[0]),
                "hadm_id": int(r[1]),
                "admittime": r[2],
                "dischtime": r[3],
                "admission_type": r[4],
                "discharge_location": r[5],
                "gender": r[6],
                "age": None if r[7] is None else int(r[7]),
            }
            for r in con.execute(
                f"""
                SELECT a.subject_id, a.hadm_id,
                       strftime(CAST(a.admittime AS TIMESTAMP), '%Y-%m-%d %H:%M:%S'),
                       strftime(CAST(a.dischtime AS TIMESTAMP), '%Y-%m-%d %H:%M:%S'),
                       a.admission_type, {location}, p.gender,
                       p.anchor_age + (year(CAST(a.admittime AS TIMESTAMP)) - p.anchor_year)
                FROM admissions a LEFT JOIN patients p USING (subject_id)
                ORDER BY a.subject_id, a.admittime, a.hadm_id
                """
            ).fetchall()
        ]

        diagnoses: dict[int, list[tuple[str, int, str]]] = defaultdict(list)
        for hadm, code, version, title in con.execute(
            """
            SELECT d.hadm_id, d.icd_code, d.icd_version, coalesce(t.long_title, d.icd_code)
            FROM diagnoses_icd d
            LEFT JOIN d_icd_diagnoses t ON t.icd_code = d.icd_code AND t.icd_version = d.icd_version
            ORDER BY d.hadm_id, d.seq_num
            """
        ).fetchall():
            diagnoses[int(hadm)].append((str(code), int(version), str(title)))

        by_itemid = {lab.itemid: lab.name for lab in LABS.values()}
        labs: dict[int, dict[str, tuple[float, float]]] = defaultdict(dict)
        for hadm, itemid, max_value, latest in lab_summary(con, by_itemid):
            labs[int(hadm)][by_itemid[int(itemid)]] = (float(max_value), float(latest))

        medications: dict[int, list[str]] = defaultdict(list)
        if has_table(con, "prescriptions"):
            for hadm, drug in con.execute(
                """
                SELECT DISTINCT hadm_id, trim(drug) FROM prescriptions
                WHERE hadm_id IS NOT NULL AND drug IS NOT NULL AND trim(drug) <> ''
                ORDER BY 1, 2
                """
            ).fetchall():
                medications[int(hadm)].append(str(drug))

        return cls(admissions, dict(diagnoses), dict(labs), dict(medications))


def _columns(con: duckdb.DuckDBPyConnection, table: str) -> set[str]:
    return {row[0] for row in con.execute(f"DESCRIBE {table}").fetchall()}


MAX_VARIANT_ATTEMPTS = 200


def _compose_note(
    adm: dict,
    data: _AdmissionData,
    seed: int,
    variant: int = 1,
    fixed: tuple[list[str], list[str]] | None = None,
) -> tuple[str, list[str], list[str]]:
    """Build one note. Returns (text, tagged concepts, negated concepts).

    With fixed=None the concepts are drawn (the original note). With fixed =
    (tagged, negated) the note mentions exactly those concepts; its phrasing is
    seeded by variant, retrying when the sampled data (diagnoses, medications)
    would mention a concept outside fixed.
    """
    if fixed is None:
        return _compose_drawn(adm, data, random.Random(f"{seed}:{adm['hadm_id']}"))
    for attempt in range(MAX_VARIANT_ATTEMPTS):
        rng = random.Random(f"{seed}:{adm['hadm_id']}:v{variant}:{attempt}")
        result = _compose_fixed(adm, data, rng, set(fixed[0]), list(fixed[1]))
        if result is not None:
            return result
    raise RuntimeError(f"hadm {adm['hadm_id']}: no phrasing of variant {variant} keeps the fixed concepts")


def _sections(adm: dict, data: _AdmissionData, rng: random.Random) -> tuple[str, str, str, str, str]:
    hadm = adm["hadm_id"]
    return (
        _opening(adm),
        _course(data.diagnoses.get(hadm, []), rng),
        _labs_paragraph(data.labs.get(hadm, {}), rng),
        _medications(data.medications.get(hadm, []), rng),
        _disposition(adm["discharge_location"]),
    )


def _compose_fixed(
    adm: dict, data: _AdmissionData, rng: random.Random, tagged: set[str], negated: list[str]
) -> tuple[str, list[str], list[str]] | None:
    opening, course, lab_text, meds, disposition = _sections(adm, data, rng)
    forced = mentioned_concepts(" ".join((opening, course, lab_text, meds, disposition)))
    if not forced <= tagged:
        return None
    clinical = [rng.choice(POSITIVE_PHRASES[c]) for c in CONCEPTS if c in tagged - forced]
    clinical += [rng.choice(NEGATED_PHRASES[c]) for c in negated]
    rng.shuffle(clinical)
    return _assemble(adm["hadm_id"], rng, opening, course, lab_text, meds, disposition, clinical, tagged, negated)


def _compose_drawn(adm: dict, data: _AdmissionData, rng: random.Random) -> tuple[str, list[str], list[str]]:
    hadm = adm["hadm_id"]
    diagnoses = data.diagnoses.get(hadm, [])
    labs = data.labs.get(hadm, {})
    opening, course, lab_text, meds, disposition = _sections(adm, data, rng)

    # Concepts already mentioned by the data itself (e.g. a "Chest pain" diagnosis
    # title or an insulin prescription) are tagged, never sampled or negated.
    forced = mentioned_concepts(" ".join((opening, course, lab_text, meds, disposition)))
    triggers = admission_triggers(
        [(c, v) for c, v, _ in diagnoses], {name: mx for name, (mx, _) in labs.items()}, adm["age"]
    )
    draws = {c: (rng.random(), rng.random()) for c in CONCEPTS}
    sampled = {c for c in CONCEPTS if draws[c][0] < concept_probability(c, triggers)}
    tagged = forced | sampled
    negated = [c for c in CONCEPTS if c not in tagged and draws[c][1] < CONCEPT_PROBS[c]["negated"]]

    clinical = [rng.choice(POSITIVE_PHRASES[c]) for c in CONCEPTS if c in sampled - forced]
    clinical += [rng.choice(NEGATED_PHRASES[c]) for c in negated]
    rng.shuffle(clinical)
    return _assemble(hadm, rng, opening, course, lab_text, meds, disposition, clinical, tagged, negated)


def _assemble(
    hadm: int,
    rng: random.Random,
    opening: str,
    course: str,
    lab_text: str,
    meds: str,
    disposition: str,
    clinical: list[str],
    tagged: set[str],
    negated: list[str],
) -> tuple[str, list[str], list[str]]:
    body = [opening, course, " ".join(clinical), lab_text]
    filler_pool = list(FILLER_SENTENCES)
    rng.shuffle(filler_pool)
    target = rng.randint(MIN_WORDS, FILLER_TARGET_MAX)
    filler: list[str] = []
    while _word_count(body + filler + [meds, disposition]) < target and filler_pool:
        filler.append(filler_pool.pop())

    paragraphs = [opening + " " + course, " ".join(clinical), lab_text, " ".join(filler), meds, disposition]
    text = "\n\n".join(p for p in paragraphs if p)

    words = len(text.split())
    if not MIN_WORDS <= words <= MAX_WORDS:
        raise RuntimeError(f"hadm {hadm}: note has {words} words, outside [{MIN_WORDS}, {MAX_WORDS}]")
    found = mentioned_concepts(text)
    expected = tagged | set(negated)
    if found != expected:
        raise RuntimeError(f"hadm {hadm}: concepts in text {sorted(found)} != tagged + negated {sorted(expected)}")
    return text, [c for c in CONCEPTS if c in tagged], negated


def _word_count(parts: list[str]) -> int:
    return sum(len(p.split()) for p in parts)


def _opening(adm: dict) -> str:
    sex = {"M": "man", "F": "woman"}.get(str(adm["gender"]), "patient")
    age = f"{adm['age']}-year-old" if adm["age"] is not None else "Adult"
    kind = str(adm["admission_type"] or "").upper()
    if "ELECTIVE" in kind:
        how = "electively"
    elif "OBSERVATION" in kind:
        how = "for observation"
    elif "EMER" in kind or "URGENT" in kind:
        how = "through the emergency department"
    else:
        how = "to the hospital"
    date = str(adm["admittime"])[:10]
    return f"{age} {sex} admitted {how} on {date}."


def _course(diagnoses: list[tuple[str, int, str]], rng: random.Random) -> str:
    titles = [title for _, _, title in diagnoses[:MAX_DIAGNOSES]]
    if not titles:
        return "Hospital course: The patient was admitted for evaluation and management."
    text = f"Hospital course: The principal diagnosis was {titles[0]}."
    rest = titles[1:]
    if rest:
        listed = rest[0] if len(rest) == 1 else ", ".join(rest[:-1]) + " and " + rest[-1]
        text += f" {rng.choice(_SECONDARY_LEADS)} {listed}."
    return text


def _labs_paragraph(labs: dict[str, tuple[float, float]], rng: random.Random) -> str:
    parts = []
    for name, lab in LABS.items():
        if name in labs:
            mx, latest = labs[name]
            parts.append(
                f"{name} {rng.choice(_LAB_VERBS)} {mx:.{lab.decimals}f} {lab.unit} "
                f"(most recent {latest:.{lab.decimals}f} {lab.unit})"
            )
    if not parts:
        return "Pertinent labs: no values were recorded for this admission."
    return "Pertinent labs: " + "; ".join(parts) + "."


def _medications(drugs: list[str], rng: random.Random) -> str:
    if not drugs:
        return "Discharge medications: see the medication reconciliation."
    chosen = rng.sample(drugs, min(MAX_MEDICATIONS, len(drugs)))
    return "Discharge medications: " + ", ".join(chosen) + "."


def _disposition(location: str | None) -> str:
    if location is None or str(location).strip() == "" or str(location) == "nan":
        return "Disposition: discharged home in stable condition."
    if str(location).upper() == "DIED":
        return "Disposition: the patient died during the admission."
    return f"Disposition: discharged to {str(location).lower()} in stable condition."


def write_notes(notes: GeneratedNotes, out_dir: str | Path) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / "discharge.csv.gz", out / "note_concepts.csv.gz", out / "note_negations.csv.gz"]
    for frame, path in zip((notes.discharge, notes.concepts, notes.negations), paths):
        frame.to_csv(path, index=False, compression="gzip")
    return paths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tables-dir", required=True, help="hosp tables folder, or a folder containing hosp/")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scale", type=int, default=1, help="notes per admission; > 1 writes to <out-dir>/scale_S/")
    args = parser.parse_args(argv)

    notes = generate_notes(args.tables_dir, args.seed, args.scale)
    out_dir = Path(args.out_dir) if args.scale == 1 else Path(args.out_dir) / f"scale_{args.scale}"
    for path in write_notes(notes, out_dir):
        print(f"wrote {path}")
    words = notes.discharge["text"].str.split().str.len()
    print(f"{len(notes.discharge)} notes, words min {words.min()} median {int(words.median())} max {words.max()}")


if __name__ == "__main__":
    main()
