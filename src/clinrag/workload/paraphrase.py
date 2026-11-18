"""Paraphrased workload questions in held-out phrasing families.

Every workload question comes from a fixed template, so any router that has
seen the template routes it perfectly. This module re-renders each question in
several phrasing families per class: lab abbreviations and synonyms, other
comparison verbs, ICD category synonyms, implicit note cues that never say
"discharge summary" or "mention", and terse clinician-style forms.

A paraphrase is rendered from the question's slots (template kind, lab,
threshold, admission id, concept, category), not by editing its text, so the
QuestionSpec slots and the gold answer are copied unchanged. Only the text and
the qid change: qid gets a suffix _p<k>, where k is the family's 1-based index
in its class.

Families are split per class into train and test halves (seeded), so a
classifier can be trained on some phrasings and tested on phrasings it never saw.

Usage:
    python -m clinrag.workload.paraphrase --workload-dir data/workload \\
        --out-dir data/workload/paraphrase --seed 0
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from clinrag.workload.labs import LABS, Lab
from clinrag.workload.schema import WorkloadQuestion, question_to_dict, read_jsonl

# ---- vocabulary -------------------------------------------------------------------------


@dataclass(frozen=True)
class LabTerms:
    abbr: tuple[str, ...]  # Cr, K+, Hgb
    formal: tuple[str, ...]  # serum creatinine
    lay: tuple[str, ...]  # kidney function marker (used after "the" or "the patient's")


LAB_TERMS: dict[str, LabTerms] = {
    "creatinine": LabTerms(("Cr",), ("serum creatinine",), ("kidney function marker",)),
    "potassium": LabTerms(("K", "K+"), ("serum potassium",), ("potassium level",)),
    "hemoglobin": LabTerms(("Hgb", "Hb"), ("blood hemoglobin",), ("blood count",)),
}


@dataclass(frozen=True)
class Comparison:
    past: str  # "Cr exceeded 2"
    base: str  # "did Cr ever exceed 2"


COMPARISONS: tuple[Comparison, ...] = (
    Comparison("exceeded", "exceed"),
    Comparison("came in over", "come in over"),
    Comparison("was elevated past", "get elevated past"),
    Comparison("topped", "top"),
)


@dataclass(frozen=True)
class ConceptTerms:
    abbr: str  # terse form: HD, SOB
    past: str  # singular subject: "the patient went on dialysis", "who was intubated"
    plural: str  # plural subject: "which patients were intubated"
    base: str  # "did any patients go on dialysis"
    yes_no: str  # "Was the patient dialyzed" (question opener, no "?")
    gerund: str  # "any record of the patient going on dialysis"


CONCEPT_TERMS: dict[str, ConceptTerms] = {
    "dialysis": ConceptTerms("HD", "went on dialysis", "went on dialysis", "go on dialysis", "Was the patient dialyzed", "going on dialysis"),
    "chest pain": ConceptTerms("CP", "had chest pain", "had chest pain", "have chest pain", "Did the patient have chest pain", "having chest pain"),
    "shortness of breath": ConceptTerms(
        "SOB", "was short of breath", "were short of breath", "become short of breath", "Was the patient short of breath", "being short of breath"
    ),
    "anticoagulation": ConceptTerms(
        "anticoag", "was anticoagulated", "were anticoagulated", "get anticoagulated", "Was the patient anticoagulated", "being anticoagulated"
    ),
    "insulin": ConceptTerms("insulin", "was on insulin", "were on insulin", "receive insulin", "Was the patient on insulin", "receiving insulin"),
    "sepsis": ConceptTerms("sepsis", "was septic", "were septic", "develop sepsis", "Was the patient septic", "being septic"),
    "fall": ConceptTerms("fall", "had a fall", "had a fall", "have a fall", "Did the patient fall", "falling"),
    "intubation": ConceptTerms("intubation", "was intubated", "were intubated", "need intubation", "Was the patient intubated", "being intubated"),
}


# Concepts that need an article when used as a noun ("the record shows a fall").
CONCEPT_NOUNS: dict[str, str] = {"fall": "a fall"}


@dataclass(frozen=True)
class CategoryTerms:
    modifier: tuple[str, ...]  # "Among cardiac patients"; modifier[0] is the terse form
    noun: tuple[str, ...]  # "a history of heart disease"


CATEGORY_TERMS: dict[str, CategoryTerms] = {
    "circulatory": CategoryTerms(("cardiac", "heart-related", "cardiovascular"), ("heart disease", "cardiovascular disease")),
    "renal_genitourinary": CategoryTerms(("GU", "kidney or urinary", "renal/GU"), ("kidney or urinary tract disease",)),
    "respiratory": CategoryTerms(("pulm", "lung or breathing", "pulmonary"), ("lung disease", "a breathing disorder")),
    "endocrine": CategoryTerms(("endo", "hormonal or metabolic", "gland-related"), ("a hormonal or metabolic disorder",)),
    "heart_failure": CategoryTerms(("CHF",), ("CHF", "congestive heart failure")),
    "ischemic_heart_disease": CategoryTerms(("CAD",), ("coronary artery disease", "CAD")),
    "atrial_fibrillation": CategoryTerms(("AFib", "AF"), ("AFib", "an irregular heart rhythm")),
    "chronic_kidney_disease": CategoryTerms(("CKD",), ("CKD", "long-standing kidney damage")),
    "acute_kidney_injury": CategoryTerms(("AKI",), ("AKI", "acute renal failure")),
    "pneumonia": CategoryTerms(("PNA",), ("a lung infection", "PNA")),
    "copd": CategoryTerms(("emphysema",), ("emphysema or chronic bronchitis",)),
    "diabetes": CategoryTerms(("DM", "diabetic"), ("DM", "diabetes mellitus")),
    "hypothyroidism": CategoryTerms(("hypothyroid",), ("an underactive thyroid", "low thyroid")),
}

_LAB_BY_ITEMID: dict[int, Lab] = {lab.itemid: lab for lab in LABS.values()}


# ---- slots ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Slots:
    """What a family needs to render a question. kind: latest, max, ever_above, list, admission, C, D."""

    qclass: str
    kind: str
    lab: Lab | None = None
    threshold: str | None = None  # formatted, e.g. "2" or "5.1"
    hadm_id: int | None = None
    concept: str | None = None
    category: str | None = None

    @property
    def unit(self) -> str:
        assert self.lab is not None
        return self.lab.unit

    @property
    def lab_terms(self) -> LabTerms:
        assert self.lab is not None
        return LAB_TERMS[self.lab.name]

    @property
    def concept_noun(self) -> str:
        """The concept as a noun phrase: "dialysis", "a fall"."""
        assert self.concept is not None
        return CONCEPT_NOUNS.get(self.concept, self.concept)

    @property
    def concept_terms(self) -> ConceptTerms:
        assert self.concept is not None
        return CONCEPT_TERMS[self.concept]

    @property
    def category_terms(self) -> CategoryTerms:
        assert self.category is not None
        return CATEGORY_TERMS[self.category]


def slots_of(question: WorkloadQuestion) -> Slots:
    spec, tid = question.spec, question.template_id
    flt = spec.relational_filter
    lab = _LAB_BY_ITEMID[flt.itemid] if flt is not None else None
    threshold = f"{flt.threshold:g}" if flt is not None and flt.threshold is not None else None
    if spec.qclass == "A":
        kind = next(k for k in ("ever_above", "latest", "max") if tid.startswith(f"A_{k}_"))
        assert flt is not None
        return Slots("A", kind, lab=lab, threshold=threshold, hadm_id=flt.hadm_id)
    if spec.qclass == "B":
        kind = "admission" if spec.note_hadm_id is not None else "list"
        return Slots("B", kind, hadm_id=spec.note_hadm_id, concept=spec.note_query)
    if spec.qclass == "C":
        return Slots("C", "C", lab=lab, threshold=threshold, concept=spec.note_query)
    return Slots("D", "D", concept=spec.note_query, category=spec.category)


# ---- families ---------------------------------------------------------------------------

Render = Callable[[Slots, random.Random], str]


@dataclass(frozen=True)
class Family:
    name: str
    qclass: str
    render: Render


def _pick(rng: random.Random, options: Sequence[str]) -> str:
    return options[rng.randrange(len(options))]


def _cmp(rng: random.Random) -> Comparison:
    return COMPARISONS[rng.randrange(len(COMPARISONS))]


def _any_lab_term(s: Slots, rng: random.Random) -> str:
    t = s.lab_terms
    return _pick(rng, (*t.abbr, *t.formal, f"the {t.lay[0]}"))


def _lab_result_term(s: Slots, rng: random.Random) -> str:
    """A lab name that reads well before "result": abbreviation or formal name."""
    return _pick(rng, (*s.lab_terms.abbr, *s.lab_terms.formal))


def _by_kind(**forms: Render) -> Render:
    def render(s: Slots, rng: random.Random) -> str:
        return forms[s.kind](s, rng)

    return render


# Class A: kinds latest, max, ever_above.
A_FAMILIES: tuple[Family, ...] = (
    Family("A_abbrev", "A", _by_kind(
        latest=lambda s, r: f"What was the last {_pick(r, s.lab_terms.abbr)} in admission {s.hadm_id}?",
        max=lambda s, r: f"What was the peak {_pick(r, s.lab_terms.abbr)} during admission {s.hadm_id}?",
        ever_above=lambda s, r: f"Did {_pick(r, s.lab_terms.abbr)} ever {_cmp(r).base} {s.threshold} {s.unit} in admission {s.hadm_id}?",
    )),
    Family("A_terse", "A", _by_kind(
        latest=lambda s, r: f"latest {_pick(r, s.lab_terms.abbr)} for {s.hadm_id}?",
        max=lambda s, r: f"max {_pick(r, s.lab_terms.abbr)}, adm {s.hadm_id}",
        ever_above=lambda s, r: f"{_pick(r, s.lab_terms.abbr)} ever > {s.threshold}, adm {s.hadm_id}?",
    )),
    Family("A_formal", "A", _by_kind(
        latest=lambda s, r: f"Report the most recent {s.lab_terms.formal[0]} measurement for hospital stay {s.hadm_id}.",
        max=lambda s, r: f"Report the highest {s.lab_terms.formal[0]} measurement recorded during hospital stay {s.hadm_id}.",
        ever_above=lambda s, r: (
            f"Report whether {s.lab_terms.formal[0]} {_cmp(r).past} {s.threshold} {s.unit} at any point during hospital stay {s.hadm_id}."
        ),
    )),
    Family("A_lay", "A", _by_kind(
        latest=lambda s, r: f"What was the patient's {s.lab_terms.lay[0]} at the end of stay {s.hadm_id}?",
        max=lambda s, r: f"How high did the patient's {s.lab_terms.lay[0]} get during stay {s.hadm_id}?",
        ever_above=lambda s, r: (
            f"During stay {s.hadm_id}, did the patient's {s.lab_terms.lay[0]} ever {_cmp(r).base} {s.threshold} {s.unit}?"
        ),
    )),
    Family("A_draw", "A", _by_kind(
        latest=lambda s, r: f"What did the final {s.lab.name} draw show for admission {s.hadm_id}?",
        max=lambda s, r: f"What was the top {s.lab.name} reading in admission {s.hadm_id}?",
        ever_above=lambda s, r: f"Did any {s.lab.name} draw in admission {s.hadm_id} {_cmp(r).base} {s.threshold} {s.unit}?",
    )),
    Family("A_stay_first", "A", _by_kind(
        latest=lambda s, r: f"For stay {s.hadm_id}, what was the most recent {_lab_result_term(s, r)} result?",
        max=lambda s, r: f"For stay {s.hadm_id}, what was the maximum {_lab_result_term(s, r)} result?",
        ever_above=lambda s, r: (
            f"For stay {s.hadm_id}, was there a {_lab_result_term(s, r)} result that "
            f"{_cmp(r).past} {s.threshold} {s.unit}?"
        ),
    )),
    Family("A_chart", "A", _by_kind(
        latest=lambda s, r: f"Pull up encounter {s.hadm_id}: what's the latest {_pick(r, s.lab_terms.abbr)}?",
        max=lambda s, r: f"Pull up encounter {s.hadm_id}: what's the highest {_pick(r, s.lab_terms.abbr)} on file?",
        ever_above=lambda s, r: f"Pull up encounter {s.hadm_id}: any {_pick(r, s.lab_terms.abbr)} that {_cmp(r).past} {s.threshold}?",
    )),
    Family("A_value", "A", _by_kind(
        latest=lambda s, r: f"Give the last recorded {s.lab_terms.formal[0]} value for admission {s.hadm_id}.",
        max=lambda s, r: f"Give the peak recorded {s.lab_terms.formal[0]} value for admission {s.hadm_id}.",
        ever_above=lambda s, r: (
            f"Answer yes or no: did {s.lab_terms.formal[0]} {_cmp(r).base} {s.threshold} {s.unit} during admission {s.hadm_id}?"
        ),
    )),
)

# Class B: kinds list (which notes mention X) and admission (yes/no for one stay).
B_FAMILIES: tuple[Family, ...] = (
    Family("B_record", "B", _by_kind(
        list=lambda s, r: f"Which patients have any record of {s.concept_terms.gerund}?",
        admission=lambda s, r: f"Any record of the patient {s.concept_terms.gerund} in stay {s.hadm_id}?",
    )),
    Family("B_natural", "B", _by_kind(
        list=lambda s, r: f"Which patients {s.concept_terms.plural}?",
        admission=lambda s, r: f"{s.concept_terms.yes_no} during stay {s.hadm_id}?",
    )),
    Family("B_terse", "B", _by_kind(
        list=lambda s, r: f"pts w/ {s.concept_terms.abbr}?",
        admission=lambda s, r: f"{s.concept_terms.abbr} in adm {s.hadm_id}? y/n",
    )),
    Family("B_document", "B", _by_kind(
        list=lambda s, r: f"Which stays document {s.concept_noun}?",
        admission=lambda s, r: f"Does the chart for stay {s.hadm_id} document {s.concept_noun}?",
    )),
    Family("B_notes", "B", _by_kind(
        list=lambda s, r: f"Which clinical notes refer to {s.concept_noun}?",
        admission=lambda s, r: f"Do the notes for admission {s.hadm_id} say anything about {s.concept_noun}?",
    )),
    Family("B_course", "B", _by_kind(
        list=lambda s, r: f"In which hospital courses did {s.concept_noun} come up?",
        admission=lambda s, r: f"Did {s.concept_noun} come up during the hospital course of stay {s.hadm_id}?",
    )),
    Family("B_command", "B", _by_kind(
        list=lambda s, r: f"List every admission where the patient {s.concept_terms.past}.",
        admission=lambda s, r: f"Confirm whether the patient {s.concept_terms.past} in admission {s.hadm_id}.",
    )),
    Family("B_writeup", "B", _by_kind(
        list=lambda s, r: f"Find admissions whose write-ups describe {s.concept_noun}.",
        admission=lambda s, r: f"Is {s.concept_noun} written up for admission {s.hadm_id}?",
    )),
)

# Class C: lab threshold filter plus a note concept.
C_FAMILIES: tuple[Family, ...] = (
    Family("C_implicit", "C", lambda s, r: (
        f"Of the stays where {_pick(r, s.lab_terms.abbr)} {_cmp(r).past} {s.threshold}, which ones document {s.concept_noun}?"
    )),
    Family("C_terse", "C", lambda s, r: (
        f"{_pick(r, s.lab_terms.abbr)} > {s.threshold} w/ {s.concept_terms.abbr}: which adms?"
    )),
    Family("C_who", "C", lambda s, r: (
        f"Which patients whose {s.lab_terms.formal[0]} {_cmp(r).past} {s.threshold} {s.unit} {s.concept_terms.plural}?"
    )),
    Family("C_lay", "C", lambda s, r: (
        f"Among patients whose {s.lab_terms.lay[0]} {_cmp(r).past} {s.threshold} {s.unit}, who {s.concept_terms.past}?"
    )),
    Family("C_records", "C", lambda s, r: (
        f"List admissions where {_any_lab_term(s, r)} {_cmp(r).past} {s.threshold} {s.unit} and the record shows {s.concept_noun}."
    )),
    Family("C_question_first", "C", lambda s, r: (
        f"Who {s.concept_terms.past}, among stays where {_pick(r, s.lab_terms.abbr)} {_cmp(r).past} {s.threshold}?"
    )),
    Family("C_notes", "C", lambda s, r: (
        f"For admissions with {s.lab_terms.formal[0]} over {s.threshold} {s.unit}, which notes reference {s.concept_noun}?"
    )),
    Family("C_cohort", "C", lambda s, r: (
        f"In the cohort whose {_any_lab_term(s, r).removeprefix('the ')} {_cmp(r).past} {s.threshold} {s.unit}, "
        f"which stays involved {s.concept_noun}?"
    )),
)

# Class D: ICD category plus a note concept.
D_FAMILIES: tuple[Family, ...] = (
    Family("D_cohort", "D", lambda s, r: (
        f"Among {_pick(r, s.category_terms.modifier)} patients, which ones {s.concept_terms.plural}?"
    )),
    Family("D_terse", "D", lambda s, r: f"{s.category_terms.modifier[0]} pts w/ {s.concept_terms.abbr}?"),
    Family("D_coded", "D", lambda s, r: (
        f"For stays coded with {_pick(r, s.category_terms.noun)}, which notes mention {s.concept_noun}?"
    )),
    Family("D_history", "D", lambda s, r: (
        f"Which patients with a history of {_pick(r, s.category_terms.noun)} {s.concept_terms.plural}?"
    )),
    Family("D_dx", "D", lambda s, r: (
        f"Among admissions with {_pick(r, s.category_terms.modifier)} diagnoses, who had {s.concept_noun} documented?"
    )),
    Family("D_question", "D", lambda s, r: (
        f"Did any {_pick(r, s.category_terms.modifier)} patients {s.concept_terms.base}? List the stays."
    )),
    Family("D_known", "D", lambda s, r: (
        f"Of the patients known to have {_pick(r, s.category_terms.noun)}, which ones {s.concept_terms.plural}?"
    )),
    Family("D_problem_list", "D", lambda s, r: (
        f"Find stays with {_pick(r, s.category_terms.noun)} on the problem list where the patient {s.concept_terms.past}."
    )),
)

FAMILIES: dict[str, tuple[Family, ...]] = {"A": A_FAMILIES, "B": B_FAMILIES, "C": C_FAMILIES, "D": D_FAMILIES}
FAMILY_BY_NAME: dict[str, Family] = {f.name: f for fams in FAMILIES.values() for f in fams}


# ---- generation -------------------------------------------------------------------------


@dataclass(frozen=True)
class Paraphrase:
    """A reworded workload question. question.spec and question.gold equal the source's except text and qid."""

    question: WorkloadQuestion
    family: str
    source_qid: str

    def to_dict(self) -> dict[str, Any]:
        d = question_to_dict(self.question)
        d["phrasing_family"] = self.family
        d["source_qid"] = self.source_qid
        return d


def split_families(seed: int, test_fraction: float = 0.5) -> dict[str, frozenset[str]]:
    """Per class, a seeded shuffle of the families; round(n * test_fraction) go to test, the rest to train.

    Returns {"train": names, "test": names}, disjoint.
    """
    train: set[str] = set()
    test: set[str] = set()
    for qclass, families in FAMILIES.items():
        names = [f.name for f in families]
        random.Random(f"{seed}:families:{qclass}").shuffle(names)
        n_test = round(len(names) * test_fraction)
        test.update(names[:n_test])
        train.update(names[n_test:])
    return {"train": frozenset(train), "test": frozenset(test)}


def paraphrase(question: WorkloadQuestion, family: Family, seed: int = 0) -> Paraphrase:
    """Render question in family. Deterministic for (seed, qid, family)."""
    if family.qclass != question.spec.qclass:
        raise ValueError(f"family {family.name} is for class {family.qclass}, question {question.spec.qid} is {question.spec.qclass}")
    index = FAMILIES[family.qclass].index(family) + 1
    rng = random.Random(f"{seed}:{question.spec.qid}:{family.name}")
    text = family.render(slots_of(question), rng)
    spec = replace(question.spec, qid=f"{question.spec.qid}_p{index}", text=text)
    return Paraphrase(WorkloadQuestion(spec, question.gold), family.name, question.spec.qid)


def paraphrase_all(
    questions: Iterable[WorkloadQuestion],
    families: Iterable[str] | None = None,
    seed: int = 0,
) -> list[Paraphrase]:
    """Every question rendered in each of its class's families (restricted to the given family names)."""
    allowed = None if families is None else set(families)
    out = []
    for q in questions:
        for family in FAMILIES[q.spec.qclass]:
            if allowed is None or family.name in allowed:
                out.append(paraphrase(q, family, seed))
    return out


def write_paraphrases(path: str | Path, paraphrases: Iterable[Paraphrase]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for p in paraphrases:
            f.write(json.dumps(p.to_dict(), sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workload-dir", type=Path, default=Path("data/workload"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/workload/paraphrase"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    split = split_families(args.seed)
    dev = paraphrase_all(read_jsonl(args.workload_dir / "dev.jsonl"), split["train"], args.seed)
    test = paraphrase_all(read_jsonl(args.workload_dir / "test.jsonl"), split["test"], args.seed)
    write_paraphrases(args.out_dir / "dev.jsonl", dev)
    write_paraphrases(args.out_dir / "test.jsonl", test)
    families = {name: sorted(names) for name, names in split.items()}
    (args.out_dir / "families.json").write_text(
        json.dumps({"seed": args.seed, **families, "n_dev": len(dev), "n_test": len(test)}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"dev (train families): {len(dev)}  test (test families): {len(test)}  -> {args.out_dir}")


if __name__ == "__main__":
    main()
