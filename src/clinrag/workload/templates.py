"""Question templates for the four workload classes, with gold answers.

A  relational only     per-admission lab questions (most recent, max, ever above T)
B  notes only          which discharge summaries mention a concept
C  filter + notes      among admissions with a lab above T, which summaries mention a concept
D  graph + notes       among admissions with a diagnosis category, which summaries mention a concept

Gold answers come from SQL over the hosp tables and the note_concepts table.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import duckdb

from clinrag.question import GRAPH, NOTES, RELATIONAL, QuestionSpec, StructuredFilter
from clinrag.workload.icd_categories import CATEGORIES
from clinrag.workload.labs import LABS, Lab
from clinrag.workload.notes_gen import CONCEPTS
from clinrag.workload.schema import WorkloadQuestion
from clinrag.workload.tables import connect_tables, lab_summary, load_csv, table_path

# Target selectivities (fraction of all admissions) for class C thresholds.
SELECTIVITY_TARGETS: tuple[float, ...] = (0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 0.90)

A_KINDS = ("latest", "max", "ever_above")
_PLACEHOLDER_QID = "unassigned"


def slug(text: str) -> str:
    return text.replace(" ", "_")


@dataclass
class WorkloadData:
    """Hosp tables plus note_concepts in one DuckDB connection."""

    con: duckdb.DuckDBPyConnection

    @classmethod
    def load(cls, tables_dir: str | Path, notes_dir: str | Path) -> WorkloadData:
        con = connect_tables(tables_dir, required=("admissions", "diagnoses_icd", "labevents"))
        path = table_path(Path(notes_dir), "note_concepts")
        if path is None:
            raise FileNotFoundError(f"note_concepts.csv(.gz) not found in {notes_dir}")
        load_csv(con, "note_concepts", path, {"concept": "VARCHAR", "note_id": "VARCHAR"})
        return cls(con)

    def admission_ids(self) -> list[int]:
        return [int(r[0]) for r in self.con.execute("SELECT DISTINCT hadm_id FROM admissions ORDER BY 1").fetchall()]

    @property
    def n_admissions(self) -> int:
        return len(self.admission_ids())

    def lab_values(self, lab: Lab) -> dict[int, tuple[float, float]]:
        """hadm_id -> (max, latest) for admissions with at least one value."""
        return {int(h): (float(mx), float(latest)) for h, _, mx, latest in lab_summary(self.con, [lab.itemid])}

    def concept_admissions(self, concept: str) -> set[int]:
        rows = self.con.execute(
            "SELECT DISTINCT n.hadm_id FROM note_concepts n JOIN admissions a USING (hadm_id) WHERE n.concept = ?",
            [concept],
        ).fetchall()
        return {int(r[0]) for r in rows}

    def lab_filter_admissions(self, lab: Lab, threshold: float) -> set[int]:
        """Admissions with any value of the lab above threshold."""
        rows = self.con.execute(
            """
            SELECT DISTINCT l.hadm_id FROM labevents l JOIN admissions a USING (hadm_id)
            WHERE l.itemid = ? AND l.valuenum > ?
            """,
            [lab.itemid, threshold],
        ).fetchall()
        return {int(r[0]) for r in rows}

    def filtered_concept_admissions(self, lab: Lab, threshold: float, concept: str) -> list[int]:
        """Class C gold: admissions above the lab threshold whose note mentions the concept."""
        rows = self.con.execute(
            """
            SELECT DISTINCT l.hadm_id
            FROM labevents l
            JOIN admissions a USING (hadm_id)
            JOIN note_concepts n USING (hadm_id)
            WHERE l.itemid = ? AND l.valuenum > ? AND n.concept = ?
            ORDER BY 1
            """,
            [lab.itemid, threshold, concept],
        ).fetchall()
        return [int(r[0]) for r in rows]

    def category_admissions(self, category: str) -> set[int]:
        cat = CATEGORIES[category]
        rows = self.con.execute(
            "SELECT DISTINCT d.hadm_id, d.icd_code, d.icd_version FROM diagnoses_icd d JOIN admissions a USING (hadm_id)"
        ).fetchall()
        return {int(h) for h, code, version in rows if cat.contains(str(code), int(version))}


# ---- class A ------------------------------------------------------------------


def class_a(data: WorkloadData, per_template: int | None, rng: random.Random) -> list[WorkloadQuestion]:
    """Per-admission lab questions, up to per_template admissions per (kind, lab) template."""
    questions = []
    for lab in LABS.values():
        values = data.lab_values(lab)
        for kind in A_KINDS:
            hadms = sorted(values)
            if per_template is not None and len(hadms) > per_template:
                hadms = sorted(rng.sample(hadms, per_template))
            for h in hadms:
                questions.append(_a_question(kind, lab, h, values[h]))
    return questions


def _a_question(kind: str, lab: Lab, hadm_id: int, values: tuple[float, float]) -> WorkloadQuestion:
    mx, latest = values
    if kind == "latest":
        text = f"What was the most recent {lab.name} value during admission {hadm_id}?"
        flt = StructuredFilter(lab.itemid, hadm_id=hadm_id)
        answer: float | int = latest
    elif kind == "max":
        text = f"What was the highest {lab.name} value during admission {hadm_id}?"
        flt = StructuredFilter(lab.itemid, hadm_id=hadm_id)
        answer = mx
    else:
        t = lab.above_threshold
        text = f"Was {lab.name} ever above {t:g} {lab.unit} during admission {hadm_id}?"
        flt = StructuredFilter(lab.itemid, ">", t, hadm_id=hadm_id)
        answer = int(mx > t)
    spec = QuestionSpec(
        qid=_PLACEHOLDER_QID,
        text=text,
        qclass="A",
        relational_filter=flt,
        required_sources=frozenset({RELATIONAL}),
    )
    gold = {"answer": answer, "template_id": f"A_{kind}_{lab.name}", "hadm_id": hadm_id, "unit": lab.unit}
    return WorkloadQuestion(spec, gold)


# ---- class B ------------------------------------------------------------------


def class_b(data: WorkloadData, concepts: Iterable[str] = CONCEPTS) -> list[WorkloadQuestion]:
    questions = []
    for concept in concepts:
        spec = QuestionSpec(
            qid=_PLACEHOLDER_QID,
            text=f"Which discharge summaries mention {concept}?",
            qclass="B",
            note_query=concept,
            required_sources=frozenset({NOTES}),
        )
        gold = {"answer": sorted(data.concept_admissions(concept)), "template_id": f"B_{slug(concept)}"}
        questions.append(WorkloadQuestion(spec, gold))
    return questions


# ---- class C ------------------------------------------------------------------


def thresholds_for(
    per_admission_max: Sequence[float],
    n_admissions: int,
    targets: Sequence[float] = SELECTIVITY_TARGETS,
    decimals: int = 1,
) -> list[float]:
    """Thresholds T so that about target * n_admissions admissions have max > T.

    For each target, n = round(target * n_admissions) and T is the n-th largest
    per-admission max (0-based), rounded; ties and rounding make the actual
    selectivity a little lower. Targets that need n < 1 or n >= the number of
    admissions with the lab are skipped. Duplicates are dropped; order follows targets.
    """
    desc = sorted(per_admission_max, reverse=True)
    out: list[float] = []
    for target in targets:
        n = round(target * n_admissions)
        if n < 1 or n >= len(desc):
            continue
        t = round(desc[n], decimals)
        if t not in out:
            out.append(t)
    return out


def class_c(
    data: WorkloadData,
    thresholds: dict[str, list[float]] | None = None,
    concepts: Iterable[str] = CONCEPTS,
) -> list[WorkloadQuestion]:
    """Lab-threshold filter plus concept. thresholds defaults to thresholds_for each lab."""
    n = data.n_admissions
    questions = []
    for lab in LABS.values():
        if thresholds is None:
            maxes = [mx for mx, _ in data.lab_values(lab).values()]
            lab_thresholds = thresholds_for(maxes, n, decimals=lab.decimals)
        else:
            lab_thresholds = thresholds.get(lab.name, [])
        for t in lab_thresholds:
            selectivity = len(data.lab_filter_admissions(lab, t)) / n
            for concept in concepts:
                spec = QuestionSpec(
                    qid=_PLACEHOLDER_QID,
                    text=(
                        f"Among admissions with {lab.name} > {t:g} {lab.unit}, "
                        f"which discharge summaries mention {concept}?"
                    ),
                    qclass="C",
                    relational_filter=StructuredFilter(lab.itemid, ">", t),
                    note_query=concept,
                    required_sources=frozenset({RELATIONAL, NOTES}),
                )
                gold = {
                    "answer": data.filtered_concept_admissions(lab, t, concept),
                    "template_id": f"C_{lab.name}_{slug(concept)}",
                    "selectivity": selectivity,
                    "threshold": t,
                    "unit": lab.unit,
                }
                questions.append(WorkloadQuestion(spec, gold))
    return questions


# ---- class D ------------------------------------------------------------------


def class_d(data: WorkloadData, concepts: Iterable[str] = CONCEPTS) -> list[WorkloadQuestion]:
    n = data.n_admissions
    questions = []
    for name, category in CATEGORIES.items():
        in_category = data.category_admissions(name)
        for concept in concepts:
            spec = QuestionSpec(
                qid=_PLACEHOLDER_QID,
                text=(
                    f"Among admissions with any {category.label} diagnosis, "
                    f"which discharge summaries mention {concept}?"
                ),
                qclass="D",
                note_query=concept,
                category=name,
                required_sources=frozenset({GRAPH, RELATIONAL, NOTES}),
            )
            gold = {
                "answer": sorted(in_category & data.concept_admissions(concept)),
                "template_id": f"D_{name}_{slug(concept)}",
                "selectivity": len(in_category) / n,
            }
            questions.append(WorkloadQuestion(spec, gold))
    return questions


# ---- filtering, capping and splitting ------------------------------------------


MAX_GOLD_FRACTION = 0.95


def gold_is_usable(q: WorkloadQuestion, n_admissions: int) -> bool:
    """List golds must be non-empty and cover at most 95% of admissions."""
    answer = q.gold["answer"]
    if not isinstance(answer, list):
        return True
    return 0 < len(answer) <= MAX_GOLD_FRACTION * n_admissions


def per_template_quota(max_per_class: int, n_templates: int) -> int:
    return math.ceil(max_per_class / n_templates)
