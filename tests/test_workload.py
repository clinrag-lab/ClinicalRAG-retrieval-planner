"""Workload templates and builder on the fixture, with hand-written concept tags.

Fixture facts used below (tests/fixtures):
  creatinine max per admission: 01 1.1, 02 6.4, 03 0.9, 04 3.2, 05 1.0, 06 0.7, 07 7.1, 08 1.6
  circulatory (I codes): 01 02 04 05 07   renal (N codes): 02 04 07   endocrine (E codes): 01 02 08
Admission ids are written by their last two digits (01 = 20000001).
"""

import json
import random
from pathlib import Path

import pytest

from clinrag.question import StructuredFilter
from clinrag.workload import build as build_module
from clinrag.workload import notes_gen
from clinrag.workload.build import build_workload, split_templates
from clinrag.workload.schema import read_jsonl, write_jsonl
from clinrag.workload.tables import find_tables_dir
from clinrag.workload.templates import (
    WorkloadData,
    class_a,
    class_b,
    class_b_admission,
    class_c,
    class_d,
    gold_is_usable,
    thresholds_for,
)

FIXTURES = Path(__file__).parent / "fixtures"
EXPECTED_SOURCES = {
    "A": {"relational"},
    "B": {"notes"},
    "C": {"relational", "notes"},
    "D": {"graph", "relational", "notes"},
}


def h(n: int) -> int:
    return 20000000 + n


CONCEPT_TAGS = {
    "dialysis": [2, 7],
    "chest pain": [1, 6],
    "insulin": [1, 8],
    "fall": [1, 2, 3, 4, 5, 6, 7, 8],  # every admission: too broad, skipped by the builder
}
# Negated (untagged) mentions: "dialysis was not required" in 04, "denied chest pain" in 03.
NEGATION_TAGS = {"dialysis": [4], "chest pain": [3]}


def write_tags(path, tags):
    lines = ["note_id,hadm_id,concept"]
    for concept, ids in tags.items():
        lines += [f"n{i},{h(i)},{concept}" for i in ids]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def data(tmp_path):
    write_tags(tmp_path / "note_concepts.csv", CONCEPT_TAGS)
    write_tags(tmp_path / "note_negations.csv", NEGATION_TAGS)
    return WorkloadData.load(FIXTURES, tmp_path)


def by_template(questions):
    out = {}
    for q in questions:
        out.setdefault(q.template_id, []).append(q)
    return out


def gold_for(questions, template_id, hadm_id):
    return next(q.gold["answer"] for q in questions if q.template_id == template_id and q.gold["hadm_id"] == hadm_id)


# ---- class A ------------------------------------------------------------------


def test_class_a_gold(data):
    qs = class_a(data, per_template=None, rng=random.Random(0))
    # 04: creatinine 3.2 (01-10) then 1.8 (01-13)
    assert gold_for(qs, "A_latest_creatinine", h(4)) == pytest.approx(1.8)
    assert gold_for(qs, "A_max_creatinine", h(4)) == pytest.approx(3.2)
    assert gold_for(qs, "A_ever_above_creatinine", h(4)) == 1
    # 08: creatinine 1.6 only
    assert gold_for(qs, "A_ever_above_creatinine", h(8)) == 0
    # 02: creatinine 6.4 then 4.9; potassium 5.8
    assert gold_for(qs, "A_latest_creatinine", h(2)) == pytest.approx(4.9)
    assert gold_for(qs, "A_ever_above_potassium", h(2)) == 1
    # 07: potassium 6.0, hemoglobin 9.2 (below 12)
    assert gold_for(qs, "A_max_potassium", h(7)) == pytest.approx(6.0)
    assert gold_for(qs, "A_latest_hemoglobin", h(7)) == pytest.approx(9.2)
    assert gold_for(qs, "A_ever_above_hemoglobin", h(7)) == 0


def test_class_a_covers_admissions_with_each_lab(data):
    qs = by_template(class_a(data, per_template=None, rng=random.Random(0)))
    assert len(qs["A_max_creatinine"]) == 8
    assert sorted(q.gold["hadm_id"] for q in qs["A_max_potassium"]) == [h(1), h(2), h(4), h(6), h(7)]
    assert sorted(q.gold["hadm_id"] for q in qs["A_max_hemoglobin"]) == [h(1), h(3), h(5), h(7), h(8)]


def test_class_a_filter_targets_one_admission(data):
    q = next(q for q in class_a(data, None, random.Random(0)) if q.template_id == "A_ever_above_creatinine")
    assert q.spec.relational_filter == StructuredFilter(50912, ">", 2.0, hadm_id=q.gold["hadm_id"])


def test_class_a_quota(data):
    qs = class_a(data, per_template=3, rng=random.Random(0))
    assert all(len(v) == 3 for v in by_template(qs).values())


# ---- classes B, C, D ----------------------------------------------------------


def test_class_b_gold(data):
    gold = {q.spec.note_query: q.gold["answer"] for q in class_b(data, concepts=CONCEPT_TAGS)}
    assert gold["dialysis"] == [h(2), h(7)]
    assert gold["chest pain"] == [h(1), h(6)]


def test_class_b_list_template_ids(data):
    assert {q.template_id for q in class_b(data, concepts=["dialysis"])} == {"B_list_dialysis"}


def test_class_b_admission_balances_yes_no_and_uses_negations(data):
    qs = class_b_admission(data, per_concept=4, rng=random.Random(0), concepts=["dialysis"])
    # yes pool {02, 07} -> 2 yes; 2 no: one negated (04) and one plain from the rest.
    picks = {(q.gold["hadm_id"], q.gold["answer"], q.gold["negated_mention"]) for q in qs}
    assert {p for p in picks if p[1] == 1} == {(h(2), 1, False), (h(7), 1, False)}
    assert (h(4), 0, True) in picks
    plain_no = [p for p in picks if p[1] == 0 and not p[2]]
    assert len(plain_no) == 1 and plain_no[0][0] in {h(i) for i in (1, 3, 5, 6, 8)}

    q = next(q for q in qs if q.gold["hadm_id"] == h(4))
    assert q.spec.text == f"According to the discharge summary for admission {h(4)}, was dialysis performed?"
    assert q.spec.note_hadm_id == h(4)
    assert q.spec.required_sources == {"notes"}
    assert q.spec.note_query == "dialysis"
    assert q.template_id == "B_admission_dialysis"


def test_class_b_admission_tops_up_when_a_pool_is_short(data):
    # chest pain: yes pool {01, 06}, negated {03}; ask for 6 -> 2 yes (all), 4 no.
    qs = class_b_admission(data, per_concept=6, rng=random.Random(0), concepts=["chest pain"])
    answers = [q.gold["answer"] for q in qs]
    assert answers.count(1) == 2 and answers.count(0) == 4
    assert sum(q.gold["negated_mention"] for q in qs) == 1


def test_thresholds_from_max_distribution():
    maxes = [1.1, 6.4, 0.9, 3.2, 1.0, 0.7, 7.1, 1.6]
    # n = round(target * 8): 0, 0, 1, 2, 3, 4, 6, 7 -> skip the zeros
    assert thresholds_for(maxes, 8) == [6.4, 3.2, 1.6, 1.1, 0.9, 0.7]


def test_class_c_gold(data):
    qs = class_c(data, thresholds={"creatinine": [2.0]}, concepts=["dialysis", "chest pain"])
    gold = {q.spec.note_query: q.gold for q in qs}
    # creatinine > 2.0: 02, 04, 07
    assert gold["dialysis"]["answer"] == [h(2), h(7)]
    assert gold["dialysis"]["selectivity"] == pytest.approx(3 / 8)
    assert gold["chest pain"]["answer"] == []
    assert qs[0].spec.relational_filter == StructuredFilter(50912, ">", 2.0)


def test_class_d_gold(data):
    gold = {(q.spec.category, q.spec.note_query): q.gold for q in class_d(data, concepts=CONCEPT_TAGS)}
    assert gold[("circulatory", "chest pain")]["answer"] == [h(1)]
    assert gold[("circulatory", "chest pain")]["selectivity"] == pytest.approx(5 / 8)
    assert gold[("renal_genitourinary", "dialysis")]["answer"] == [h(2), h(7)]
    assert gold[("renal_genitourinary", "dialysis")]["selectivity"] == pytest.approx(3 / 8)
    assert gold[("endocrine", "insulin")]["answer"] == [h(1), h(8)]
    assert gold[("respiratory", "dialysis")]["answer"] == []


def test_gold_rule_skips_empty_and_too_broad(data):
    qs = {q.spec.note_query: q for q in class_b(data, concepts=["dialysis", "fall", "sepsis"])}
    assert gold_is_usable(qs["dialysis"], 8)
    assert not gold_is_usable(qs["fall"], 8)  # 8 of 8 admissions > 95%
    assert not gold_is_usable(qs["sepsis"], 8)  # empty


# ---- builder --------------------------------------------------------------------


def test_build_split_and_sources(data):
    workload = build_workload(data, seed=0)
    dev_templates = {q.template_id for q in workload.dev}
    test_templates = {q.template_id for q in workload.test}
    assert dev_templates and test_templates
    assert not dev_templates & test_templates

    every = workload.dev + workload.test
    for q in every:
        assert q.spec.required_sources == EXPECTED_SOURCES[q.spec.qclass], q.spec.qid
        assert gold_is_usable(q, 8)
    assert len({q.spec.qid for q in every}) == len(every)
    assert len({q.spec.text for q in every}) == len(every)
    # "fall" tags every admission: the B question is too broad and skipped, while
    # C and D questions on fall survive because their filter narrows the gold.
    assert "B_list_fall" not in {q.template_id for q in every}
    assert any(q.spec.qclass == "D" and q.spec.note_query == "fall" for q in every)


def test_build_b_mixes_list_and_admission_templates(data):
    workload = build_workload(data, seed=0)
    b = [q for q in workload.dev + workload.test if q.spec.qclass == "B"]
    assert len(b) <= 100
    assert any(q.template_id.startswith("B_list_") for q in b)
    admission = [q for q in b if q.template_id.startswith("B_admission_")]
    assert admission and all(q.spec.note_hadm_id == q.gold["hadm_id"] for q in admission)
    counts = workload.summary["b_admission_answers"]
    assert counts["yes"] + counts["no"] == len(admission)


def test_build_caps_each_class(data):
    workload = build_workload(data, seed=0, max_per_class=5)
    every = workload.dev + workload.test
    for qclass in "ABCD":
        assert sum(q.spec.qclass == qclass for q in every) <= 5


def test_split_templates_is_seeded_and_disjoint():
    ids = [f"T{i}" for i in range(10)]
    dev = split_templates(ids, 0.6, random.Random(3))
    assert dev == split_templates(ids, 0.6, random.Random(3))
    assert len(dev) == 6
    assert split_templates(["only"], 0.6, random.Random(0)) == {"only"}
    assert len(split_templates(["a", "b"], 0.9, random.Random(0))) == 1


def test_jsonl_round_trip(data, tmp_path):
    workload = build_workload(data, seed=0)
    write_jsonl(tmp_path / "q.jsonl", workload.dev)
    loaded = read_jsonl(tmp_path / "q.jsonl")
    assert [q.spec for q in loaded] == [q.spec for q in workload.dev]
    assert [q.gold for q in loaded] == [q.gold for q in workload.dev]


def test_end_to_end_cli_on_fixture(tmp_path):
    notes_dir, out_dir = tmp_path / "generated", tmp_path / "workload"
    notes_gen.main(["--tables-dir", str(FIXTURES), "--out-dir", str(notes_dir), "--seed", "0"])
    build_module.main(
        ["--tables-dir", str(FIXTURES), "--notes-dir", str(notes_dir), "--out-dir", str(out_dir), "--seed", "0"]
    )
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    dev, test = read_jsonl(out_dir / "dev.jsonl"), read_jsonl(out_dir / "test.jsonl")
    assert summary["total"] == {"dev": len(dev), "test": len(test)}
    assert summary["n_admissions"] == 8
    assert not {q.template_id for q in dev} & {q.template_id for q in test}


def test_find_tables_dir(tmp_path):
    hosp = tmp_path / "mimic" / "2.2" / "hosp"
    hosp.mkdir(parents=True)
    (hosp / "admissions.csv.gz").write_bytes(b"")
    assert find_tables_dir(tmp_path / "mimic") == hosp
    assert find_tables_dir(hosp) == hosp
    with pytest.raises(FileNotFoundError):
        find_tables_dir(tmp_path / "missing")
