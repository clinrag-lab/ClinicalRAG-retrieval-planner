"""Paraphrase generator and the held-out-phrasing routing report, on the fixture workload."""

import json
import re
from dataclasses import replace

import pytest

from clinrag.planner import train_classifier
from clinrag.question import GRAPH, NOTES, RELATIONAL, QuestionSpec, StructuredFilter
from clinrag.workload.icd_categories import CATEGORIES
from clinrag.workload.labs import LABS
from clinrag.workload.notes_gen import CONCEPTS
from clinrag.workload.paraphrase import (
    CATEGORY_TERMS,
    CONCEPT_TERMS,
    FAMILIES,
    LAB_TERMS,
    paraphrase,
    paraphrase_all,
    split_families,
)
from clinrag.workload.schema import WorkloadQuestion, write_jsonl
from tests.test_classifier import workload  # noqa: F401  (module-scoped fixture workload)

ALL_FAMILIES = [f for families in FAMILIES.values() for f in families]


def synthetic_questions():
    """One question per (template kind, lab), concept and category, so every vocabulary entry is rendered."""
    qs = []
    for lab in LABS.values():
        for kind, flt in (
            ("latest", StructuredFilter(lab.itemid, hadm_id=21234567)),
            ("max", StructuredFilter(lab.itemid, hadm_id=21234567)),
            ("ever_above", StructuredFilter(lab.itemid, ">", lab.above_threshold, hadm_id=21234567)),
        ):
            spec = QuestionSpec(qid=f"A_{kind}_{lab.name}", text="t", qclass="A", relational_filter=flt,
                                required_sources=frozenset({RELATIONAL}))
            qs.append(WorkloadQuestion(spec, {"answer": 1, "template_id": f"A_{kind}_{lab.name}"}))
        for concept in CONCEPTS:
            spec = QuestionSpec(qid=f"C_{lab.name}_{concept}", text="t", qclass="C",
                                relational_filter=StructuredFilter(lab.itemid, ">", 2.5), note_query=concept,
                                required_sources=frozenset({RELATIONAL, NOTES}))
            qs.append(WorkloadQuestion(spec, {"answer": [1], "template_id": f"C_{lab.name}_{concept}"}))
    for concept in CONCEPTS:
        for hadm in (None, 21234567):
            kind = "list" if hadm is None else "admission"
            spec = QuestionSpec(qid=f"B_{kind}_{concept}", text="t", qclass="B", note_query=concept, note_hadm_id=hadm,
                                required_sources=frozenset({NOTES}))
            qs.append(WorkloadQuestion(spec, {"answer": [1], "template_id": f"B_{kind}_{concept}"}))
        for category in CATEGORIES:
            spec = QuestionSpec(qid=f"D_{category}_{concept}", text="t", qclass="D", note_query=concept,
                                category=category, required_sources=frozenset({GRAPH, RELATIONAL, NOTES}))
            qs.append(WorkloadQuestion(spec, {"answer": [1], "template_id": f"D_{category}_{concept}"}))
    return qs


def test_vocabulary_covers_every_lab_concept_and_category():
    assert set(LAB_TERMS) == set(LABS)
    assert set(CONCEPT_TERMS) == set(CONCEPTS)
    assert set(CATEGORY_TERMS) == set(CATEGORIES)


def test_at_least_six_families_per_class():
    for qclass in "ABCD":
        assert len(FAMILIES[qclass]) >= 6
        assert all(f.qclass == qclass for f in FAMILIES[qclass])
    assert len({f.name for f in ALL_FAMILIES}) == len(ALL_FAMILIES)


def test_paraphrases_keep_gold_and_slots(workload):  # noqa: F811
    questions = workload.dev + workload.test
    paraphrases = paraphrase_all(questions, seed=0)
    by_qid = {q.spec.qid: q for q in questions}
    assert len(paraphrases) == sum(len(FAMILIES[q.spec.qclass]) for q in questions)
    for p in paraphrases:
        source = by_qid[p.source_qid]
        spec = p.question.spec
        assert p.question.gold == source.gold
        assert replace(spec, qid=source.spec.qid, text=source.spec.text) == source.spec
        index = [f.name for f in FAMILIES[spec.qclass]].index(p.family) + 1
        assert spec.qid == f"{source.spec.qid}_p{index}"
        assert spec.text != source.spec.text
        d = p.to_dict()
        assert d["phrasing_family"] == p.family and d["source_qid"] == p.source_qid and d["gold"] == source.gold


def test_families_are_disjoint_across_train_and_test():
    split = split_families(seed=0)
    assert not split["train"] & split["test"]
    assert split["train"] | split["test"] == {f.name for f in ALL_FAMILIES}
    for qclass, families in FAMILIES.items():
        names = {f.name for f in families}
        n_test = len(names & split["test"])
        assert n_test == round(len(names) / 2)
        assert names & split["train"]


def test_family_split_is_seeded():
    assert split_families(seed=3) == split_families(seed=3)
    assert any(split_families(seed=s) != split_families(seed=0) for s in range(1, 10))


def test_generation_is_deterministic_for_a_seed(workload):  # noqa: F811
    questions = workload.dev + workload.test
    first = [p.to_dict() for p in paraphrase_all(questions, seed=7)]
    assert first == [p.to_dict() for p in paraphrase_all(questions, seed=7)]
    others = [[p.to_dict() for p in paraphrase_all(questions, seed=s)] for s in range(8, 12)]
    assert any(o != first for o in others)


def test_train_and_test_items_use_their_own_families(workload):  # noqa: F811
    split = split_families(seed=0)
    train = paraphrase_all(workload.dev, split["train"], seed=0)
    test = paraphrase_all(workload.test, split["test"], seed=0)
    assert {p.family for p in train} <= split["train"]
    assert {p.family for p in test} <= split["test"]
    assert {p.family for p in train}.isdisjoint({p.family for p in test})


_BAD_TEXT = [
    (re.compile(r"[{}]|\bNone\b"), "unfilled slot"),
    (re.compile(r"\s{2,}"), "double space"),
    (re.compile(r"\s[?.,:]"), "space before punctuation"),
    (re.compile(r"\b(patients|ones)\s+was\b"), "plural subject with 'was'"),
    (re.compile(r"\b(a|an)\s+(a|an|the)\b"), "double article"),
    (re.compile(r"\b(\w+) \1\b"), "repeated word"),
]


@pytest.mark.parametrize("family", ALL_FAMILIES, ids=lambda f: f.name)
def test_every_family_produces_grammatical_text(family, workload):  # noqa: F811
    questions = [q for q in synthetic_questions() + workload.dev + workload.test if q.spec.qclass == family.qclass]
    assert questions
    for q in questions:
        text = paraphrase(q, family, seed=0).question.spec.text
        assert text.strip() == text and len(text.split()) >= 3, text
        for pattern, problem in _BAD_TEXT:
            assert not pattern.search(text), f"{problem}: {text!r}"
        assert text[-1] in "?." or text[-1].isalnum(), text


def test_implicit_note_families_avoid_template_cues(workload):  # noqa: F811
    questions = synthetic_questions() + workload.dev + workload.test
    for p in paraphrase_all(questions):
        assert "discharge summar" not in p.question.spec.text.lower()
        if p.family in {"B_record", "B_natural", "B_command", "B_terse", "C_implicit", "C_lay", "D_cohort", "D_history"}:
            assert "mention" not in p.question.spec.text.lower(), p.question.spec.text


def test_family_for_wrong_class_is_rejected(workload):  # noqa: F811
    a_question = next(q for q in workload.dev if q.spec.qclass == "A")
    with pytest.raises(ValueError):
        paraphrase(a_question, FAMILIES["B"][0])


def test_paraphrase_report_on_fixture(workload, tmp_path):  # noqa: F811
    write_jsonl(tmp_path / "workload" / "dev.jsonl", workload.dev)
    write_jsonl(tmp_path / "workload" / "test.jsonl", workload.test)
    report_path = tmp_path / "out" / "routing.json"
    train_classifier.main([
        "--workload-dir", str(tmp_path / "workload"), "--paraphrase", "--featurizers", "tfidf",
        "--out", str(tmp_path / "out" / "clf.joblib"), "--report", str(report_path),
    ])
    report = json.loads(report_path.read_text(encoding="utf-8"))

    split = split_families(seed=0)
    assert set(report["families"]["test"]) == split["test"]
    assert report["n_train"]["original_dev"] == len(workload.dev)
    assert report["n_eval"]["source_test_questions"] == len(workload.test)
    assert set(report["routers"]) == {"P1_rules", "P2_tfidf"}
    assert (tmp_path / "out" / "clf_paraphrase_tfidf.joblib").exists()
    for name, router in report["routers"].items():
        assert set(router["per_source"]) == {"relational", "notes", "graph"}
        assert 0.0 <= router["exact_match"] <= 1.0 and 0.0 <= router["missed_source_rate"] <= 1.0
        assert len(router["top_miss_families"]) <= 5
        for row in router["top_miss_families"]:
            assert row["family"] in split["test"] and row["missed"] > 0 and row["example"]["text"]
    assert report["routers"]["P1_rules"]["tau_sweep"] is None
    taus = [r["tau"] for r in report["routers"]["P2_tfidf"]["tau_sweep"]]
    assert taus == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def test_route_defaults():
    assert train_classifier.p1_route("Tell me about this patient.") == {"notes"}
    assert train_classifier.p1_route("latest creatinine for 21234567?") == {"relational"}
    assert train_classifier.p2_route({"relational": 0.2, "notes": 0.4, "graph": 0.1}, 0.5) == {"notes"}
    assert train_classifier.p2_route({"relational": 0.6, "notes": 0.4, "graph": 0.1}, 0.3) == {"relational", "notes"}
