"""Source classifier and its training script on the fixture workload. TF-IDF only; torch is never loaded."""

import json
import sys

import pytest

from clinrag.planner import train_classifier
from clinrag.planner.classifier import (
    SourceClassifier,
    mentions_admission_id,
    mentions_category,
    mentions_comparison,
    mentions_lab,
    mentions_notes,
    mentions_number,
    normalize_text,
    rule_features,
    threshold,
)
from clinrag.question import QuestionSpec, StructuredFilter
from clinrag.workload.build import build_workload
from clinrag.workload.schema import write_jsonl
from clinrag.workload.templates import WorkloadData
from tests.test_workload import CONCEPT_TAGS, FIXTURES, NEGATION_TAGS, write_tags


@pytest.fixture(scope="module")
def workload(tmp_path_factory):
    notes_dir = tmp_path_factory.mktemp("notes")
    write_tags(notes_dir / "note_concepts.csv", CONCEPT_TAGS)
    write_tags(notes_dir / "note_negations.csv", NEGATION_TAGS)
    return build_workload(WorkloadData.load(FIXTURES, notes_dir), seed=0)


@pytest.fixture(scope="module")
def fitted(workload):
    return SourceClassifier().fit([q.spec for q in workload.dev])


def spec(text, sources, **kw):
    return QuestionSpec(qid="x", text=text, qclass="A", required_sources=frozenset(sources), **kw)


def test_rule_features():
    a = "Was creatinine ever above 2 mg/dL during admission 20000004?"
    assert mentions_lab(a) and mentions_comparison(a) and mentions_number(a) and mentions_admission_id(a)
    assert not mentions_notes(a) and not mentions_category(a)
    b = "According to the discharge summary for admission 20000004, was dialysis performed?"
    assert mentions_notes(b) and mentions_admission_id(b)
    assert not mentions_number(b)  # an admission id is not a threshold
    assert not mentions_lab(b) and not mentions_comparison(b)
    d = "Among admissions with any heart failure diagnosis, which discharge summaries mention sepsis?"
    assert mentions_category(d) and mentions_notes(d)
    assert mentions_category("patients with COPD") and mentions_category("renal or genitourinary codes")
    assert list(rule_features(a)) == [1.0, 1.0, 0.0, 0.0, 1.0]


def test_normalize_text_hides_ids_and_numbers():
    assert normalize_text("creatinine > 2.5 in admission 20000004").split() == [
        "creatinine", ">", "num", "in", "admission", "admid",
    ]


def test_classifier_learns_fixture_workload(workload, fitted):
    # Templates are split between dev and test, so test questions use unseen templates.
    for q in workload.dev + workload.test:
        assert fitted.predict(q.spec, 0.5) == set(q.spec.required_sources), q.spec.text


def test_predict_proba_covers_every_source(fitted, workload):
    probs = fitted.predict_proba(workload.test[0].spec)
    assert set(probs) == {"relational", "notes", "graph"}
    assert all(0.0 <= p <= 1.0 for p in probs.values())
    assert fitted.predict_proba(workload.test[0].spec.text) == probs  # plain text works too


def test_predict_respects_tau(fitted, workload):
    for q in workload.test[:20]:
        probs = fitted.predict_proba(q.spec)
        for tau in (0.0, 0.1, 0.5, 0.9, 1.0):
            assert fitted.predict(q.spec, tau) == {s for s, p in probs.items() if p >= tau}
        assert fitted.predict(q.spec, 0.0) == {"relational", "notes", "graph"}
    assert threshold({"notes": 0.5, "graph": 0.49}, 0.5) == {"notes"}
    with pytest.raises(ValueError):
        threshold({"notes": 0.5}, 1.5)


def test_latency_is_recorded(fitted, workload):
    fitted.last_latency_ms = 0.0
    fitted.predict_proba(workload.test[0].spec)
    assert fitted.last_latency_ms > 0.0
    fitted.last_latency_ms = 0.0
    fitted.predict(workload.test[1].spec, 0.5)
    assert fitted.last_latency_ms > 0.0


def test_constant_label_gets_constant_probability():
    questions = [
        spec("What was the latest creatinine in admission 20000001?", {"relational"}, relational_filter=StructuredFilter(50912)),
        spec("Which discharge summaries mention sepsis?", {"notes"}, note_query="sepsis"),
    ]
    clf = SourceClassifier().fit(questions)
    assert clf.constant == {"graph": 0.0}
    assert clf.predict_proba("anything at all")["graph"] == 0.0


def test_save_load_round_trip(fitted, workload, tmp_path):
    path = fitted.save(tmp_path / "sub" / "clf.joblib")
    loaded = SourceClassifier.load(path)
    q = workload.test[0].spec
    assert loaded.predict_proba(q) == pytest.approx(fitted.predict_proba(q))


def test_unfitted_and_bad_featurizer():
    with pytest.raises(RuntimeError):
        SourceClassifier().predict_proba("x")
    with pytest.raises(ValueError):
        SourceClassifier(featurizer="word2vec")


def test_tfidf_never_imports_torch(fitted, workload):
    fitted.predict_proba(workload.test[0].spec)
    assert "sentence_transformers" not in sys.modules
    assert "torch" not in sys.modules


def test_training_script_writes_report(workload, tmp_path):
    write_jsonl(tmp_path / "workload" / "dev.jsonl", workload.dev)
    write_jsonl(tmp_path / "workload" / "test.jsonl", workload.test)
    out = tmp_path / "out" / "clf.joblib"
    train_classifier.main(["--workload-dir", str(tmp_path / "workload"), "--out", str(out)])

    assert out.exists()
    report = json.loads((tmp_path / "out" / "clf_report.json").read_text(encoding="utf-8"))
    assert report["featurizer"] == "tfidf"
    assert report["n_dev"] == len(workload.dev) and report["n_test"] == len(workload.test)
    assert report["dev_cv"]["folds"] == 5
    assert set(report["test"]["per_source"]) == {"relational", "notes", "graph"}
    assert report["test"]["exact_match"] == 1.0
    assert report["predict_latency_ms"]["n"] == len(workload.test)
    assert report["predict_latency_ms"]["p95_ms"] >= 0
    assert [row["tau"] for row in report["tau_sweep_test"]] == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for row in report["tau_sweep_test"]:
        assert 0.0 <= row["missed_needed_source_fraction"] <= 1.0
        assert 0.0 <= row["avg_sources_predicted"] <= 3.0


def test_set_metrics_and_tau_sweep():
    required = [frozenset({"notes"}), frozenset({"relational", "notes"})]
    predicted = [{"notes"}, {"notes"}]
    m = train_classifier.set_metrics(predicted, required)
    assert m["exact_match"] == 0.5
    assert m["per_source"]["relational"] == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 1}
    assert m["per_source"]["notes"]["f1"] == 1.0

    probs = [{"relational": 0.0, "notes": 0.9, "graph": 0.0}, {"relational": 0.35, "notes": 0.9, "graph": 0.0}]
    sweep = {row["tau"]: row for row in train_classifier.tau_sweep(probs, required)}
    assert sweep[0.3]["missed_needed_source_fraction"] == 0.0
    assert sweep[0.3]["avg_sources_predicted"] == 1.5
    assert sweep[0.4]["missed_needed_source_fraction"] == 0.5
    assert sweep[0.4]["avg_sources_predicted"] == 1.0
