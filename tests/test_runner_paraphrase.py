"""Runner on a paraphrased (held-out phrasing) workload, with P2 trained on dev plus train-family paraphrases."""

import csv
import json
from pathlib import Path

import pytest

from clinrag.eval import runner
from clinrag.planner.classifier import SourceClassifier
from clinrag.workload import paraphrase
from clinrag.workload.paraphrase import paraphrase_all, split_families
from clinrag.workload.schema import read_jsonl, write_jsonl
from tests.test_classifier import workload  # noqa: F401  (module-scoped fixture workload)

FIXTURES = Path(__file__).parent / "fixtures"
ROUTERS = ["p1", "p2_tfidf", "oracle", "query_all"]


@pytest.fixture(scope="module")
def paraphrase_run(workload, tmp_path_factory):  # noqa: F811
    base = tmp_path_factory.mktemp("e2e")
    write_jsonl(base / "workload" / "dev.jsonl", workload.dev)
    write_jsonl(base / "workload" / "test.jsonl", workload.test)
    paraphrase.main(["--workload-dir", str(base / "workload"), "--out-dir", str(base / "paraphrase"), "--seed", "0"])
    out = runner.main([
        "--sources", "fake", "--tables-dir", str(FIXTURES), "--notes-dir", str(FIXTURES),
        "--workload", str(base / "paraphrase"), "--split", "test", "--train-workload", str(base / "workload"),
        "--p2-train", "dev+paraphrases", "--p2-model-dir", str(base / "models"),
        "--routers", *ROUTERS, "--repeats", "1", "--no-analysis", "--fig6", "--out", str(base / "run"),
    ])
    return base, out


def test_runs_every_held_out_phrasing(paraphrase_run, workload):  # noqa: F811
    base, out = paraphrase_run
    records = [json.loads(line) for line in (out / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    expected = paraphrase_all(workload.test, split_families(0)["test"], 0)
    assert len(records) == len(ROUTERS) * len(expected)
    assert {r["qid"] for r in records} == {p.question.spec.qid for p in expected}
    for r in records:
        assert set(r["plan_missed_sources"]) == set(r["required_sources"]) - set(r["plan_sources"])
    oracle = [r for r in records if r["router"] == "oracle"]
    assert all(r["plan_missed_sources"] == [] for r in oracle)


def test_summary_reports_missed_source_rate(paraphrase_run):
    _, out = paraphrase_run
    with (out / "results.csv").open(encoding="utf-8") as f:
        rows = {(r["router"], r["qclass"]): r for r in csv.DictReader(f)}
    assert float(rows[("oracle", "all")]["missed_source_rate"]) == 0.0
    assert 0.0 <= float(rows[("p1", "all")]["missed_source_rate"]) <= 1.0
    assert "missed_source_rate" in (out / "results.md").read_text(encoding="utf-8")


def test_fig6_is_written(paraphrase_run):
    _, out = paraphrase_run
    assert (out / "figures" / "fig6_paraphrase_e2e.png").read_bytes()[:4] == b"\x89PNG"


def test_p2_model_is_trained_on_dev_plus_paraphrases_and_reused(paraphrase_run, workload, monkeypatch):  # noqa: F811
    base, _ = paraphrase_run
    path = base / "models" / "p2_tfidf.joblib"
    assert path.exists()
    saved = SourceClassifier.load(path)
    # Same training data as the routing report: dev plus the dev train-family paraphrases.
    fresh = runner.train_classifier(base / "workload", "tfidf", "dev+paraphrases")
    question = read_jsonl(base / "paraphrase" / "test.jsonl")[0].spec
    assert saved.predict_proba(question) == pytest.approx(fresh.predict_proba(question))
    assert len(fresh.vectorizer.vocabulary_) > len(runner.train_classifier(base / "workload", "tfidf").vectorizer.vocabulary_)

    def fail(*args, **kwargs):
        raise AssertionError("should reuse the saved model")

    monkeypatch.setattr(runner, "train_classifier", fail)
    loaded = runner.load_or_train_classifiers(["p2_tfidf"], base / "workload", "dev+paraphrases", base / "models")
    assert set(loaded) == {"tfidf"}
