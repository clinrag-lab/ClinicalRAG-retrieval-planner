"""Results freeze: SUMMARY.md and figure collection from a run directory laid out like final-demo."""

import csv
import json

import pytest

from clinrag.eval import freeze, runner
from clinrag.eval.runner import SUMMARY_FIELDS
from clinrag.workload.schema import write_jsonl
from tests.test_classifier import workload  # noqa: F401  (module-scoped fixture workload)

PNG = b"\x89PNG\r\n\x1a\nfake"


def write_results(path, routers):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for router in routers:
            for qclass in ("A", "all"):
                writer.writerow({k: "" for k in SUMMARY_FIELDS} | {
                    "router": router, "qclass": qclass, "n": 10, "mean_recall": 0.9, "mean_tokens": 700.0,
                    "mean_latency_ms": 5.0, "p95_latency_ms": 9.0, "mean_overhead_ms": 1.0,
                    "fallback_rate": 0.0, "missed_source_rate": 0.1,
                })


def make_run(root):
    (root / "config").mkdir(parents=True)
    (root / "config" / "calibrated_config.yaml").write_text("s: 0.5\nbeta: 1.0\n", encoding="utf-8")
    (root / "config" / "tune_order_report.json").write_text(json.dumps({
        "winner": {"s": 0.5, "beta": 1.0, "agreement": 0.746}, "default": {"agreement": 0.195}, "n_questions": 118,
    }), encoding="utf-8")
    write_results(root / "template-test" / "results.csv", ["oracle", "p1"])
    (root / "template-test" / "analysis").mkdir(parents=True)
    (root / "template-test" / "analysis" / "summary.json").write_text(json.dumps({
        "cost_model": {"overall": {"n_plans": 400, "spearman_cost": 0.88, "spearman_latency": 0.65,
                                   "spearman_tokens": 0.93, "cheapest_plan_agreement": 0.58}},
        "overhead_share": [{"router": "oracle", "mean_overhead_ms": 6.0, "mean_retrieval_ms": 6.5,
                            "overhead_share_of_total_time": 0.48}],
    }), encoding="utf-8")
    (root / "scale-study" / "analysis").mkdir(parents=True)
    (root / "scale-study" / "analysis" / "order_scale_summary.json").write_text(json.dumps({"summaries": [
        {"total_notes": 2750, "filter_first_better": 0.7, "policies": [
            {"policy": "selectivity_rule", "picks_better": 0.8, "mean_recall": 0.6, "mean_latency_ms": 11.0}]},
    ]}), encoding="utf-8")
    write_results(root / "paraphrase-test" / "results.csv", ["p1", "p2_tfidf"])
    (root / "routing").mkdir()
    (root / "routing" / "routing_paraphrase_report.json").write_text(json.dumps({"routers": {
        "P1_rules": {"exact_match": 0.41, "missed_source_rate": 0.59, "latency_ms": {"mean_ms": 0.04},
                     "per_source": {s: {"f1": 0.5} for s in ("relational", "notes", "graph")}},
        "P2_bge": {"skipped": "ImportError"},
    }}), encoding="utf-8")
    for name, src in freeze.FIGURES.items():
        (root / src / "figures").mkdir(parents=True, exist_ok=True)
        (root / src / "figures" / name).write_bytes(PNG)


def test_freeze_writes_summary_and_copies_figures(tmp_path):
    run = tmp_path / "final-demo"
    make_run(run)
    path = freeze.main(["--run", str(run)])
    text = path.read_text(encoding="utf-8")
    for heading in ("## Template test run", "## Cost model", "## Overhead share", "## Order check per scale",
                    "## End to end on held-out phrasings", "## Classifier routing", "## Figures"):
        assert heading in text
    # Every section carries the command that produced it.
    for commands in freeze.COMMANDS.values():
        for command in commands:
            assert command in text
    assert "s = 0.5, beta = 1.0" in text and "74.6%" in text
    assert "| P2_bge | skipped |" in text
    assert "| oracle | 0.900 | 700.0 |" in text
    assert sorted(p.name for p in (run / "figures").iterdir()) == sorted(freeze.FIGURES)


def run_router(base, router):
    from tests.test_runner_paraphrase import FIXTURES

    return runner.main([
        "--sources", "fake", "--tables-dir", str(FIXTURES), "--notes-dir", str(FIXTURES),
        "--workload", str(base / "workload"), "--split", "test", "--routers", router,
        "--repeats", "1", "--no-analysis", "--out", str(base / "parts" / router),
    ])


def test_merge_combines_one_router_per_run(workload, tmp_path):  # noqa: F811
    write_jsonl(tmp_path / "workload" / "test.jsonl", workload.test)
    parts = {r: run_router(tmp_path, r) for r in ("p1", "oracle")}
    questions = tmp_path / "workload" / "test.jsonl"
    freeze.main(["--merge", *(f"{r}={p}" for r, p in parts.items()), "--questions", str(questions),
                 "--into", str(tmp_path / "merged")])
    records = [json.loads(line) for line in (tmp_path / "merged" / "results.jsonl").read_text(encoding="utf-8")
               .splitlines()]
    qids = [q.spec.qid for q in workload.test]
    assert [r["router"] for r in records] == ["p1"] * len(qids) + ["oracle"] * len(qids)
    assert [r["qid"] for r in records] == qids * 2
    assert (tmp_path / "merged" / "results.md").exists()
    assert (tmp_path / "merged" / "figures" / "fig6_paraphrase_e2e.png").read_bytes()[:4] == PNG[:4]

    # A router missing a question is refused, not silently merged.
    path = parts["oracle"] / "results.jsonl"
    path.write_text("".join(path.read_text(encoding="utf-8").splitlines(keepends=True)[:-1]), encoding="utf-8")
    with pytest.raises(ValueError, match="oracle"):
        freeze.merge_parts(parts, questions, tmp_path / "merged2")
