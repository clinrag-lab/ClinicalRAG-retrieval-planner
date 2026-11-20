"""Experiment runner end to end on the fixture workload and the fake (fixture) sources."""

import csv
import json
from pathlib import Path

import pytest

from clinrag.eval import runner
from clinrag.workload.schema import write_jsonl
from tests.test_classifier import workload  # noqa: F401  (module-scoped fixture workload)

FIXTURES = Path(__file__).parent / "fixtures"
ROUTERS = ["oracle", "p1", "p2_tfidf", "query_all", "always_filter_first", "always_search_first"]


@pytest.fixture(scope="module")
def run_dir(workload, tmp_path_factory):  # noqa: F811
    base = tmp_path_factory.mktemp("runner")
    write_jsonl(base / "workload" / "dev.jsonl", workload.dev)
    write_jsonl(base / "workload" / "test.jsonl", workload.test)
    out = runner.main([
        "--sources", "fake", "--tables-dir", str(FIXTURES), "--notes-dir", str(FIXTURES),
        "--workload", str(base / "workload"), "--split", "test", "--repeats", "2",
        "--routers", *ROUTERS, "--sweep-on", "original", "--out", str(base / "run"),
    ])
    return out, workload


def test_results_jsonl_is_valid(run_dir):
    out, wl = run_dir
    lines = (out / "results.jsonl").read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    assert len(records) == len(ROUTERS) * len(wl.test)
    required = {
        "router", "qid", "qclass", "plan_name", "predicted_sources", "required_sources", "planning_overhead_ms",
        "est_cost", "tokens", "latency_ms", "sources_queried", "fallbacks_used", "executed_plans",
        "recall", "recall_metric", "measured_cost", "selectivity", "template_id", "attempts", "latency_runs_ms",
    }
    for r in records:
        assert required <= set(r), required - set(r)
        assert 0.0 <= r["recall"] <= 1.0
        assert len(r["latency_runs_ms"]) == 2
        assert r["measured_cost"] == pytest.approx(r["latency_ms"] + r["tokens"])
        assert r["fallbacks_used"] == len(r["executed_plans"]) - 1
        if r["qclass"] in ("C", "D"):
            assert r["selectivity"] is not None
    assert {r["router"] for r in records} == set(ROUTERS)
    by_router = {r["router"]: r for r in records if r["qclass"] == "C"}
    assert by_router["query_all"]["plan_name"] == "query_all"
    assert by_router["always_filter_first"]["plan_name"].endswith("filter_first")
    assert by_router["always_search_first"]["plan_name"].endswith("search_first")


def test_results_csv_and_md(run_dir):
    out, wl = run_dir
    with (out / "results.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    overall = {r["router"]: r for r in rows if r["qclass"] == "all"}
    assert set(overall) == set(ROUTERS)
    assert all(int(r["n"]) == len(wl.test) for r in overall.values())
    assert set(rows[0]) == set(runner.SUMMARY_FIELDS)
    md = (out / "results.md").read_text(encoding="utf-8")
    assert "## All classes" in md and "## Per class" in md
    assert json.loads((out / "run.json").read_text(encoding="utf-8"))["n_questions"] == len(wl.test)


def test_analyses_and_figures_are_written(run_dir):
    out, _ = run_dir
    summary = json.loads((out / "analysis" / "summary.json").read_text(encoding="utf-8"))
    assert {"cost_model", "order_check", "overhead_share", "tau_sweep"} <= set(summary)
    assert [r["tau"] for r in summary["tau_sweep"]["rows"]] == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    assert {r["router"] for r in summary["overhead_share"]} == set(ROUTERS)
    for name in ("fig1_tokens_vs_recall", "fig2_order_vs_selectivity", "fig3_tau_sweep", "fig4_overhead"):
        path = out / "figures" / f"{name}.png"
        assert path.exists() and path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert (out / "analysis" / "analysis.md").read_text(encoding="utf-8").startswith("# Analyses")


def test_build_router_rejects_unknown():
    with pytest.raises(ValueError):
        runner.build_router("nope", None, None, Path("."), 0.5)
    with pytest.raises(ValueError):
        runner.build_sources("nope")
