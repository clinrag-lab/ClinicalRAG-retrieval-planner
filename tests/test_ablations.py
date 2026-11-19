"""The runner's ablation switches, --no-fallback and --no-probes, on the fake (fixture) sources."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from clinrag.config import PlannerConfig
from clinrag.eval import runner
from clinrag.eval.calibrate import probe_defaults, relational_queries
from clinrag.planner.enumerate import enumerate_plans
from clinrag.planner.planner import CostPlanner
from clinrag.planner.probes import DEFAULTS, TOTAL_NOTES, ProbeDefaults, default_probes, run_probes
from clinrag.sources.base import Sources
from clinrag.workload.schema import write_jsonl
from tests.test_classifier import workload  # noqa: F401  (module-scoped fixture workload)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def sources():
    return runner.build_sources("fake")


def first(questions, qclass):
    return next(q for q in questions if q.spec.qclass == qclass)


# ---- --no-fallback ----------------------------------------------------------------------------


def test_no_fallback_runs_the_chosen_plan_once(sources, workload):  # noqa: F811
    # An unreachable similarity floor fails every note result, so the full router falls back.
    config = replace(PlannerConfig(), min_similarity=2.0)
    question = first(workload.test, "C")
    router = runner.build_router("oracle", sources, config, Path("."), 0.5)

    full = runner.run_question("oracle", router, question, sources, config, repeats=1)
    ablated = runner.run_question("oracle", router, question, sources, config, repeats=1, max_fallbacks=0)

    assert full["fallbacks_used"] >= 1 and len(full["attempts"]) >= 2
    assert ablated["fallbacks_used"] == 0 and len(ablated["attempts"]) == 1
    assert ablated["attempts"][0]["plan_name"] == full["attempts"][0]["plan_name"]
    assert ablated["evidence_ok"] is False  # the failed check is recorded, not acted on


# ---- --no-probes ------------------------------------------------------------------------------


class NoProbing:
    """Wraps a source; the probe methods raise, everything else is passed through."""

    def __init__(self, inner, forbidden):
        self._inner, self._forbidden = inner, set(forbidden)

    def __getattr__(self, name):
        if name in self._forbidden:
            raise AssertionError(f"{name} called with --no-probes")
        return getattr(self._inner, name)


def guarded(sources: Sources) -> Sources:
    return Sources(
        relational=NoProbing(sources.relational, {"count", "count_matches", "matching_ids", "count_notes"}),
        # count_notes_for stays allowed: per-admission questions read their admission's note
        # count (index metadata) in both modes; the order-choice path never reaches it here.
        notes=NoProbing(sources.notes, {"probe_search"}),
        graph=NoProbing(sources.graph, {"expand", "descendant_count"}),
    )


DEFAULTS_SMALL = ProbeDefaults(relational_rows=7, graph_codes=3)


@pytest.mark.parametrize("qclass", ["C", "D"])
def test_no_probes_costs_every_plan_from_defaults(sources, workload, qclass):  # noqa: F811
    question = first(workload.test, qclass).spec
    config = replace(PlannerConfig(), s=0.5)  # results-demo-v1's s
    decision = CostPlanner(guarded(sources), config, DEFAULTS_SMALL).plan(question)

    total = sources.notes.total_notes()
    probes = decision.probes
    assert probes.ran == (TOTAL_NOTES, DEFAULTS)
    assert probes.relational_rows == 7
    assert probes.matching_notes == probes.matching_admissions == round(total * 0.3)
    assert probes.probe_scores == ()
    assert probes.graph_codes == (3 if qclass == "D" else None)
    # Still enumerates and costs every plan.
    assert [c.plan.name for c in decision.candidates] == [p.name for p in enumerate_plans(question)]
    assert all(c.cost is not None for c in decision.candidates)
    # f / N = 0.3: below s = 0.5 the order rule always picks filter_first, above s = 0.1 search_first.
    assert decision.order.order == "filter_first"
    low_s = CostPlanner(guarded(sources), replace(config, s=0.1), DEFAULTS_SMALL).plan(question)
    assert low_s.order.order == "search_first"


def test_no_probes_leaves_questions_without_an_order_choice_alone(sources, workload):  # noqa: F811
    for qclass in ("A", "B"):
        question = first(workload.test, qclass).spec
        got = default_probes(question, sources, PlannerConfig(), DEFAULTS_SMALL)
        want = run_probes(question, sources, PlannerConfig())
        assert replace(got, probe_ms=0.0) == replace(want, probe_ms=0.0)


def test_p2_no_probes_widened_query_all_uses_defaults_too(sources, workload):  # noqa: F811
    from clinrag.planner.classifier import SourceClassifier
    from clinrag.planner.router import ClassifierCostRouter

    model = SourceClassifier("tfidf").fit([q.spec for q in workload.dev])
    router = ClassifierCostRouter(model, guarded(sources), PlannerConfig(), probe_defaults=DEFAULTS_SMALL)
    for question in workload.test:
        router.decide(question.spec)  # would raise if any probe ran


def test_probe_defaults_are_calibration_medians(sources, workload):  # noqa: F811
    defaults = probe_defaults(workload.dev, sources)
    rows = [len(sources.relational.run(q).items) for q in relational_queries(workload.dev, sources)]
    cats = sorted({q.spec.category for q in workload.dev if q.spec.category})
    assert defaults.relational_rows == round(float(np.median(rows)))
    assert defaults.graph_codes == round(float(np.median([len(sources.graph.expand(c)) for c in cats])))
    assert defaults.matching_notes_share == 0.3


# ---- the CLI ----------------------------------------------------------------------------------


@pytest.mark.parametrize("flag", ["--no-fallback", "--no-probes"])
def test_cli_flags_reach_run_json_and_rows(workload, tmp_path, flag):  # noqa: F811
    write_jsonl(tmp_path / "workload" / "dev.jsonl", workload.dev)
    write_jsonl(tmp_path / "workload" / "test.jsonl", workload.test)
    (tmp_path / "config.yaml").write_text("s: 0.5\n", encoding="utf-8")
    out = runner.main([
        "--sources", "fake", "--workload", str(tmp_path / "workload"), "--split", "test", "--repeats", "1",
        "--routers", "oracle", "p2_tfidf", "--config", str(tmp_path / "config.yaml"), "--no-analysis",
        "--out", str(tmp_path / "run"), flag,
    ])
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (out / "results.jsonl").open(encoding="utf-8")]
    assert len(rows) == 2 * len(workload.test)
    if flag == "--no-fallback":
        assert run["no_fallback"] is True and run["probe_defaults"] is None
        assert all(r["no_fallback"] and not r["no_probes"] and r["fallbacks_used"] == 0 for r in rows)
    else:
        assert run["no_probes"] is True and run["probe_defaults"]["matching_notes_share"] == 0.3
        assert all(r["no_probes"] and not r["no_fallback"] for r in rows)
        mixed = [r for r in rows if r["qclass"] in ("C", "D") and r["router"] == "oracle"]
        assert mixed and all("filter_first" in r["attempts"][0]["plan_name"] for r in mixed)
