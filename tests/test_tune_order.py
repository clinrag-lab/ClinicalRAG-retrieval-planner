"""Order-rule tuner: grid scoring by hand, and measurement on the fakes with planted latencies."""

from dataclasses import replace

import pytest

from clinrag.config import load_planner_config, save_planner_config
from clinrag.eval import tune_order
from clinrag.eval.tune_order import measure_orders, pick_winner, score_grid
from clinrag.planner.execute import execute as real_execute
from tests.test_analysis import LOOSE, TIGHT, sources
from tests.test_planner import CONFIG


def rows():
    # k = 5, N = 1000. Row 1: f = 40; row 2: f = 600.
    return [
        {"total_notes": 1000, "matching_notes": 40,
         "filter_first": {"recall": 1.0, "latency_ms": 5.0},
         "search_first": {1.0: {"recall": 0.5, "latency_ms": 3.0}, 3.0: {"recall": 1.0, "latency_ms": 4.0}}},
        {"total_notes": 1000, "matching_notes": 600,
         "filter_first": {"recall": 1.0, "latency_ms": 9.0},
         "search_first": {1.0: {"recall": 1.0, "latency_ms": 2.0}, 3.0: {"recall": 1.0, "latency_ms": 2.5}}},
    ]


def test_score_grid_hand_computed():
    # Selectivities: row 1 4%, row 2 60%.
    grid = {(g["s"], g["beta"]): g for g in score_grid(rows(), k=5, ss=(0.02, 0.1), betas=(1.0, 3.0))}
    # beta 1: better = ff (recall) for row 1, sf (latency tie-break) for row 2.
    # s 0.1: 4% < 10% -> ff, 60% -> sf: both right.
    assert grid[(0.1, 1.0)]["agreement"] == 1.0
    assert grid[(0.1, 1.0)]["mean_latency_ms"] == pytest.approx((5.0 + 2.0) / 2)
    # s 0.02: 4% >= 2% -> sf for row 1: wrong.
    assert grid[(0.02, 1.0)]["agreement"] == 0.5
    assert grid[(0.02, 1.0)]["mean_recall"] == 0.75
    # beta 3: better = sf for both (recall tie, sf faster). s 0.1 -> ff for row 1: wrong.
    assert grid[(0.1, 3.0)]["agreement"] == 0.5
    assert grid[(0.02, 3.0)]["agreement"] == 1.0
    assert grid[(0.02, 3.0)]["picks_filter_first"] == 0.0


def test_winner_breaks_ties_by_recall_then_latency():
    grid = score_grid(rows(), k=5, ss=(0.02, 0.1), betas=(1.0, 3.0))
    # (0.1, 1.0) and (0.02, 3.0) both agree on 100% with mean recall 1.0;
    # latency 3.5 vs 3.25 ms picks (0.02, 3.0).
    winner = pick_winner(grid)
    assert (winner["s"], winner["beta"]) == (0.02, 3.0)


@pytest.fixture
def planted(monkeypatch):
    def fake_execute(plan, src, question, config):
        result = real_execute(plan, src, question, config)
        search = plan.note_search
        ms = 2.0 if search.mode == "filter_first" else 1.0 + search.n_candidates / 100
        return replace(result, total_latency_ms=ms)

    monkeypatch.setattr(tune_order, "execute", fake_execute)


def test_measure_orders_runs_search_first_per_beta(planted):
    measured = {r["qid"]: r for r in measure_orders([TIGHT, LOOSE], sources(), CONFIG, betas=(1.0, 2.0), repeats=1)}
    tight = measured["tight"]
    assert (tight["total_notes"], tight["matching_notes"]) == (1000, 40)
    # Candidates: ceil(beta * 5 * 1000 / 40) = 125 and 250.
    assert tight["search_first"][1.0]["candidates"] == 125 and tight["search_first"][2.0]["candidates"] == 250
    assert tight["search_first"][2.0]["latency_ms"] == pytest.approx(3.5)
    assert tight["filter_first"]["latency_ms"] == 2.0
    assert set(tight["search_first"][1.0]) == {"recall", "latency_ms", "candidates"}


def test_tune_writes_config_and_report(planted, tmp_path):
    base = replace(CONFIG, s=0.3, beta=1.0)
    tuned, report = tune_order.tune([TIGHT, LOOSE], sources(), base, repeats=1)
    assert (tuned.s, tuned.beta) == (report["winner"]["s"], report["winner"]["beta"])
    assert len(report["grid"]) == len(tune_order.GRID_S) * len(tune_order.GRID_BETA) == 21
    assert report["default"]["s"] == 0.1 and report["default"]["beta"] == 1.0
    save_planner_config(tuned, tmp_path / "cfg.yaml")
    assert load_planner_config(tmp_path / "cfg.yaml") == tuned
    md = tune_order.write_report(report, tmp_path).read_text(encoding="utf-8")
    assert md.startswith("# Order rule tuning") and "Winner" in md


def test_tuner_refuses_the_test_split():
    with pytest.raises(SystemExit):
        tune_order.main(["--split", "test"])
