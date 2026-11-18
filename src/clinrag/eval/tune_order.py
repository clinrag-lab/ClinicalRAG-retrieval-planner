"""Tune the order rule's c and beta on class C/D dev questions.

Usage:
    python -m clinrag.eval.tune_order --split dev --config outputs/calibrated_config.yaml

For every class C and D dev question, filter_first is executed once and
search_first once per beta in the grid (beta changes how many candidates
search-first fetches, and so its recall). Latency is the median of --repeats.
Per question and beta, the better order is the one with higher recall, ties
broken by lower measured latency.

Each (c, beta) in the grid is scored by the fraction of questions where the
rule (planner/order.py) picks the better order; ties are broken by the mean
recall, then the mean latency, of the orders the rule picks. The winner's c and
beta are written into the config (default outputs/calibrated_config.yaml); the
full grid goes to outputs/tune_order_report.json and .md. Only the dev split is
accepted.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from clinrag.config import PlannerConfig, load_planner_config, save_planner_config
from clinrag.eval.analysis import _variant, better_order
from clinrag.eval.metrics import retrieval_score
from clinrag.planner.execute import execute
from clinrag.planner.order import candidates_to_fetch, choose_order
from clinrag.planner.planner import CostPlanner
from clinrag.sources.base import Sources
from clinrag.workload.schema import WorkloadQuestion, read_jsonl

GRID_C: tuple[float, ...] = (0.02, 0.05, 0.1, 0.25, 0.5, 1.0)
GRID_BETA: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0)


def measure_orders(
    questions: Sequence[WorkloadQuestion],
    sources: Sources,
    config: PlannerConfig,
    betas: Sequence[float] = GRID_BETA,
    repeats: int = 3,
) -> list[dict[str, Any]]:
    """Recall and median latency of filter_first, and of search_first at each beta, per C/D question."""
    planner = CostPlanner(sources, config)
    rows = []
    for q in questions:
        if q.spec.qclass not in ("C", "D"):
            continue
        decision = planner.plan(q.spec)
        ff, sf = _variant(decision, "filter_first"), _variant(decision, "search_first")
        if ff is None or sf is None:
            continue
        total, matching = decision.probes.total_notes, decision.probes.matching_notes
        assert total is not None and matching is not None

        def run(plan):
            results = [execute(plan, sources, q.spec, config) for _ in range(repeats)]
            return {
                "latency_ms": statistics.median(r.total_latency_ms for r in results),
                "recall": retrieval_score(q, results[0].items, config.top_k).value,
            }

        search_first = {}
        for beta in betas:
            n = candidates_to_fetch(config.top_k, total, matching, beta)
            search_first[beta] = {**run(sf.plan.with_candidates(n)), "candidates": n}
        rows.append({
            "qid": q.spec.qid,
            "qclass": q.spec.qclass,
            "selectivity": q.gold.get("selectivity"),
            "total_notes": total,
            "matching_notes": matching,
            "filter_first": run(ff.plan),
            "search_first": search_first,
        })
    return rows


def score_grid(
    rows: Sequence[dict[str, Any]], k: int, cs: Sequence[float] = GRID_C, betas: Sequence[float] = GRID_BETA
) -> list[dict[str, Any]]:
    """One row per (c, beta): rule agreement with the better order, and recall / latency of its picks."""
    grid = []
    for beta in betas:
        for c in cs:
            agree, recall, latency, ff_picks = [], [], [], []
            for row in rows:
                measured = {"filter_first": row["filter_first"], "search_first": row["search_first"][beta]}
                pick = choose_order(k, row["total_notes"], row["matching_notes"], c=c, beta=beta).order
                agree.append(pick == better_order(measured))
                recall.append(measured[pick]["recall"])
                latency.append(measured[pick]["latency_ms"])
                ff_picks.append(pick == "filter_first")
            grid.append({
                "c": c,
                "beta": beta,
                "n": len(rows),
                "agreement": round(float(np.mean(agree)), 4) if rows else 0.0,
                "mean_recall": round(float(np.mean(recall)), 4) if rows else 0.0,
                "mean_latency_ms": round(float(np.mean(latency)), 4) if rows else 0.0,
                "picks_filter_first": round(float(np.mean(ff_picks)), 4) if rows else 0.0,
            })
    return grid


def pick_winner(grid: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Highest agreement; ties by higher mean recall, then lower mean latency (then grid order)."""
    return min(
        enumerate(grid), key=lambda pair: (-pair[1]["agreement"], -pair[1]["mean_recall"], pair[1]["mean_latency_ms"], pair[0])
    )[1]


def tune(
    questions: Sequence[WorkloadQuestion], sources: Sources, config: PlannerConfig, repeats: int = 3
) -> tuple[PlannerConfig, dict[str, Any]]:
    rows = measure_orders(questions, sources, config, GRID_BETA, repeats)
    grid = score_grid(rows, config.top_k)
    winner = pick_winner(grid)
    tuned = replace(config, c=winner["c"], beta=winner["beta"])
    default = next(g for g in grid if g["c"] == PlannerConfig().c and g["beta"] == PlannerConfig().beta)
    report = {"winner": winner, "default": default, "n_questions": len(rows), "grid": grid, "questions": rows}
    return tuned, report


def write_report(report: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tune_order_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    w, d = report["winner"], report["default"]
    lines = [
        "# Order rule tuning (dev, class C and D)",
        "",
        f"Questions: {report['n_questions']}. Better order: higher recall, ties to lower latency.",
        "",
        f"Winner: c = {w['c']}, beta = {w['beta']}: rule picks the better order on {w['agreement']:.1%} "
        f"(mean recall {w['mean_recall']:.3f}, mean latency {w['mean_latency_ms']:.3f} ms).",
        f"Default (c = {d['c']}, beta = {d['beta']}): {d['agreement']:.1%}.",
        "",
        "| beta | c | agreement | mean_recall | mean_latency_ms | picks_filter_first |",
        "|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {g['beta']} | {g['c']} | {g['agreement']:.3f} | {g['mean_recall']:.3f} | {g['mean_latency_ms']:.3f} | "
        f"{g['picks_filter_first']:.3f} |"
        for g in report["grid"]
    ]
    path = out_dir / "tune_order_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> PlannerConfig:
    from clinrag.eval.runner import build_sources

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", choices=["fake", "reference"], default="reference")
    parser.add_argument("--tables-dir", type=Path, default=None)
    parser.add_argument("--notes-dir", type=Path, default=None)
    parser.add_argument("--workload", type=Path, default=Path("data/workload"))
    parser.add_argument("--split", choices=["dev"], default="dev", help="dev only: the test split is never used for tuning")
    parser.add_argument("--config", type=Path, default=Path("outputs/calibrated_config.yaml"),
                        help="base config, updated in place with the tuned c and beta")
    parser.add_argument("--out-config", type=Path, default=None, help="where to write the tuned config (default: --config)")
    parser.add_argument("--report-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)

    base = load_planner_config(args.config) if args.config.exists() else PlannerConfig()
    sources = build_sources(args.sources, args.tables_dir, args.notes_dir)
    questions = read_jsonl(args.workload / f"{args.split}.jsonl")
    tuned, report = tune(questions, sources, base, args.repeats)
    out_config = args.out_config or args.config
    save_planner_config(tuned, out_config, "Planner constants: calibrated timings, order rule c and beta tuned on dev.")
    md = write_report(report, args.report_dir)
    print(md.read_text(encoding="utf-8"))
    print(f"config: {out_config}  ({json.dumps({'c': tuned.c, 'beta': tuned.beta})})")
    return tuned


if __name__ == "__main__":
    main()
