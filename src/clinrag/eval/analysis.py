"""Analyses run at the end of the experiment runner, and the figures.

cost_model_check  every candidate plan of every question executed once; Spearman
                  rank correlation between estimated and measured cost, overall and
                  per class (also for latency and tokens alone), and how often the
                  estimated-cheapest plan is the measured-cheapest.
order_check       (RQ2) every class C and D question run both filter_first and
                  search_first; which was faster (median of repeats), the recall of
                  each, and how often the order rule's pick and the cost model's
                  pick match the faster order, binned by selectivity.
overhead_share    planning_overhead_ms / (planning_overhead_ms + retrieval latency)
                  per router, from the runner's records.
tau_sweep         P2 TF-IDF at tau = 0.1 .. 0.9: missed-source rate (a needed source
                  absent from the chosen plan) and mean tokens; by default on the run's
                  questions rendered in held-out phrasing families.

Figures (written to <out>/figures/): fig1_tokens_vs_recall.png,
fig2_order_vs_selectivity.png, fig3_tau_sweep.png, fig4_overhead.png.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from clinrag.config import PlannerConfig
from clinrag.eval.metrics import retrieval_score
from clinrag.planner.classifier import SourceClassifier
from clinrag.planner.execute import execute
from clinrag.planner.fallback import run_with_fallback
from clinrag.planner.planner import CostPlanner, PlanDecision, ScoredPlan
from clinrag.planner.router import ClassifierCostRouter
from clinrag.sources.base import Sources
from clinrag.workload.paraphrase import paraphrase_all, split_families
from clinrag.workload.schema import WorkloadQuestion, read_jsonl

SELECTIVITY_BINS: tuple[tuple[float, float], ...] = (
    (0.0, 0.05), (0.05, 0.10), (0.10, 0.25), (0.25, 0.50), (0.50, 0.75), (0.75, 1.0001),
)
TAUS: tuple[float, ...] = tuple(round(0.1 * i, 1) for i in range(1, 10))


def _rho(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    value = spearmanr(x, y).statistic
    return None if value is None or math.isnan(value) else round(float(value), 4)


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# ---- cost model check ---------------------------------------------------------------------


def cost_model_check(
    questions: Sequence[WorkloadQuestion], sources: Sources, config: PlannerConfig
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Execute every candidate plan of every question once; correlate estimated and measured cost."""
    planner = CostPlanner(sources, config)
    rows: list[dict[str, Any]] = []
    for q in questions:
        decision = planner.plan(q.spec)
        for scored in decision.candidates:
            result = execute(scored.plan, sources, q.spec, config)
            assert scored.cost is not None
            rows.append({
                "qid": q.spec.qid,
                "qclass": q.spec.qclass,
                "plan": scored.plan.name,
                "pruned": not scored.feasible,
                "est_cost": scored.cost.total,
                "est_latency_ms": scored.cost.est_latency_ms,
                "est_tokens": scored.cost.est_tokens,
                "measured_latency_ms": result.total_latency_ms,
                "measured_tokens": result.tokens,
                "measured_cost": result.total_latency_ms + config.alpha * result.tokens,
            })

    def stats(group: Sequence[dict[str, Any]]) -> dict[str, Any]:
        by_q: dict[str, list[dict[str, Any]]] = {}
        for r in group:
            by_q.setdefault(r["qid"], []).append(r)
        multi = [rs for rs in by_q.values() if len(rs) > 1]
        hits = sum(
            min(rs, key=lambda r: r["est_cost"])["plan"] == min(rs, key=lambda r: r["measured_cost"])["plan"]
            for rs in multi
        )
        return {
            "n_plans": len(group),
            "n_questions": len(by_q),
            "spearman_cost": _rho([r["est_cost"] for r in group], [r["measured_cost"] for r in group]),
            "spearman_latency": _rho([r["est_latency_ms"] for r in group], [r["measured_latency_ms"] for r in group]),
            "spearman_tokens": _rho([r["est_tokens"] for r in group], [r["measured_tokens"] for r in group]),
            "cheapest_plan_agreement": round(hits / len(multi), 4) if multi else None,
            "questions_with_choice": len(multi),
        }

    summary = {"overall": stats(rows)}
    for qclass in sorted({r["qclass"] for r in rows}):
        summary[qclass] = stats([r for r in rows if r["qclass"] == qclass])
    return summary, rows


# ---- order check (RQ2) --------------------------------------------------------------------


def _variant(decision: PlanDecision, mode: str) -> ScoredPlan | None:
    for scored in decision.candidates:
        search = scored.plan.note_search
        if scored.plan.is_mixed and search is not None and search.mode == mode:
            return scored
    return None


def order_check(
    questions: Sequence[WorkloadQuestion], sources: Sources, config: PlannerConfig, repeats: int = 3
) -> list[dict[str, Any]]:
    """Run filter_first and search_first on every question with a filtered note search."""
    planner = CostPlanner(sources, config)
    rows = []
    for q in questions:
        if q.spec.qclass not in ("C", "D"):
            continue
        decision = planner.plan(q.spec)
        ff, sf = _variant(decision, "filter_first"), _variant(decision, "search_first")
        if ff is None or sf is None or decision.order is None:
            continue
        measured = {}
        for mode, scored in (("filter_first", ff), ("search_first", sf)):
            runs = [execute(scored.plan, sources, q.spec, config) for _ in range(repeats)]
            measured[mode] = {
                "latency_ms": statistics.median(r.total_latency_ms for r in runs),
                "tokens": runs[0].tokens,
                "recall": retrieval_score(q, runs[0].items, config.top_k).value,
                "est_cost": scored.cost.total if scored.cost else None,
            }
        faster = min(measured, key=lambda m: measured[m]["latency_ms"])
        rows.append({
            "qid": q.spec.qid,
            "qclass": q.spec.qclass,
            "selectivity": q.gold.get("selectivity"),
            "matching_notes": decision.probes.matching_notes,
            "total_notes": decision.probes.total_notes,
            "filter_first_ms": measured["filter_first"]["latency_ms"],
            "search_first_ms": measured["search_first"]["latency_ms"],
            "filter_first_recall": measured["filter_first"]["recall"],
            "search_first_recall": measured["search_first"]["recall"],
            "faster": faster,
            "rule_pick": decision.order.order,
            "cost_pick": decision.cost_order,
            "rule_matches_faster": decision.order.order == faster,
            "cost_matches_faster": decision.cost_order == faster,
        })
    return rows


def order_table(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Order-check rows binned by selectivity, plus an "all" row."""
    table = []
    bins = [(f"{lo:.2f}-{min(hi, 1.0):.2f}", lo, hi) for lo, hi in SELECTIVITY_BINS] + [("all", -1.0, 2.0)]
    for label, lo, hi in bins:
        group = [r for r in rows if r["selectivity"] is not None and lo <= r["selectivity"] < hi]
        if not group:
            continue
        table.append({
            "selectivity": label,
            "n": len(group),
            "filter_first_faster": round(sum(r["faster"] == "filter_first" for r in group) / len(group), 3),
            "rule_matches_faster": round(sum(r["rule_matches_faster"] for r in group) / len(group), 3),
            "cost_matches_faster": round(sum(r["cost_matches_faster"] for r in group) / len(group), 3),
            "rule_picks_filter_first": round(sum(r["rule_pick"] == "filter_first" for r in group) / len(group), 3),
            "mean_filter_first_ms": round(float(np.mean([r["filter_first_ms"] for r in group])), 3),
            "mean_search_first_ms": round(float(np.mean([r["search_first_ms"] for r in group])), 3),
            "mean_filter_first_recall": round(float(np.mean([r["filter_first_recall"] for r in group])), 3),
            "mean_search_first_recall": round(float(np.mean([r["search_first_recall"] for r in group])), 3),
        })
    return table


# ---- overhead share -------------------------------------------------------------------------


def overhead_share(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for router in dict.fromkeys(r["router"] for r in records):
        mine = [r for r in records if r["router"] == router]
        overhead = [r["planning_overhead_ms"] for r in mine]
        latency = [r["latency_ms"] for r in mine]
        shares = [o / (o + l) if o + l > 0 else 0.0 for o, l in zip(overhead, latency)]
        rows.append({
            "router": router,
            "n": len(mine),
            "mean_overhead_ms": round(float(np.mean(overhead)), 4),
            "mean_retrieval_ms": round(float(np.mean(latency)), 4),
            "mean_overhead_share": round(float(np.mean(shares)), 4),
            "overhead_share_of_total_time": round(sum(overhead) / (sum(overhead) + sum(latency)), 4),
        })
    return rows


# ---- tau sweep ------------------------------------------------------------------------------


def tau_sweep(
    questions: Sequence[WorkloadQuestion],
    sources: Sources,
    config: PlannerConfig,
    workload_dir: Path,
    phrasing: str = "paraphrased",
    seed: int = 0,
) -> list[dict[str, Any]]:
    """P2 TF-IDF at each tau. phrasing "paraphrased": the classifier trains on dev plus
    train-family paraphrases and is run on the questions rendered in held-out test
    families; "original": trains on dev and runs on the questions as written."""
    dev = read_jsonl(workload_dir / "dev.jsonl")
    if phrasing == "paraphrased":
        split = split_families(seed)
        train = [q.spec for q in dev] + [p.question.spec for p in paraphrase_all(dev, split["train"], seed)]
        eval_questions = [p.question for p in paraphrase_all(questions, split["test"], seed)]
    else:
        train = [q.spec for q in dev]
        eval_questions = list(questions)
    clf = SourceClassifier("tfidf").fit(train)
    rows = []
    for tau in TAUS:
        router = ClassifierCostRouter(clf, sources, config, tau=tau, name="p2_tfidf")
        tokens, missed, recall = [], [], []
        for q in eval_questions:
            decision = router.decide(q.spec)
            run = run_with_fallback(decision, sources, q.spec, config)
            tokens.append(run.tokens)
            missed.append(bool(q.spec.required_sources - decision.plan.sources))
            recall.append(retrieval_score(q, run.items, config.top_k).value)
        rows.append({
            "tau": tau,
            "n": len(eval_questions),
            "missed_source_rate": round(float(np.mean(missed)), 4),
            "mean_tokens": round(float(np.mean(tokens)), 2),
            "mean_recall": round(float(np.mean(recall)), 4),
        })
    return rows


# ---- figures --------------------------------------------------------------------------------


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _se(values: Sequence[float]) -> float:
    return float(np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0


def fig_tokens_vs_recall(records: Sequence[dict[str, Any]], path: Path) -> None:
    plt = _plt()
    routers = list(dict.fromkeys(r["router"] for r in records))
    stats = {}
    for router in routers:
        mine = [r for r in records if r["router"] == router]
        tok, rec = [r["tokens"] for r in mine], [r["recall"] for r in mine]
        stats[router] = (np.mean(tok), _se(tok), np.mean(rec), _se(rec))
    best = max(routers, key=lambda r: stats[r][2])
    cheapest = min(routers, key=lambda r: stats[r][0])
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for i, router in enumerate(routers):
        mt, st, mr, sr = stats[router]
        ax.errorbar(mt, mr, xerr=st, yerr=sr, fmt="o", capsize=4, color=f"C{i}", label=router)
    ax.set_xlabel("Mean evidence tokens per question (tokens, ±1 SE)")
    ax.set_ylabel("Mean retrieval recall (0 to 1, ±1 SE)")
    ax.set_title(f"{best} reaches the highest recall; {cheapest} sends the fewest tokens", fontsize=10)
    ax.legend(title="router", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_order_vs_selectivity(rows: Sequence[dict[str, Any]], config: PlannerConfig, path: Path) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.5, 5))
    xs = [r["selectivity"] for r in rows]
    ax.scatter(xs, [r["filter_first_ms"] for r in rows], label="filter_first", marker="o", alpha=0.7)
    ax.scatter(xs, [r["search_first_ms"] for r in rows], label="search_first", marker="^", alpha=0.7)
    switch = config.top_k / config.max_candidates
    ax.axvline(switch, color="gray", linestyle="--", label=f"rule switch (k / max_candidates = {switch:.3g})")
    n_ff = sum(r["faster"] == "filter_first" for r in rows)
    ax.set_xlabel("Filter selectivity (fraction of admissions passing the filter)")
    ax.set_ylabel("Measured retrieval latency (ms, median of repeats)")
    ax.set_title(
        f"filter_first is faster on {n_ff} of {len(rows)} C/D questions; the rule matches the faster order "
        f"on {sum(r['rule_matches_faster'] for r in rows)}",
        fontsize=9,
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_tau_sweep(rows: Sequence[dict[str, Any]], path: Path) -> None:
    plt = _plt()
    taus = [r["tau"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    l1 = ax.plot(taus, [r["missed_source_rate"] for r in rows], "o-", color="C3", label="missed-source rate")
    ax.set_xlabel("Classifier threshold tau")
    ax.set_ylabel("Missed-source rate (fraction of questions)")
    ax2 = ax.twinx()
    l2 = ax2.plot(taus, [r["mean_tokens"] for r in rows], "s--", color="C0", label="mean tokens")
    ax2.set_ylabel("Mean evidence tokens per question (tokens)")
    first, last = rows[0], rows[-1]
    ax.set_title(
        f"P2 TF-IDF: tau {first['tau']} to {last['tau']} moves missed sources "
        f"{first['missed_source_rate']:.0%} to {last['missed_source_rate']:.0%} and tokens "
        f"{first['mean_tokens']:.0f} to {last['mean_tokens']:.0f}",
        fontsize=9,
    )
    ax.legend(l1 + l2, [line.get_label() for line in l1 + l2], fontsize=8, loc="upper center")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_overhead(rows: Sequence[dict[str, Any]], path: Path) -> None:
    plt = _plt()
    routers = [r["router"] for r in rows]
    overhead = [r["mean_overhead_ms"] for r in rows]
    retrieval = [r["mean_retrieval_ms"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(routers, overhead, label="planning overhead")
    ax.bar(routers, retrieval, bottom=overhead, label="retrieval (incl. fallbacks)")
    shares = [r["overhead_share_of_total_time"] for r in rows]
    ax.set_xlabel("Router")
    ax.set_ylabel("Mean time per question (ms)")
    ax.set_title(f"Planning overhead is {min(shares):.0%} to {max(shares):.0%} of total time across routers", fontsize=10)
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---- all ------------------------------------------------------------------------------------


def run_analyses(
    records: Sequence[dict[str, Any]],
    questions: Sequence[WorkloadQuestion],
    sources: Sources,
    config: PlannerConfig,
    workload_dir: Path,
    out: Path,
    repeats: int = 3,
    sweep_phrasing: str = "paraphrased",
) -> dict[str, Any]:
    """Every analysis and figure; writes <out>/analysis/ and <out>/figures/."""
    adir, fdir = out / "analysis", out / "figures"
    adir.mkdir(parents=True, exist_ok=True)
    fdir.mkdir(parents=True, exist_ok=True)

    cost_summary, cost_rows = cost_model_check(questions, sources, config)
    _write_csv(adir / "cost_model_plans.csv", cost_rows)
    order_rows = order_check(questions, sources, config, repeats)
    _write_csv(adir / "order_check.csv", order_rows)
    order_bins = order_table(order_rows)
    _write_csv(adir / "order_check_by_selectivity.csv", order_bins)
    overhead = overhead_share(records)
    _write_csv(adir / "overhead_share.csv", overhead)
    sweep = [] if sweep_phrasing == "none" else tau_sweep(questions, sources, config, workload_dir, sweep_phrasing)
    _write_csv(adir / "tau_sweep.csv", sweep)

    fig_tokens_vs_recall(records, fdir / "fig1_tokens_vs_recall.png")
    if order_rows:
        fig_order_vs_selectivity(order_rows, config, fdir / "fig2_order_vs_selectivity.png")
    if sweep:
        fig_tau_sweep(sweep, fdir / "fig3_tau_sweep.png")
    fig_overhead(overhead, fdir / "fig4_overhead.png")

    summary = {
        "cost_model": cost_summary,
        "order_check": order_bins,
        "overhead_share": overhead,
        "tau_sweep": {"phrasing": sweep_phrasing, "rows": sweep},
    }
    (adir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    from clinrag.eval.runner import markdown_table

    cost_table = [{"group": g, **v} for g, v in cost_summary.items()]
    md = [
        "# Analyses",
        "",
        "## Cost model: estimated vs measured (Spearman, every candidate plan executed once)",
        "",
        markdown_table(cost_table, list(cost_table[0])),
        "",
        "## Order check (RQ2): filter_first vs search_first on class C and D, by selectivity",
        "",
        markdown_table(order_bins, list(order_bins[0])) if order_bins else "(no class C or D questions)",
        "",
        "## Planning overhead share",
        "",
        markdown_table(overhead, list(overhead[0])),
        "",
        f"## P2 TF-IDF tau sweep ({sweep_phrasing} phrasings)",
        "",
        markdown_table(sweep, list(sweep[0])) if sweep else "(skipped)",
        "",
    ]
    (adir / "analysis.md").write_text("\n".join(md), encoding="utf-8")
    return summary
