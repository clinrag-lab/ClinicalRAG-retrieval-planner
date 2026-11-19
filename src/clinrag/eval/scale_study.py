"""Order check (RQ2) at several note-index sizes.

Usage:
    python -m clinrag.workload.notes_gen --tables-dir data/mimic-iv-demo --out-dir data/generated --scale 10
    python -m clinrag.eval.scale_study --split test --scales 1 10 50 \\
        --config outputs/calibrated_config.yaml --out outputs/runs/<run>

For each scale S the notes come from <notes-root> (S = 1) or
<notes-root>/scale_S/. The cost constants are re-calibrated on the scaled
sources with the dev split's queries (--config supplies the base, so the
tuned s and beta are kept), then every class C and D question of --split
runs both filter_first and search_first (clinrag.eval.analysis.order_check).

Writes to --out:
    analysis/order_check_scale_S.csv, analysis/order_scale_summary.json and .md
    figures/fig2_order_vs_selectivity.png  one panel per scale: latency of both
                                           orders vs selectivity, rule switch marked
    figures/fig5_scale.png                 notes in the index vs the share of
                                           questions where filter_first is the
                                           better order, and the rule's agreement
    analysis/calibrated_config_scale_S.yaml  the config used at each scale
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from clinrag.config import PlannerConfig, load_planner_config, save_planner_config
from clinrag.eval.analysis import _plt, _write_csv, order_check, order_table
from clinrag.eval.calibrate import calibrate, load_concept_tags
from clinrag.sources.reference import build_reference_sources
from clinrag.workload.schema import read_jsonl


def notes_dir_for(root: Path, scale: int) -> Path:
    return root if scale == 1 else root / f"scale_{scale}"


def scale_summary(scale: int, rows: Sequence[dict[str, Any]], config: PlannerConfig) -> dict[str, Any]:
    total = int(statistics.median(r["total_notes"] for r in rows)) if rows else 0
    n = len(rows) or 1
    return {
        "scale": scale,
        "total_notes": total,
        "n": len(rows),
        "s": config.s,
        "beta": config.beta,
        "switch_selectivity": config.s,
        "filter_first_faster": sum(r["faster"] == "filter_first" for r in rows) / n,
        "filter_first_better": sum(r["better"] == "filter_first" for r in rows) / n,
        "rule_picks_filter_first": sum(r["rule_pick"] == "filter_first" for r in rows) / n,
        "rule_matches_better": sum(r["rule_matches_better"] for r in rows) / n,
        "rule_matches_faster": sum(r["rule_matches_faster"] for r in rows) / n,
        "cost_matches_better": sum(r["cost_matches_better"] for r in rows) / n,
        "mean_filter_first_ms": float(np.mean([r["filter_first_ms"] for r in rows])) if rows else None,
        "mean_search_first_ms": float(np.mean([r["search_first_ms"] for r in rows])) if rows else None,
        "mean_filter_first_recall": float(np.mean([r["filter_first_recall"] for r in rows])) if rows else None,
        "mean_search_first_recall": float(np.mean([r["search_first_recall"] for r in rows])) if rows else None,
    }


def fig_order_by_scale(results: Sequence[tuple[dict[str, Any], Sequence[dict[str, Any]]]], path: Path) -> None:
    plt = _plt()
    fig, axes = plt.subplots(1, len(results), figsize=(5.2 * len(results), 4.6), squeeze=False)
    for ax, (summary, rows) in zip(axes[0], results):
        ordered = sorted(rows, key=lambda r: r["selectivity"])
        xs = [r["selectivity"] for r in ordered]
        ax.plot(xs, [r["filter_first_ms"] for r in ordered], "o", alpha=0.6, label="filter_first")
        ax.plot(xs, [r["search_first_ms"] for r in ordered], "^", alpha=0.6, label="search_first")
        if summary["switch_selectivity"] is not None:
            ax.axvline(summary["switch_selectivity"], color="gray", linestyle="--",
                       label=f"rule switch (selectivity {summary['switch_selectivity']:.3g})")
        ax.set_xscale("log")
        ax.set_xlabel("Filter selectivity (fraction of admissions, log scale)")
        ax.set_ylabel("Measured retrieval latency (ms, median of repeats)")
        ax.set_title(f"{summary['total_notes']:,} notes: filter_first faster on {summary['filter_first_faster']:.0%}",
                     fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    first, last = results[0][0], results[-1][0]
    fig.suptitle(
        f"As the index grows from {first['total_notes']:,} to {last['total_notes']:,} notes, filter_first is faster on "
        f"{first['filter_first_faster']:.0%} to {last['filter_first_faster']:.0%} of C/D questions",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_scale(summaries: Sequence[dict[str, Any]], path: Path) -> None:
    plt = _plt()
    xs = [s["total_notes"] for s in summaries]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(xs, [s["filter_first_better"] for s in summaries], "o-", label="filter_first is the better order")
    ax.plot(xs, [s["rule_matches_better"] for s in summaries], "s--", label="rule picks the better order")
    ax.plot(xs, [s["cost_matches_better"] for s in summaries], "^:", label="cost model picks the better order")
    ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Notes in the index (log scale)")
    ax.set_ylabel("Fraction of class C/D questions")
    last = summaries[-1]
    ax.set_title(
        f"At {last['total_notes']:,} notes filter_first is better on {last['filter_first_better']:.0%}; "
        f"the rule agrees with the better order on {last['rule_matches_better']:.0%}",
        fontsize=9,
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> list[dict[str, Any]]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tables-dir", type=Path, default=Path("data/mimic-iv-demo"))
    parser.add_argument("--notes-root", type=Path, default=Path("data/generated"))
    parser.add_argument("--workload", type=Path, default=Path("data/workload"))
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--scales", type=int, nargs="+", default=[1, 10, 50])
    parser.add_argument("--config", type=Path, default=Path("outputs/calibrated_config.yaml"),
                        help="base config (tuned s and beta are kept; timings are re-calibrated per scale)")
    parser.add_argument("--no-calibrate", action="store_true", help="use --config as is at every scale")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    base = load_planner_config(args.config) if args.config.exists() else PlannerConfig()
    questions = read_jsonl(args.workload / f"{args.split}.jsonl")
    dev = read_jsonl(args.workload / "dev.jsonl")
    adir, fdir = args.out / "analysis", args.out / "figures"
    adir.mkdir(parents=True, exist_ok=True)
    fdir.mkdir(parents=True, exist_ok=True)

    results = []
    for scale in args.scales:
        notes_dir = notes_dir_for(args.notes_root, scale)
        sources = build_reference_sources(args.tables_dir, notes_dir)
        config = base
        if not args.no_calibrate:
            config, _ = calibrate(sources, dev, base=base, repeats=args.repeats, concept_tags=load_concept_tags(notes_dir))
            save_planner_config(config, adir / f"calibrated_config_scale_{scale}.yaml",
                                f"Calibrated on notes scale {scale}; s and beta from {args.config}.")
        rows = order_check(questions, sources, config, args.repeats)
        _write_csv(adir / f"order_check_scale_{scale}.csv", rows)
        _write_csv(adir / f"order_check_by_selectivity_scale_{scale}.csv", order_table(rows))
        summary = scale_summary(scale, rows, config)
        print(f"scale {scale}: {summary['total_notes']} notes, {summary['n']} questions, "
              f"filter_first better {summary['filter_first_better']:.2f}, rule agrees {summary['rule_matches_better']:.2f}")
        results.append((summary, rows))

    summaries = [s for s, _ in results]
    fig_order_by_scale(results, fdir / "fig2_order_vs_selectivity.png")
    fig_scale(summaries, fdir / "fig5_scale.png")
    (adir / "order_scale_summary.json").write_text(
        json.dumps({"split": args.split, "summaries": summaries,
                    "by_selectivity": {s["scale"]: order_table(rows) for s, rows in results}}, indent=2) + "\n",
        encoding="utf-8",
    )
    from clinrag.eval.runner import markdown_table

    md = ["# Order check by index size", "", markdown_table(summaries, list(summaries[0])), ""]
    for summary, rows in results:
        table = order_table(rows)
        md += [f"## {summary['total_notes']:,} notes (scale {summary['scale']})", "",
               markdown_table(table, list(table[0])) if table else "(no rows)", ""]
    (adir / "order_scale_summary.md").write_text("\n".join(md), encoding="utf-8")
    return summaries


if __name__ == "__main__":
    main()
