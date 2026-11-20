"""Helpers for the v2 accuracy job on Sol (scripts/sol/run_v2.sbatch, scripts/sol/prepare_data.sh).

Usage:
    python -m clinrag.eval.sol_v2 counts data/workload_v2/test.jsonl data/workload_v2/paraphrase/test.jsonl
    python -m clinrag.eval.sol_v2 complete --part DIR --router p1 --questions data/workload_v2/test.jsonl \\
        --answerer llama --llm-dtype bf16
    python -m clinrag.eval.sol_v2 summary --run outputs/runs/v2
    python -m clinrag.eval.sol_v2 fig8 --run outputs/runs/v2 --out docs/figures/fig8_accuracy_vs_tokens.png

counts    question count, per-class counts and SHA-256 of each question file, to compare two machines.
complete  exit 0 if DIR holds a finished one-router run: results.csv written (the runner writes it
          after the last row), exactly one row per question for the router, and, when given, the
          answerer and LLM dtype the run used. Otherwise print why and exit 1. The job skips
          complete routers, so a timed-out job can be resubmitted.
summary   merge the template-test parts into <run>/template-test/results.{jsonl,csv,md}, read the
          paraphrase-test rows merged by `clinrag.eval.freeze --merge`, and write <run>/SUMMARY.md:
          recall, tokens, latency, overhead, answer accuracy, parse failure rate and truncated rate
          per router and class.
fig8      answer accuracy vs mean evidence tokens per router, one panel for the template test split
          and one for the paraphrase test split, from the "all" row of each parts/<router>/results.csv.

Layout under --run (written by run_v2.sbatch):
    template-test/parts/<router>/     one runner run per router on the template test split
    paraphrase-test/parts/<router>/   one runner run per router on the paraphrase test split
    paraphrase-test/models/           the P2 model (dev + train-family paraphrases)
    paraphrase-test/results.jsonl     freeze --merge of the paraphrase parts
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from clinrag.eval.runner import CLASSES, markdown_table, summarize, write_summaries
from clinrag.workload.schema import read_jsonl

TEMPLATE_ROUTERS: tuple[str, ...] = (
    "oracle", "p1", "p2_tfidf", "query_all", "always_filter_first", "always_search_first",
)
PARAPHRASE_ROUTERS: tuple[str, ...] = ("p1", "p2_tfidf", "oracle", "query_all")
TEMPLATE_QUESTIONS = Path("data/workload_v2/test.jsonl")
PARAPHRASE_QUESTIONS = Path("data/workload_v2/paraphrase/test.jsonl")
COLUMNS: tuple[str, ...] = (
    "router", "qclass", "n", "mean_recall", "mean_tokens", "mean_latency_ms", "mean_overhead_ms",
    "answer_accuracy", "parse_failure_rate", "truncated_rate",
)


# ---- counts ------------------------------------------------------------------------------


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def counts(path: Path) -> dict[str, Any]:
    """Questions in a workload file, per class, and the file's SHA-256."""
    questions = read_jsonl(path)
    by_class = Counter(q.spec.qclass for q in questions)
    return {"path": str(path), "n": len(questions), "by_class": {c: by_class.get(c, 0) for c in CLASSES},
            "sha256": sha256(path)}


def format_counts(info: dict[str, Any]) -> str:
    per_class = " ".join(f"{c}={n}" for c, n in info["by_class"].items())
    return f"{info['path']}: {info['n']} questions ({per_class})  sha256 {info['sha256']}"


# ---- one router's part -------------------------------------------------------------------


def qids_of(questions: Path) -> list[str]:
    return [q.spec.qid for q in read_jsonl(questions)]


def part_rows(part: Path, router: str, qids: Sequence[str]) -> list[dict[str, Any]]:
    """The router's rows in part/results.jsonl, in question order. Raises ValueError unless there is
    exactly one row per question."""
    path = Path(part) / "results.jsonl"
    if not path.exists():
        raise ValueError(f"no {path}")
    with path.open(encoding="utf-8") as f:
        rows = [r for r in (json.loads(line) for line in f if line.strip()) if r["router"] == router]
    got = [r["qid"] for r in rows]
    if len(got) != len(set(got)) or set(got) != set(qids):
        raise ValueError(f"{router} in {part}: {len(set(got) & set(qids))} of {len(qids)} questions, "
                         f"{len(got)} rows")
    order = {qid: i for i, qid in enumerate(qids)}
    return sorted(rows, key=lambda r: order[r["qid"]])


def check_part(
    part: Path, router: str, questions: Path, answerer: str | None = None, llm_dtype: str | None = None
) -> str | None:
    """None if part holds a finished run of router over questions (see the module docstring), else why not."""
    part = Path(part)
    if not (part / "results.csv").exists():
        return f"no {part / 'results.csv'} (not started, or stopped before the end)"
    try:
        rows = part_rows(part, router, qids_of(questions))
    except ValueError as e:
        return str(e)
    if answerer is not None:
        used = sorted({str(r.get("answerer")) for r in rows})
        if used != [answerer]:
            return f"{router} in {part}: answerer {', '.join(used)}, expected {answerer}"
    if llm_dtype is not None:
        run_json = part / "run.json"
        dtype = json.loads(run_json.read_text(encoding="utf-8")).get("llm_dtype") if run_json.exists() else None
        if dtype != llm_dtype:
            return f"{router} in {part}: llm_dtype {dtype}, expected {llm_dtype}"
    return None


# ---- summary -----------------------------------------------------------------------------


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def merge_template(run: Path, questions: Path, routers: Sequence[str]) -> list[dict[str, Any]]:
    """Every router's rows from template-test/parts/<router>, written to template-test/results.*."""
    qids = qids_of(questions)
    records = [row for router in routers for row in part_rows(run / "template-test" / "parts" / router, router, qids)]
    into = run / "template-test"
    with (into / "results.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True, default=str) + "\n")
    write_summaries(into, records)
    return records


def read_merged(path: Path, questions: Path, routers: Sequence[str]) -> list[dict[str, Any]]:
    """Rows of a freeze --merge results.jsonl, checked to hold every router on every question."""
    qids = qids_of(questions)
    with Path(path).open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    records = []
    for router in routers:
        mine = [r for r in rows if r["router"] == router]
        got = [r["qid"] for r in mine]
        if len(got) != len(set(got)) or set(got) != set(qids):
            raise ValueError(f"{router} in {path}: {len(set(got) & set(qids))} of {len(qids)} questions, "
                             f"{len(got)} rows")
        records += mine
    return records


def _section(title: str, records: Sequence[dict[str, Any]], part_dir: Path, routers: Sequence[str],
             questions: Path) -> list[str]:
    rows = summarize(records)
    lines = [f"## {title}", "", f"Questions: `{questions}` ({len(qids_of(questions))}, sha256 `{sha256(questions)}`)",
             "", "### All classes", "", markdown_table([r for r in rows if r["qclass"] == "all"], COLUMNS),
             "", "### Per class", "", markdown_table([r for r in rows if r["qclass"] != "all"], COLUMNS),
             "", "### Commands", "", "```"]
    for router in routers:
        command = part_dir / router / "command.txt"
        lines.append(command.read_text(encoding="utf-8").strip() if command.exists() else f"# {router}: no command.txt")
    return [*lines, "```", ""]


def summary(
    run: Path,
    template_questions: Path = TEMPLATE_QUESTIONS,
    paraphrase_questions: Path = PARAPHRASE_QUESTIONS,
    template_routers: Sequence[str] = TEMPLATE_ROUTERS,
    paraphrase_routers: Sequence[str] = PARAPHRASE_ROUTERS,
) -> Path:
    run = Path(run)
    template = merge_template(run, template_questions, template_routers)
    paraphrase = read_merged(run / "paraphrase-test" / "results.jsonl", paraphrase_questions, paraphrase_routers)
    answerers = sorted({str(r.get("answerer")) for r in [*template, *paraphrase]})
    lines = [
        "# v2 accuracy run",
        "",
        f"- commit `{git_commit()}`, written {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        + (f", Slurm job {os.environ['SLURM_JOB_ID']}" if os.environ.get("SLURM_JOB_ID") else ""),
        f"- answerer: {', '.join(answerers)}",
        "- recall: value row (A), gold admission's note (per-admission B), normalized recall@k (B list, C, D)",
        "- tokens: evidence tokens retrieved; latency and overhead in ms (median over repeats, mean over questions)",
        "- answer_accuracy: mean answer score; a null or unparseable answer scores 0",
        "- parse_failure_rate: share of answers with no value of the expected type",
        "- truncated_rate: share of questions whose evidence was cut to max_context_tokens",
        "",
        *_section("Template test split", template, run / "template-test" / "parts", template_routers,
                  template_questions),
        *_section("Paraphrase test split (P2 trained on v2 dev + train-family paraphrases)", paraphrase,
                  run / "paraphrase-test" / "parts", paraphrase_routers, paraphrase_questions),
    ]
    path = run / "SUMMARY.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ---- fig8 --------------------------------------------------------------------------------

FIG8_PANELS: tuple[tuple[str, str], ...] = (
    ("template-test", "Template test"),
    ("paraphrase-test", "Paraphrase test (held-out phrasings)"),
)


def overall_points(part_dir: Path, routers: Sequence[str]) -> list[dict[str, Any]]:
    """(router, n, mean_tokens, answer_accuracy) from the "all" row of each <part_dir>/<router>/results.csv."""
    points = []
    for router in routers:
        path = Path(part_dir) / router / "results.csv"
        with path.open(newline="", encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if r["router"] == router and r["qclass"] == "all"]
        if len(rows) != 1:
            raise ValueError(f"{path}: expected one 'all' row for {router}, found {len(rows)}")
        row = rows[0]
        if row.get("answer_accuracy") in (None, ""):
            raise ValueError(f"{path}: no answer_accuracy; was the run made with an answerer?")
        points.append({"router": router, "n": int(row["n"]), "mean_tokens": float(row["mean_tokens"]),
                       "answer_accuracy": float(row["answer_accuracy"])})
    return points


def _label_groups(points: Sequence[dict[str, Any]], x_tol: float, y_tol: float) -> list[list[dict[str, Any]]]:
    """Routers whose points nearly coincide, so they share one label instead of overprinting."""
    groups: list[list[dict[str, Any]]] = []
    for p in points:
        for g in groups:
            if abs(g[0]["mean_tokens"] - p["mean_tokens"]) <= x_tol and abs(
                    g[0]["answer_accuracy"] - p["answer_accuracy"]) <= y_tol:
                g.append(p)
                break
        else:
            groups.append([p])
    return groups


def fig8_title(panels: Sequence[Sequence[dict[str, Any]]], planner: str = "p2_tfidf",
               baseline: str = "query_all") -> str:
    """Takeaway from the data: the planner's accuracy and token saving vs the baseline, per panel."""
    parts = []
    for points in panels:
        by = {p["router"]: p for p in points}
        if planner not in by or baseline not in by:
            continue
        pl, bl = by[planner], by[baseline]
        fewer = 1 - pl["mean_tokens"] / bl["mean_tokens"]
        parts.append(f"{pl['answer_accuracy']:.2f} vs {bl['answer_accuracy']:.2f} at {fewer:.0%} fewer tokens")
    verdict = "answers more questions correctly than" if all(
        {p["router"]: p for p in pts}[planner]["answer_accuracy"] > {p["router"]: p for p in pts}[baseline][
            "answer_accuracy"] for pts in panels if {planner, baseline} <= {p["router"] for p in pts}
    ) else "does not beat"
    return f"The {planner} planner {verdict} {baseline} with far less evidence\n({'; '.join(parts)})"


def fig8(run: Path, path: Path, template_routers: Sequence[str] = TEMPLATE_ROUTERS,
         paraphrase_routers: Sequence[str] = PARAPHRASE_ROUTERS) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run = Path(run)
    routers = {"template-test": template_routers, "paraphrase-test": paraphrase_routers}
    panels = [overall_points(run / split / "parts", routers[split]) for split, _ in FIG8_PANELS]
    all_routers = list(dict.fromkeys(p["router"] for pts in panels for p in pts))
    colors = {r: f"C{i}" for i, r in enumerate(all_routers)}
    x_max = max(p["mean_tokens"] for pts in panels for p in pts) * 1.15

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax, (split, label), points in zip(axes, FIG8_PANELS, panels):
        for p in points:
            ax.scatter(p["mean_tokens"], p["answer_accuracy"], s=60, color=colors[p["router"]], zorder=3)
        groups = _label_groups(points, x_tol=0.03 * x_max, y_tol=0.02)
        for group in groups:
            p = group[0]
            tokens = sorted(q["mean_tokens"] for q in group)
            accs = sorted(q["answer_accuracy"] for q in group)
            tok = f"{tokens[0]:,.0f}" if round(tokens[0]) == round(tokens[-1]) else f"{tokens[0]:,.0f}-{tokens[-1]:,.0f}"
            acc = f"{accs[0]:.2f}" if f"{accs[0]:.2f}" == f"{accs[-1]:.2f}" else f"{accs[0]:.2f}-{accs[-1]:.2f}"
            names = [q["router"] for q in group]
            name_lines = "\n".join(", ".join(names[i:i + 2]) for i in range(0, len(names), 2))
            text = f"{name_lines}\n{acc} accuracy, {tok} tokens"
            # A label goes below its point when another labeled point sits just above it, and to the
            # below-left of points in the right third, so labels neither overprint nor leave the axes.
            below = any(g is not group and abs(g[0]["mean_tokens"] - p["mean_tokens"]) < 0.2 * x_max
                        and 0 < g[0]["answer_accuracy"] - p["answer_accuracy"] < 0.12 for g in groups)
            left = p["mean_tokens"] > 0.66 * x_max
            below = below or left
            ax.annotate(text, (p["mean_tokens"], p["answer_accuracy"]),
                        xytext=(-8 if left else 8, -8 if below else 8), textcoords="offset points", fontsize=8,
                        ha="right" if left else "left", va="top" if below else "bottom")
        ax.set_title(f"{label}, {points[0]['n']} questions", fontsize=10)
        ax.set_xlabel("Mean evidence tokens per question (tokens)")
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Answer accuracy (mean answer score, 0 to 1)")
    fig.suptitle(fig8_title(panels), fontsize=10)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---- CLI ---------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p_counts = sub.add_parser("counts", help="question counts and SHA-256 of workload files")
    p_counts.add_argument("files", nargs="+", type=Path)
    p_complete = sub.add_parser("complete", help="exit 0 if a one-router run directory is finished")
    p_complete.add_argument("--part", type=Path, required=True)
    p_complete.add_argument("--router", required=True)
    p_complete.add_argument("--questions", type=Path, required=True)
    p_complete.add_argument("--answerer", default=None)
    p_complete.add_argument("--llm-dtype", default=None)
    p_summary = sub.add_parser("summary", help="write <run>/SUMMARY.md")
    p_summary.add_argument("--run", type=Path, default=Path("outputs/runs/v2"))
    p_summary.add_argument("--template-questions", type=Path, default=TEMPLATE_QUESTIONS)
    p_summary.add_argument("--paraphrase-questions", type=Path, default=PARAPHRASE_QUESTIONS)
    p_summary.add_argument("--template-routers", nargs="+", default=list(TEMPLATE_ROUTERS))
    p_summary.add_argument("--paraphrase-routers", nargs="+", default=list(PARAPHRASE_ROUTERS))
    p_fig8 = sub.add_parser("fig8", help="answer accuracy vs evidence tokens per router, template and paraphrase")
    p_fig8.add_argument("--run", type=Path, default=Path("outputs/runs/v2"))
    p_fig8.add_argument("--out", type=Path, default=Path("docs/figures/fig8_accuracy_vs_tokens.png"))
    p_fig8.add_argument("--template-routers", nargs="+", default=list(TEMPLATE_ROUTERS))
    p_fig8.add_argument("--paraphrase-routers", nargs="+", default=list(PARAPHRASE_ROUTERS))
    args = parser.parse_args(argv)

    if args.command == "counts":
        for path in args.files:
            print(format_counts(counts(path)))
        return 0
    if args.command == "complete":
        reason = check_part(args.part, args.router, args.questions, args.answerer, args.llm_dtype)
        print(f"{args.router}: complete" if reason is None else f"{args.router}: not complete: {reason}")
        return 0 if reason is None else 1
    if args.command == "fig8":
        print(fig8(args.run, args.out, args.template_routers, args.paraphrase_routers))
        return 0
    path = summary(args.run, args.template_questions, args.paraphrase_questions, args.template_routers,
                   args.paraphrase_routers)
    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
