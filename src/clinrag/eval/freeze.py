"""Write the frozen results summary for a demo run directory.

Usage:
    python -m clinrag.eval.freeze --run outputs/runs/final-demo
    python -m clinrag.eval.freeze --merge p1=DIR p2_tfidf=DIR ... --questions test.jsonl --into DIR

Expects the run directory laid out by the commands in COMMANDS:
    config/       calibrated_config.yaml (tuned s and beta), calibration and tune_order reports
    template-test/  runner on the template test split, with analyses
    scale-study/    order check at notes scales 1, 10 and 50
    paraphrase-test/  runner on the held-out phrasing test set, one router per run under parts/,
                      merged with --merge
    routing/        classifier routing report on held-out phrasings (the only p2_bge evaluation)
Writes SUMMARY.md (headline numbers, each with the command and config that
produced it) and copies every figure into <run>/figures/.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

CONFIG = "outputs/runs/final-demo/config/calibrated_config.yaml"
PY = "python -m"
COMMANDS: dict[str, list[str]] = {
    "config": [
        f"{PY} clinrag.eval.calibrate --sources reference --workload data/workload --split dev --out {CONFIG}",
        f"{PY} clinrag.workload.notes_gen --tables-dir data/mimic-iv-demo --out-dir data/generated --seed 0 --scale 10",
        f"{PY} clinrag.workload.notes_gen --tables-dir data/mimic-iv-demo --out-dir data/generated --seed 0 --scale 50",
        f"{PY} clinrag.eval.tune_order --split dev --config {CONFIG} --notes-dir data/generated/scale_10 "
        "--report-dir outputs/runs/final-demo/config",
    ],
    "template-test": [
        f"{PY} clinrag.eval.runner --sources reference --workload data/workload --split test "
        "--routers oracle p1 p2_tfidf query_all always_filter_first always_search_first "
        f"--config {CONFIG} --out outputs/runs/final-demo/template-test",
    ],
    "scale-study": [
        f"{PY} clinrag.eval.scale_study --split test --scales 1 10 50 --config {CONFIG} "
        "--out outputs/runs/final-demo/scale-study",
    ],
    "paraphrase-test": [
        f"{PY} clinrag.workload.paraphrase --workload-dir data/workload --out-dir data/workload/paraphrase --seed 0",
        # p1 rows: kept from a five-router run (p1 p2_tfidf p2_bge oracle query_all) that ran out of memory
        # in p2_bge; its results.jsonl is kept as parts/partial/results.jsonl.
        *[
            f"{PY} clinrag.eval.runner --sources reference --workload data/workload/paraphrase --split test "
            "--train-workload data/workload --p2-train dev+paraphrases "
            "--p2-model-dir outputs/runs/final-demo/paraphrase-test/models "
            f"--routers {router} --config {CONFIG} --no-analysis "
            f"--out outputs/runs/final-demo/paraphrase-test/parts/{router}"
            for router in ("p2_tfidf", "oracle", "query_all")
        ],
        f"{PY} clinrag.eval.freeze --merge p1=outputs/runs/final-demo/paraphrase-test/parts/partial "
        + " ".join(f"{r}=outputs/runs/final-demo/paraphrase-test/parts/{r}" for r in ("p2_tfidf", "oracle", "query_all"))
        + " --questions data/workload/paraphrase/test.jsonl --into outputs/runs/final-demo/paraphrase-test",
    ],
    "routing": [
        f"{PY} clinrag.planner.train_classifier --workload-dir data/workload --paraphrase",
        "cp outputs/routing_paraphrase_report.json outputs/runs/final-demo/routing/",
    ],
}
BGE_NOTE = (
    "P2 with BGE embeddings (p2_bge) was evaluated for routing only, because the end-to-end run with BGE "
    "exceeded laptop memory."
)
FIGURES: dict[str, str] = {
    "fig1_tokens_vs_recall.png": "template-test",
    "fig2_order_vs_selectivity.png": "scale-study",
    "fig3_tau_sweep.png": "template-test",
    "fig4_overhead.png": "template-test",
    "fig5_scale.png": "scale-study",
    "fig6_paraphrase_e2e.png": "paraphrase-test",
}


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _f(value: Any, digits: int = 3) -> str:
    if value in (None, ""):
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _table(header: list[str], rows: list[list[Any]]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + [
        "| " + " | ".join(str(c) for c in row) + " |" for row in rows
    ]


def _commands(key: str) -> list[str]:
    return ["```", *COMMANDS[key], "```"]


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def summary(run: Path) -> str:
    config_path = run / "config" / "calibrated_config.yaml"
    config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()[:12]
    tune = _json(run / "config" / "tune_order_report.json")
    w = tune["winner"]
    lines = [
        "# Results: demo freeze (results-demo-v1)",
        "",
        f"Code: commit `{git_commit()}`. Data: open MIMIC-IV demo (275 admissions), generated notes, "
        "workload in data/workload. Only the dev split is used for calibration, tuning and training.",
        f"Config for every run: `{CONFIG}` (sha256 `{config_sha}`).",
        "",
        "## Config: calibration and order-rule tuning",
        "",
        *_commands("config"),
        "",
        f"Tuned order rule (dev, class C/D, notes scale 10): s = {w['s']}, beta = {w['beta']}; the rule picks "
        f"the better order on {w['agreement']:.1%} of {tune['n_questions']} dev questions "
        f"(default s = 0.1, beta = 1: {tune['default']['agreement']:.1%}).",
        "",
    ]

    rows = [r for r in _csv(run / "template-test" / "results.csv") if r["qclass"] == "all"]
    lines += ["## Template test run (171 test questions, scale 1)", "", *_commands("template-test"), ""]
    lines += _table(
        ["router", "recall", "tokens", "latency ms", "p95 ms", "overhead ms", "fallback", "missed sources"],
        [[r["router"], _f(r["mean_recall"]), _f(r["mean_tokens"], 1), _f(r["mean_latency_ms"], 2),
          _f(r["p95_latency_ms"], 2), _f(r["mean_overhead_ms"], 2), _f(r["fallback_rate"]),
          _f(r["missed_source_rate"])] for r in rows],
    )

    analysis = _json(run / "template-test" / "analysis" / "summary.json")
    cm = analysis["cost_model"]
    lines += ["", "## Cost model: Spearman, estimated vs measured (template test run, every candidate plan once)", ""]
    lines += _table(
        ["group", "plans", "cost", "latency only", "tokens only", "cheapest-plan agreement"],
        [[g, v["n_plans"], _f(v["spearman_cost"]), _f(v["spearman_latency"]), _f(v["spearman_tokens"]),
          _f(v["cheapest_plan_agreement"])] for g, v in cm.items()],
    )

    lines += ["", "## Overhead share (template test run)", ""]
    lines += _table(
        ["router", "overhead ms", "retrieval ms", "overhead share of total time"],
        [[r["router"], _f(r["mean_overhead_ms"], 2), _f(r["mean_retrieval_ms"], 2),
          _f(r["overhead_share_of_total_time"])] for r in analysis["overhead_share"]],
    )

    scale = _json(run / "scale-study" / "analysis" / "order_scale_summary.json")
    lines += ["", "## Order check per scale (test, class C/D, re-calibrated per scale, s and beta fixed)", "",
              *_commands("scale-study"), ""]
    policy_rows = []
    for s in scale["summaries"]:
        for p in s["policies"]:
            policy_rows.append([f"{s['total_notes']:,}", f"{s['filter_first_better']:.3f}", p["policy"],
                                _f(p["picks_better"]), _f(p["mean_recall"]), _f(p["mean_latency_ms"], 2)])
    lines += _table(
        ["notes", "filter_first better", "policy", "picks better order", "mean recall", "mean latency ms"],
        policy_rows,
    )

    para = _csv(run / "paraphrase-test" / "results.csv")
    lines += ["", "## End to end on held-out phrasings (684 test paraphrases, scale 1)", "",
              *_commands("paraphrase-test"), "", BGE_NOTE, ""]
    lines += _table(
        ["router", "class", "recall", "tokens", "latency ms", "overhead ms", "fallback", "missed sources"],
        [[r["router"], r["qclass"], _f(r["mean_recall"]), _f(r["mean_tokens"], 1), _f(r["mean_latency_ms"], 2),
          _f(r["mean_overhead_ms"], 2), _f(r["fallback_rate"]), _f(r["missed_source_rate"])] for r in para],
    )

    routing = _json(run / "routing" / "routing_paraphrase_report.json")
    lines += ["", "## Classifier routing on held-out phrasings (684 test paraphrases, tau 0.5)", "",
              *_commands("routing"), "", "This is the only evaluation of p2_bge in the freeze; see the note above.",
              ""]
    route_rows = []
    for name, r in routing["routers"].items():
        if "skipped" in r:
            route_rows.append([name, "skipped", "", "", "", "", ""])
            continue
        per = r["per_source"]
        route_rows.append([name, _f(r["exact_match"]), _f(r["missed_source_rate"]), _f(per["relational"]["f1"]),
                           _f(per["notes"]["f1"]), _f(per["graph"]["f1"]), _f(r["latency_ms"]["mean_ms"], 2)])
    lines += _table(["router", "exact match", "missed sources", "F1 relational", "F1 notes", "F1 graph",
                     "predict ms"], route_rows)

    lines += ["", "## Figures", ""]
    lines += [f"- `figures/{name}` (from {src}/)" for name, src in FIGURES.items()]
    return "\n".join(lines) + "\n"


def copy_figures(run: Path) -> list[Path]:
    out = run / "figures"
    out.mkdir(parents=True, exist_ok=True)
    copied = []
    for name, src in FIGURES.items():
        path = run / src / "figures" / name
        if path.exists():
            copied.append(Path(shutil.copy2(path, out / name)))
    return copied


def merge_parts(parts: dict[str, Path], questions: Path, into: Path) -> list[dict[str, Any]]:
    """Take each router's rows from its own run directory and write one results.jsonl, results.csv,
    results.md and figures/fig6_paraphrase_e2e.png into `into`.

    Every router must have exactly one row per question in `questions`; anything else raises.
    """
    from clinrag.eval.analysis import fig_paraphrase_e2e
    from clinrag.eval.runner import write_summaries
    from clinrag.workload.schema import read_jsonl

    qids = [q.spec.qid for q in read_jsonl(questions)]
    records: list[dict[str, Any]] = []
    for router, part in parts.items():
        with (part / "results.jsonl").open(encoding="utf-8") as f:
            rows = [r for r in (json.loads(line) for line in f if line.strip()) if r["router"] == router]
        got = [r["qid"] for r in rows]
        if len(got) != len(set(got)) or set(got) != set(qids):
            raise ValueError(f"{router} in {part}: {len(set(got))} distinct of {len(qids)} questions, "
                             f"{len(got)} rows")
        order = {qid: i for i, qid in enumerate(qids)}
        records += sorted(rows, key=lambda r: order[r["qid"]])
    into.mkdir(parents=True, exist_ok=True)
    with (into / "results.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True, default=str) + "\n")
    write_summaries(into, records)
    (into / "figures").mkdir(parents=True, exist_ok=True)
    fig_paraphrase_e2e(records, into / "figures" / "fig6_paraphrase_e2e.png")
    return records


def _part(value: str) -> tuple[str, Path]:
    router, sep, path = value.partition("=")
    if not sep or not router or not path:
        raise argparse.ArgumentTypeError(f"expected ROUTER=DIR, got {value!r}")
    return router, Path(path)


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, default=Path("outputs/runs/final-demo"))
    parser.add_argument("--merge", nargs="+", type=_part, metavar="ROUTER=DIR",
                        help="instead of the summary, merge each router's rows from its run directory")
    parser.add_argument("--questions", type=Path, help="with --merge: the question file every router must cover")
    parser.add_argument("--into", type=Path, help="with --merge: output directory")
    args = parser.parse_args(argv)
    if args.merge:
        if args.questions is None or args.into is None:
            parser.error("--merge needs --questions and --into")
        records = merge_parts(dict(args.merge), args.questions, args.into)
        print(f"{args.into / 'results.jsonl'} ({len(records)} rows)")
        return args.into / "results.jsonl"
    copied = copy_figures(args.run)
    path = args.run / "SUMMARY.md"
    path.write_text(summary(args.run), encoding="utf-8")
    print(f"{path} ({len(copied)} figures copied)")
    return path


if __name__ == "__main__":
    main()
