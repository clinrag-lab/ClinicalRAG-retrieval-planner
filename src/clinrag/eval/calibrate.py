"""Fit the planner's cost constants to measured timings on the actual sources.

Usage:
    python -m clinrag.eval.calibrate --sources reference --workload data/workload --split dev

Measurements (each timed --repeats times; the median is kept):

  relational_fixed_ms      intercept of RelationalSource.run time against rows returned,
                           over the workload's relational queries
  relational_ms            median run time minus relational_fixed_ms (the typical
                           per-query part above the fixed cost)
  avg_tokens_per_row       mean tokens of the rows those queries return
  graph_fixed_ms, graph_ms the same for GraphSource.expand against codes returned,
                           over the workload's categories
  note_search_fixed_ms     intercept of the id-filter regression below: a search that
                           scans and fetches nothing costs only its fixed part
  id_filter_ms_per_1k_ids  slope of search(q, 1, hadm_ids=m ids that match no note) against m
  note_scan_ms_per_1k_notes slope of search(q, 1, hadm_ids=s admissions) against the notes
                           those admissions own, after subtracting the id filter
  note_fetch_ms_per_1k_candidates slope of unfiltered search(q, n) against n
  avg_tokens_per_chunk     mean tokens of a note chunk
  min_similarity           (when concept tags are available) the score threshold that best
                           separates notes tagged with the queried concept from the rest,
                           by balanced accuracy; the evidence check's similarity floor
                           depends on the scoring function, so it is fitted per source

Slopes and intercepts come from least squares; the report lists slope,
intercept and R^2 for every fit. Intercepts are clipped at 0. Writes outputs/calibrated_config.yaml
(all planner keys, fitted ones replaced) and outputs/calibration_report.json / .md.
The runner uses the YAML through --config.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from clinrag.config import PlannerConfig, load_planner_config
from clinrag.question import NOTES, RELATIONAL
from clinrag.sources.base import RelationalQuery, Sources
from clinrag.sources.tokens import count_tokens
from clinrag.workload.schema import WorkloadQuestion, read_jsonl
from clinrag.workload.tables import table_path

Clock = Callable[[], float]  # seconds
ID_FILTER_SIZES: tuple[int, ...] = (100, 1000, 5000, 20000)
MAX_QUERIES = 40


@dataclass(frozen=True)
class LineFit:
    """y = slope * x + intercept, by least squares. slope is clipped at 0 for the config."""

    slope: float
    intercept: float
    r2: float
    n: int

    @classmethod
    def fit(cls, x: Sequence[float], y: Sequence[float]) -> LineFit:
        xs, ys = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        if len(xs) < 2 or np.ptp(xs) == 0:
            return cls(0.0, float(np.median(ys)) if len(ys) else 0.0, 0.0, len(xs))
        slope, intercept = np.polyfit(xs, ys, 1)
        pred = slope * xs + intercept
        ss_tot = float(np.sum((ys - ys.mean()) ** 2))
        r2 = 1.0 - float(np.sum((ys - pred) ** 2)) / ss_tot if ss_tot > 0 else 1.0
        return cls(float(slope), float(intercept), r2, len(xs))


def _timed(clock: Clock, fn: Callable[[], Any], repeats: int) -> tuple[float, Any]:
    """Median ms over repeats, and the last result."""
    times, result = [], None
    for _ in range(repeats):
        start = clock()
        result = fn()
        times.append((clock() - start) * 1000.0)
    return statistics.median(times), result


def _spread(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "median": float(np.median(values)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
    }


def _sizes(total: int) -> list[int]:
    """Admission counts from 1 to total, roughly geometric."""
    out = {1, total}
    n = 2
    while n < total:
        out.add(n)
        n = int(n * 2)
    return sorted(s for s in out if 1 <= s <= total)


def relational_queries(questions: Sequence[WorkloadQuestion], sources: Sources) -> list[RelationalQuery]:
    """Distinct relational queries of the workload (graph codes expanded), at most MAX_QUERIES."""
    seen: dict[RelationalQuery, None] = {}
    for q in questions:
        spec = q.spec
        if RELATIONAL not in spec.required_sources:
            continue
        codes = tuple(sources.require_graph().expand(spec.category)) if spec.category else ()
        seen.setdefault(RelationalQuery(lab=spec.relational_filter, icd_codes=codes), None)
    return list(seen)[:MAX_QUERIES]


def note_queries(questions: Sequence[WorkloadQuestion]) -> list[str]:
    queries = sorted({q.spec.note_query for q in questions if NOTES in q.spec.required_sources and q.spec.note_query})
    return queries or ["discharge"]


def fit_min_similarity(sources: Sources, concept_tags: dict[str, set[int]]) -> dict[str, Any]:
    """Threshold on note score that best separates tagged from untagged notes (balanced accuracy)."""
    notes = sources.require_notes()
    total = notes.total_notes()
    scores, labels = [], []
    for concept, tagged in concept_tags.items():
        for item in notes.search(concept, total).items:
            scores.append(float(item["score"]))
            labels.append(int(item["hadm_id"]) in tagged)
    s, y = np.asarray(scores), np.asarray(labels, dtype=bool)
    if not y.any() or y.all():
        return {"fitted": False, "reason": "need both tagged and untagged notes"}
    best_t, best_acc = None, -1.0
    for t in np.unique(s):
        pred = s >= t
        acc = 0.5 * (pred[y].mean() + (~pred[~y]).mean())
        if acc > best_acc:
            best_t, best_acc = float(t), float(acc)
    return {
        "fitted": True,
        "min_similarity": best_t,
        "balanced_accuracy": best_acc,
        "tagged_median_score": float(np.median(s[y])),
        "untagged_median_score": float(np.median(s[~y])),
        "n_pairs": int(len(s)),
    }


def calibrate(
    sources: Sources,
    questions: Sequence[WorkloadQuestion],
    base: PlannerConfig | None = None,
    repeats: int = 5,
    clock: Clock = time.perf_counter,
    concept_tags: dict[str, set[int]] | None = None,
) -> tuple[PlannerConfig, dict[str, Any]]:
    """Measure the sources and return (fitted config, report)."""
    base = base or PlannerConfig()
    report: dict[str, Any] = {"repeats": repeats}
    updates: dict[str, float] = {}

    # Relational: one query each, and the tokens of its rows.
    if sources.relational is not None:
        queries = relational_queries(questions, sources)
        run_ms, run_rows, count_ms, row_tokens = [], [], [], []
        for query in queries:
            ms, result = _timed(clock, lambda q=query: sources.relational.run(q), repeats)
            run_ms.append(ms)
            run_rows.append(len(result.items))
            count_ms.append(_timed(clock, lambda q=query: sources.relational.count(q), repeats)[0])
            row_tokens += [count_tokens(item["text"]) for item in result.items]
        if run_ms:
            rel_fit = LineFit.fit(run_rows, run_ms)
            fixed = max(rel_fit.intercept, 0.0)
            updates["relational_fixed_ms"] = fixed
            updates["relational_ms"] = max(float(np.median(run_ms)) - fixed, 0.0)
            report.setdefault("fits", {})["relational_ms_vs_rows"] = asdict(rel_fit)
            report["relational_run_ms"] = _spread(run_ms)
            report["relational_count_ms"] = _spread(count_ms)
        if row_tokens:
            updates["avg_tokens_per_row"] = float(np.mean(row_tokens))
            report["tokens_per_row"] = _spread(row_tokens)

    # Graph expansion.
    if sources.graph is not None:
        categories = sorted({q.spec.category for q in questions if q.spec.category})
        graph_ms, graph_codes = [], []
        for category in categories:
            ms, codes = _timed(clock, lambda c=category: sources.graph.expand(c), repeats)
            graph_ms.append(ms)
            graph_codes.append(len(codes))
        if graph_ms:
            graph_fit = LineFit.fit(graph_codes, graph_ms)
            fixed = max(graph_fit.intercept, 0.0)
            updates["graph_fixed_ms"] = fixed
            updates["graph_ms"] = max(float(np.median(graph_ms)) - fixed, 0.0)
            report.setdefault("fits", {})["graph_ms_vs_codes"] = asdict(graph_fit)
            report["graph_expand_ms"] = _spread(graph_ms)

    # Note search: id filter, scan and fetch.
    if sources.notes is not None:
        notes = sources.notes
        total = notes.total_notes()
        texts_by_hadm: dict[int, list[str]] = {}
        for item in notes.search(note_queries(questions)[0], total).items:
            texts_by_hadm.setdefault(int(item["hadm_id"]), []).append(item["text"])
        hadms = sorted(texts_by_hadm)
        chunk_tokens = [count_tokens(t) for texts in texts_by_hadm.values() for t in texts]
        updates["avg_tokens_per_chunk"] = float(np.mean(chunk_tokens))
        report["tokens_per_chunk"] = _spread(chunk_tokens)
        queries = note_queries(questions)

        def median_over_queries(call: Callable[[str], Any]) -> float:
            return float(np.median([_timed(clock, lambda q=q: call(q), repeats)[0] for q in queries]))

        missing_start = -(max(ID_FILTER_SIZES) + 1)
        id_x, id_y = [], []
        for m in ID_FILTER_SIZES:
            fake_ids = list(range(missing_start, missing_start + m))
            id_x.append(m / 1000.0)
            id_y.append(median_over_queries(lambda q, ids=fake_ids: notes.search(q, 1, hadm_ids=ids)))
        id_fit = LineFit.fit(id_x, id_y)

        scan_x, scan_y = [], []
        for s in _sizes(len(hadms)):
            ids = hadms[:s]
            ms = median_over_queries(lambda q, ids=ids: notes.search(q, 1, hadm_ids=ids))
            scan_x.append(notes.count_notes_for(ids) / 1000.0)
            scan_y.append(ms - max(id_fit.slope, 0.0) * s / 1000.0)
        scan_fit = LineFit.fit(scan_x, scan_y)

        fetch_x, fetch_y = [], []
        for n in _sizes(total):
            fetch_x.append(n / 1000.0)
            fetch_y.append(median_over_queries(lambda q, n=n: notes.search(q, n)))
        fetch_fit = LineFit.fit(fetch_x, fetch_y)

        updates["note_search_fixed_ms"] = max(id_fit.intercept, 0.0)
        updates["id_filter_ms_per_1k_ids"] = max(id_fit.slope, 0.0)
        updates["note_scan_ms_per_1k_notes"] = max(scan_fit.slope, 0.0)
        updates["note_fetch_ms_per_1k_candidates"] = max(fetch_fit.slope, 0.0)
        report.setdefault("fits", {}).update({
            "id_filter_ms_vs_1k_ids": asdict(id_fit),
            "scan_ms_vs_1k_notes": asdict(scan_fit),
            "fetch_ms_vs_1k_candidates": asdict(fetch_fit),
        })
        report["total_notes"] = total

        if concept_tags:
            sim = fit_min_similarity(sources, concept_tags)
            report["min_similarity_fit"] = sim
            if sim.get("fitted"):
                updates["min_similarity"] = sim["min_similarity"]

    config = replace(base, **updates)
    report["fitted"] = {k: getattr(config, k) for k in updates}
    report["unchanged"] = {f.name: getattr(config, f.name) for f in fields(config) if f.name not in updates}
    return config, report


def load_concept_tags(notes_dir: Path) -> dict[str, set[int]] | None:
    path = table_path(notes_dir, "note_concepts")
    if path is None:
        return None
    frame = pd.read_csv(path)
    return {str(c): set(int(h) for h in g["hadm_id"]) for c, g in frame.groupby("concept")}


def write_outputs(config: PlannerConfig, report: dict[str, Any], out_yaml: Path) -> tuple[Path, Path]:
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    header = "# Planner constants fitted by python -m clinrag.eval.calibrate; use with the runner's --config.\n"
    out_yaml.write_text(header + yaml.safe_dump(asdict(config), sort_keys=False), encoding="utf-8")
    report_json = out_yaml.with_name("calibration_report.json")
    report_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md = ["# Calibration", "", "| constant | fitted value |", "|---|---|"]
    md += [f"| {k} | {v:.6g} |" for k, v in report["fitted"].items()]
    if "fits" in report:
        md += ["", "| fit | slope | intercept (ms) | R^2 | points |", "|---|---|---|---|---|"]
        md += [
            f"| {name} | {f['slope']:.4g} | {f['intercept']:.4g} | {f['r2']:.3f} | {f['n']} |"
            for name, f in report["fits"].items()
        ]
    for key in ("relational_run_ms", "relational_count_ms", "graph_expand_ms"):
        if key in report and report[key].get("n"):
            r = report[key]
            md.append(f"\n{key}: median {r['median']:.4g} ms (IQR {r['p25']:.4g} to {r['p75']:.4g}, n={r['n']})")
    sim = report.get("min_similarity_fit")
    if sim and sim.get("fitted"):
        md.append(
            f"\nmin_similarity {sim['min_similarity']:.4g}: balanced accuracy {sim['balanced_accuracy']:.3f} "
            f"(tagged median score {sim['tagged_median_score']:.3g}, untagged {sim['untagged_median_score']:.3g})"
        )
    report_md = out_yaml.with_name("calibration_report.md")
    report_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    return report_json, report_md


def main(argv: list[str] | None = None) -> PlannerConfig:
    from clinrag.eval.runner import DEMO_NOTES, FIXTURE_DIR, build_sources

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", choices=["fake", "reference"], default="reference")
    parser.add_argument("--tables-dir", type=Path, default=None)
    parser.add_argument("--notes-dir", type=Path, default=None)
    parser.add_argument("--workload", type=Path, default=Path("data/workload"))
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--no-similarity", action="store_true", help="keep the default min_similarity")
    parser.add_argument("--out", type=Path, default=Path("outputs/calibrated_config.yaml"))
    parser.add_argument(
        "--base-config", type=Path, default=None,
        help="config whose unfitted keys (e.g. tuned c and beta) are kept; default PlannerConfig()",
    )
    args = parser.parse_args(argv)

    sources = build_sources(args.sources, args.tables_dir, args.notes_dir)
    notes_dir = args.notes_dir or (FIXTURE_DIR if args.sources == "fake" else DEMO_NOTES)
    tags = None if args.no_similarity else load_concept_tags(notes_dir)
    questions = read_jsonl(args.workload / f"{args.split}.jsonl")
    base = load_planner_config(args.base_config) if args.base_config else PlannerConfig()
    config, report = calibrate(sources, questions, base=base, repeats=args.repeats, concept_tags=tags)
    _, report_md = write_outputs(config, report, args.out)
    print(report_md.read_text(encoding="utf-8"))
    print(f"config: {args.out}")
    return config


if __name__ == "__main__":
    main()
