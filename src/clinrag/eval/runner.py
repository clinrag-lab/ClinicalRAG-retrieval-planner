"""Experiment runner: every router on every question, executed with fallback, scored and summarized.

Usage:
    python -m clinrag.eval.runner --sources reference --workload data/workload --split dev \\
        --routers oracle p1 p2_tfidf query_all always_filter_first always_search_first \\
        --config outputs/calibrated_config.yaml

Sources:
    fake       reference implementations over the bundled fixture tables and
               notes (tests/fixtures); for smoke tests
    reference  reference implementations over the MIMIC-IV demo tables and the
               generated notes (--tables-dir, --notes-dir)
    faiss      the demo tables and ICD graph as in reference, with the notes
               served by FaissNotesSource from a prebuilt index (--index-dir,
               built by scripts/build_vector_index.py); BGE chunks, so an
               admission can own several chunks. Query embeddings go through
               a CachingEncoder, cleared before every question execution
    duckdb     DuckDBRelationalSource over a prebuilt DuckDB file (--db, built by
               scripts/build_duckdb.py), with the reference notes and ICD graph
    duckdb_faiss  DuckDBRelationalSource (--db) with the FAISS notes as in faiss

For each (router, question) the runner decides and executes with
run_with_fallback --repeats times. Latency and planning overhead are the
medians over repeats; tokens, items and recall come from the first repeat
(they do not change between repeats). Output in --out (default
outputs/runs/<timestamp>/):

    results.jsonl  one line per (router, question): every RoutingResult field,
                   recall, estimated and measured cost, selectivity, answer fields
    results.csv    per router and class: recall, tokens, latency, overhead, fallbacks
    results.md     the same as tables
    run.json       arguments and planner config
    analysis/      cost model check, order check, overhead share, tau sweep
                   (clinrag.eval.analysis; skip with --no-analysis)
    figures/       fig1 to fig4

P2 routers train their classifier on <workload>/dev.jsonl at start-up, so on
the dev split P2 is evaluated on its own training questions.

Ablations (recorded in run.json and in every results.jsonl row):
    --no-fallback  execute the chosen plan once; never fall back
    --no-probes    the cost-based routers (oracle, p2_*) cost plans with fixed
                   defaults instead of probe results: relational rows and graph
                   codes are the medians over the calibration queries on
                   <train-workload>/dev.jsonl (clinrag.eval.calibrate.probe_defaults),
                   matching notes = total_notes * 0.3. Planning still enumerates
                   and costs every plan.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from clinrag.baselines.fixed import AlwaysFilterFirstRouter, AlwaysSearchFirstRouter, QueryAllRouter
from clinrag.config import PlannerConfig, load_planner_config
from clinrag.eval.metrics import retrieval_score, score_answer
from clinrag.llm.answerer import Answerer, NullAnswerer
from clinrag.planner.classifier import SourceClassifier
from clinrag.planner.probes import ProbeDefaults
from clinrag.planner.fallback import run_with_fallback
from clinrag.planner.router import ClassifierCostRouter, OracleCostRouter, Router, RoutingResult
from clinrag.planner.rule_planner import RulePlanner
from clinrag.question import ALL_SOURCES
from clinrag.sources.base import Sources
from clinrag.sources.caching_encoder import CachingEncoder, clear_query_cache
from clinrag.sources.reference import build_reference_sources
from clinrag.workload.schema import WorkloadQuestion, read_jsonl

ROUTERS: tuple[str, ...] = (
    "oracle", "p1", "p2_tfidf", "p2_bge", "query_all", "always_filter_first", "always_search_first",
)
FIXTURE_DIR = Path("tests/fixtures")
DEMO_TABLES = Path("data/mimic-iv-demo")
DEMO_NOTES = Path("data/generated")
FAISS_INDEX = Path("data/faiss_index")
DUCKDB_FILE = Path("data/clinrag.duckdb")
BGE_MODEL = "BAAI/bge-base-en-v1.5"
CLASSES = ("A", "B", "C", "D")


# ---- set-up -----------------------------------------------------------------------------


SOURCE_KINDS = ("fake", "reference", "faiss", "duckdb", "duckdb_faiss")


def build_sources(
    kind: str,
    tables_dir: Path | None = None,
    notes_dir: Path | None = None,
    index_dir: Path | None = None,
    encoder: Any = None,
    cache_queries: bool = True,
    db_path: Path | None = None,
) -> Sources:
    if kind == "fake":
        return build_reference_sources(tables_dir or FIXTURE_DIR, notes_dir or FIXTURE_DIR)
    if kind == "reference":
        return build_reference_sources(tables_dir or DEMO_TABLES, notes_dir or DEMO_NOTES)
    if kind == "faiss":
        return build_faiss_sources(tables_dir or DEMO_TABLES, index_dir or FAISS_INDEX, encoder, cache_queries)
    if kind == "duckdb":
        from clinrag.sources.duckdb_relational import build_duckdb_sources

        return build_duckdb_sources(db_path or DUCKDB_FILE, notes_dir or DEMO_NOTES)
    if kind == "duckdb_faiss":
        return build_duckdb_faiss_sources(db_path or DUCKDB_FILE, index_dir or FAISS_INDEX, encoder, cache_queries)
    raise ValueError(f"unknown sources {kind!r}; expected one of {SOURCE_KINDS}")


def build_faiss_sources(
    tables_dir: Path, index_dir: Path, encoder: Any = None, cache_queries: bool = True
) -> Sources:
    """DemoRelationalSource + FaissNotesSource + IcdGraph.

    The BGE model is loaded once here (unless an encoder is passed) and shared
    with the source, which uses it for every query embedding. The index's chunk
    metadata (one row per chunk) is registered with DuckDB, so note-count probes
    count chunks, the same unit as total_notes() and count_notes_for().

    cache_queries wraps the encoder in a CachingEncoder, so a question's probe_search
    and search share one embedding; run_question clears it before each execution.
    Calibration passes False so every timed search pays its own encode.
    """
    from clinrag.sources.faiss_notes import FaissNotesSource
    from clinrag.sources.icd_graph import IcdGraph
    from clinrag.sources.reference import DemoRelationalSource

    if not (Path(index_dir) / "notes.faiss").exists():
        raise FileNotFoundError(f"no FAISS index in {index_dir}; build it with scripts/build_vector_index.py")
    if encoder is None:
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer(BGE_MODEL)
    if cache_queries and not isinstance(encoder, CachingEncoder):
        encoder = CachingEncoder(encoder)
    relational = DemoRelationalSource.from_tables(tables_dir)
    graph = IcdGraph.from_connection(relational.con)
    notes = FaissNotesSource(index_dir, model_name=BGE_MODEL, encoder=encoder)
    relational.register_notes([c["note_id"] for c in notes.chunks], [int(c["hadm_id"]) for c in notes.chunks])
    return Sources(relational=relational, notes=notes, graph=graph)


def build_duckdb_faiss_sources(
    db_path: Path, index_dir: Path, encoder: Any = None, cache_queries: bool = True
) -> Sources:
    """DuckDBRelationalSource + FaissNotesSource + IcdGraph; as build_faiss_sources, with the
    relational store read from a prebuilt DuckDB file (scripts/build_duckdb.py)."""
    from clinrag.sources.duckdb_relational import DuckDBRelationalSource
    from clinrag.sources.faiss_notes import FaissNotesSource

    if not (Path(index_dir) / "notes.faiss").exists():
        raise FileNotFoundError(f"no FAISS index in {index_dir}; build it with scripts/build_vector_index.py")
    if encoder is None:
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer(BGE_MODEL)
    if cache_queries and not isinstance(encoder, CachingEncoder):
        encoder = CachingEncoder(encoder)
    relational = DuckDBRelationalSource(db_path)
    graph = relational.icd_graph()
    notes = FaissNotesSource(index_dir, model_name=BGE_MODEL, encoder=encoder)
    relational.register_notes([c["note_id"] for c in notes.chunks], [int(c["hadm_id"]) for c in notes.chunks])
    return Sources(relational=relational, notes=notes, graph=graph)


P2_TRAIN = ("dev", "dev+paraphrases")


def train_classifier(workload_dir: Path, featurizer: str, train: str = "dev", seed: int = 0) -> SourceClassifier:
    """Fit on the dev split only: the original dev questions, plus (train="dev+paraphrases")
    the dev questions rendered in the train phrasing families (seed), as in the routing report."""
    if train not in P2_TRAIN:
        raise ValueError(f"train must be one of {P2_TRAIN}")
    dev = read_jsonl(workload_dir / "dev.jsonl")
    specs = [q.spec for q in dev]
    if train == "dev+paraphrases":
        from clinrag.workload.paraphrase import paraphrase_all, split_families

        specs += [p.question.spec for p in paraphrase_all(dev, split_families(seed)["train"], seed)]
    return SourceClassifier(featurizer=featurizer).fit(specs)


def load_or_train_classifiers(
    router_names: Sequence[str], workload_dir: Path, train: str, model_dir: Path | None, seed: int = 0
) -> dict[str, SourceClassifier]:
    """P2 classifiers by featurizer. With model_dir, reuse <model_dir>/p2_<featurizer>.joblib when present,
    else train and save it there."""
    classifiers: dict[str, SourceClassifier] = {}
    for name in router_names:
        if not name.startswith("p2_"):
            continue
        featurizer = name.split("_", 1)[1]
        path = None if model_dir is None else model_dir / f"p2_{featurizer}.joblib"
        if path is not None and path.exists():
            classifiers[featurizer] = SourceClassifier.load(path)
            continue
        classifiers[featurizer] = train_classifier(workload_dir, featurizer, train, seed)
        if path is not None:
            classifiers[featurizer].save(path)
    return classifiers


def build_router(
    name: str,
    sources: Sources,
    config: PlannerConfig,
    workload_dir: Path,
    tau: float,
    classifiers: dict[str, SourceClassifier] | None = None,
    probe_defaults: ProbeDefaults | None = None,
) -> Router:
    """Router for a CLI name. classifiers caches trained P2 models by featurizer.

    probe_defaults (no-probes ablation) applies to the cost-based routers, oracle and p2_*.
    """
    if name == "oracle":
        return OracleCostRouter(sources, config, probe_defaults)
    if name == "p1":
        return RulePlanner()
    if name in ("p2_tfidf", "p2_bge"):
        featurizer = name.split("_", 1)[1]
        cache = classifiers if classifiers is not None else {}
        if featurizer not in cache:
            cache[featurizer] = train_classifier(workload_dir, featurizer)
        return ClassifierCostRouter(
            cache[featurizer], sources, config, tau=tau, name=name, probe_defaults=probe_defaults
        )
    if name == "query_all":
        return QueryAllRouter(sources, config)
    if name == "always_filter_first":
        return AlwaysFilterFirstRouter(sources, config)
    if name == "always_search_first":
        return AlwaysSearchFirstRouter(sources, config)
    raise ValueError(f"unknown router {name!r}; expected one of {ROUTERS}")


# ---- one question -----------------------------------------------------------------------


def run_question(
    label: str,
    router: Router,
    question: WorkloadQuestion,
    sources: Sources,
    config: PlannerConfig,
    repeats: int = 3,
    answerer: Answerer | None = None,
    max_fallbacks: int | None = None,
) -> dict[str, Any]:
    """Decide and execute repeats times; one JSON-ready record.

    max_fallbacks: None uses config.max_fallbacks; 0 is the no-fallback ablation.
    """
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    spec = question.spec
    decisions, runs = [], []
    for _ in range(repeats):
        clear_query_cache(sources)  # each execution pays its own first encode
        decision = router.decide(spec)
        decisions.append(decision)
        runs.append(run_with_fallback(decision, sources, spec, config, max_fallbacks=max_fallbacks))
    decision, run = decisions[0], runs[0]
    latencies = [r.total_latency_ms for r in runs]
    overheads = [d.planning_overhead_ms for d in decisions]

    result = RoutingResult.from_run(label, spec, decision, run)
    result = replace(result, latency_ms=statistics.median(latencies), planning_overhead_ms=statistics.median(overheads))
    score = retrieval_score(question, run.items, config.top_k)

    answerer = answerer or NullAnswerer()
    answer, prompt_tokens, answer_ms = answerer.answer(spec, run.items)
    answer_score = None if answer is None else score_answer(question, answer).value

    record = result.to_dict()
    plan_sources = decision.plan.sources
    record.update({
        "plan_sources": sorted(plan_sources),
        "plan_missed_sources": sorted(spec.required_sources - plan_sources),
        "template_id": question.template_id,
        "selectivity": question.gold.get("selectivity"),
        "recall": score.value,
        "recall_metric": score.metric,
        "measured_cost": result.latency_ms + config.alpha * result.tokens,
        "evidence_ok": run.ok,
        "attempts": [asdict(a) | {"sources_queried": sorted(a.sources_queried)} for a in run.attempts],
        "latency_runs_ms": latencies,
        "overhead_runs_ms": overheads,
        "answerer": answerer.name,
        "answer": answer,
        "answer_score": answer_score,
        "prompt_tokens": prompt_tokens,
        "answer_latency_ms": answer_ms,
    })
    return record


# ---- summaries --------------------------------------------------------------------------

SUMMARY_FIELDS = (
    "router", "qclass", "n", "mean_recall", "mean_tokens", "mean_latency_ms", "p95_latency_ms",
    "mean_overhead_ms", "fallback_rate", "missed_source_rate", "mean_sources_queried", "sources_queried",
    "mean_est_cost", "mean_measured_cost",
)


def _mean(values: Sequence[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return float(np.mean(present)) if present else None


def summarize(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per (router, class) plus (router, "all"), routers in first-seen order."""
    routers = list(dict.fromkeys(r["router"] for r in records))
    rows = []
    for router in routers:
        mine = [r for r in records if r["router"] == router]
        for qclass in (*CLASSES, "all"):
            group = mine if qclass == "all" else [r for r in mine if r["qclass"] == qclass]
            if not group:
                continue
            latencies = [r["latency_ms"] for r in group]
            share = {s: sum(s in r["sources_queried"] for r in group) / len(group) for s in sorted(ALL_SOURCES)}
            rows.append({
                "router": router,
                "qclass": qclass,
                "n": len(group),
                "mean_recall": _mean([r["recall"] for r in group]),
                "mean_tokens": _mean([r["tokens"] for r in group]),
                "mean_latency_ms": _mean(latencies),
                "p95_latency_ms": float(np.percentile(latencies, 95)),
                "mean_overhead_ms": _mean([r["planning_overhead_ms"] for r in group]),
                "fallback_rate": sum(r["fallbacks_used"] > 0 for r in group) / len(group),
                "missed_source_rate": sum(bool(r.get("plan_missed_sources")) for r in group) / len(group),
                "mean_sources_queried": _mean([len(r["sources_queried"]) for r in group]),
                "sources_queried": " ".join(f"{s}:{v:.2f}" for s, v in share.items()),
                "mean_est_cost": _mean([r["est_cost"] for r in group]),
                "mean_measured_cost": _mean([r["measured_cost"] for r in group]),
            })
    return rows


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def markdown_table(rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> str:
    lines = ["| " + " | ".join(fields) + " |", "|" + "---|" * len(fields)]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(f)) for f in fields) + " |")
    return "\n".join(lines)


def write_summaries(out: Path, records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = summarize(records)
    with (out / "results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    overall = [r for r in rows if r["qclass"] == "all"]
    per_class = [r for r in rows if r["qclass"] != "all"]
    short = ("router", "qclass", "n", "mean_recall", "mean_tokens", "mean_latency_ms", "p95_latency_ms",
             "mean_overhead_ms", "fallback_rate", "missed_source_rate", "sources_queried")
    md = [
        "# Results",
        "",
        "Latency and overhead in ms (median over repeats per question; mean and p95 over questions).",
        "Recall: value row (A), gold admission's note (per-admission B), normalized recall@k (B list, C, D).",
        "fallback_rate: share of questions that ran at least one fallback plan.",
        "missed_source_rate: share of questions whose chosen plan lacks a source the question needs.",
        "sources_queried: share of questions that queried each source.",
        "",
        "## All classes",
        "",
        markdown_table(overall, short),
        "",
        "## Per class",
        "",
        markdown_table(per_class, short),
        "",
    ]
    (out / "results.md").write_text("\n".join(md), encoding="utf-8")
    return rows


# ---- main -------------------------------------------------------------------------------


def run_experiment(
    sources: Sources,
    questions: Sequence[WorkloadQuestion],
    router_names: Sequence[str],
    config: PlannerConfig,
    workload_dir: Path,
    out: Path,
    tau: float = 0.5,
    repeats: int = 3,
    answerer: Answerer | None = None,
    progress: Callable[[str], None] | None = None,
    classifiers: dict[str, SourceClassifier] | None = None,
    no_fallback: bool = False,
    probe_defaults: ProbeDefaults | None = None,
) -> list[dict[str, Any]]:
    """Run every router on every question; write results.jsonl, results.csv, results.md.

    classifiers: P2 models by featurizer; missing ones are trained on workload_dir/dev.jsonl.
    no_fallback, probe_defaults: the ablations (see the module docstring).
    """
    ablation = {"no_fallback": no_fallback, "no_probes": probe_defaults is not None}
    out.mkdir(parents=True, exist_ok=True)
    classifiers = {} if classifiers is None else classifiers
    records: list[dict[str, Any]] = []
    with (out / "results.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for name in router_names:
            start = time.perf_counter()
            router = build_router(name, sources, config, workload_dir, tau, classifiers, probe_defaults)
            for question in questions:
                record = run_question(name, router, question, sources, config, repeats, answerer,
                                      max_fallbacks=0 if no_fallback else None)
                record.update(ablation)
                records.append(record)
                f.write(json.dumps(record, sort_keys=True, default=str) + "\n")
            if progress:
                progress(f"{name}: {len(questions)} questions in {time.perf_counter() - start:.1f} s")
    write_summaries(out, records)
    return records


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", choices=SOURCE_KINDS, default="fake")
    parser.add_argument("--tables-dir", type=Path, default=None)
    parser.add_argument("--notes-dir", type=Path, default=None)
    parser.add_argument("--index-dir", type=Path, default=None, help=f"with --sources faiss (default {FAISS_INDEX})")
    parser.add_argument("--db", type=Path, default=None, help=f"with --sources duckdb* (default {DUCKDB_FILE})")
    parser.add_argument("--workload", type=Path, default=Path("data/workload"), help="directory with dev.jsonl and test.jsonl")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--routers", nargs="+", choices=ROUTERS, default=[r for r in ROUTERS if r != "p2_bge"])
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--repeats", type=int, default=3, help="executions per question; median latency is kept")
    parser.add_argument("--config", type=Path, default=None, help="planner config YAML, e.g. outputs/calibrated_config.yaml")
    parser.add_argument("--limit", type=int, default=None, help="use only the first N questions (smoke runs)")
    parser.add_argument("--out", type=Path, default=None, help="default outputs/runs/<timestamp>/")
    parser.add_argument("--no-analysis", action="store_true", help="skip the analyses and figures")
    parser.add_argument(
        "--sweep-on", choices=["paraphrased", "original", "none"], default="paraphrased",
        help="questions for the P2 TF-IDF tau sweep: held-out phrasings (default), as written, or skip",
    )
    parser.add_argument("--train-workload", type=Path, default=None,
                        help="workload whose dev.jsonl trains P2 (default --workload); use data/workload when "
                             "--workload is a paraphrase set")
    parser.add_argument("--p2-train", choices=P2_TRAIN, default="dev",
                        help="P2 training data: dev questions, or dev plus their train-family paraphrases")
    parser.add_argument("--p2-model-dir", type=Path, default=None,
                        help="reuse <dir>/p2_<featurizer>.joblib if present, else train and save there")
    parser.add_argument("--no-fallback", action="store_true",
                        help="ablation: execute the chosen plan once and never fall back")
    parser.add_argument("--no-probes", action="store_true",
                        help="ablation: cost plans with fixed defaults instead of probe results (oracle, p2_*)")
    parser.add_argument("--fig6", action="store_true", help="write figures/fig6_paraphrase_e2e.png (tokens vs recall)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    config = load_planner_config(args.config) if args.config else PlannerConfig()
    out = args.out or Path("outputs/runs") / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    sources = build_sources(args.sources, args.tables_dir, args.notes_dir, args.index_dir, db_path=args.db)
    questions = read_jsonl(args.workload / f"{args.split}.jsonl")
    if args.limit is not None:
        questions = questions[: args.limit]
    train_workload = args.train_workload or args.workload
    probe_defaults = None
    if args.no_probes:
        from clinrag.eval.calibrate import probe_defaults as calibration_defaults

        probe_defaults = calibration_defaults(read_jsonl(train_workload / "dev.jsonl"), sources)
    (out / "run.json").write_text(
        json.dumps({**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    "planner_config": asdict(config), "n_questions": len(questions),
                    "probe_defaults": None if probe_defaults is None else asdict(probe_defaults)}, indent=2) + "\n",
        encoding="utf-8",
    )
    classifiers = load_or_train_classifiers(args.routers, train_workload, args.p2_train, args.p2_model_dir)
    records = run_experiment(
        sources, questions, args.routers, config, train_workload, out, args.tau, args.repeats,
        progress=print, classifiers=classifiers, no_fallback=args.no_fallback, probe_defaults=probe_defaults,
    )
    if args.fig6:
        from clinrag.eval.analysis import fig_paraphrase_e2e

        (out / "figures").mkdir(parents=True, exist_ok=True)
        fig_paraphrase_e2e(records, out / "figures" / "fig6_paraphrase_e2e.png")
    if not args.no_analysis:
        from clinrag.eval.analysis import run_analyses

        start = time.perf_counter()
        run_analyses(records, questions, sources, config, args.workload, out, args.repeats, args.sweep_on)
        print(f"analyses: {time.perf_counter() - start:.1f} s")
    print(f"results: {out}")
    return out


if __name__ == "__main__":
    main()
