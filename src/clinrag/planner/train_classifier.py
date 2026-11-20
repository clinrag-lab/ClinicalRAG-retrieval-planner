"""Train the source classifier on the dev split and report how well it routes.

Usage:
    python -m clinrag.planner.train_classifier --workload-dir data/workload \\
        --out outputs/classifier.joblib --featurizer tfidf

Reports (written as JSON next to the model, <out stem>_report.json, unless --report is given):
- dev, 5-fold cross-validation grouped by template_id, and test: per-source
  precision / recall / F1 and exact-match accuracy of the predicted source set
  at --tau;
- mean and p95 latency of single-question predict_proba on test, in ms;
- tau sweep on test, tau = 0.1 .. 0.9: fraction of questions where a needed
  source was missed, and the average number of sources predicted.

With --paraphrase it instead compares routers on held-out phrasings
(clinrag.workload.paraphrase): classifiers train on the original dev questions
plus their paraphrases in the train families, and every router is evaluated on
test-template questions rendered only in the test families. Routers: P1 keyword
rules, and P2 (SourceClassifier) for each --featurizers entry. The report goes
to outputs/routing_paraphrase_report.json unless --report is given:

    python -m clinrag.planner.train_classifier --workload-dir data/workload --paraphrase
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import GroupKFold

from clinrag.planner.classifier import SOURCES, SourceClassifier, threshold
from clinrag.planner.rule_planner import rule_sources
from clinrag.question import NOTES
from clinrag.workload.paraphrase import Paraphrase, paraphrase_all, split_families
from clinrag.workload.schema import WorkloadQuestion, read_jsonl

TAUS: tuple[float, ...] = tuple(round(0.1 * i, 1) for i in range(1, 10))
N_FOLDS = 5


def set_metrics(predicted: Sequence[set[str]], required: Sequence[frozenset[str]]) -> dict[str, Any]:
    """Per-source precision/recall/F1/support and exact-match accuracy of predicted source sets."""
    per_source: dict[str, dict[str, float]] = {}
    for source in SOURCES:
        tp = sum(source in p and source in r for p, r in zip(predicted, required))
        fp = sum(source in p and source not in r for p, r in zip(predicted, required))
        fn = sum(source not in p and source in r for p, r in zip(predicted, required))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_source[source] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": tp + fn,
        }
    exact = sum(set(p) == set(r) for p, r in zip(predicted, required))
    return {"n": len(required), "per_source": per_source, "exact_match": round(exact / len(required), 4)}


def tau_sweep(probs: Sequence[dict[str, float]], required: Sequence[frozenset[str]]) -> list[dict[str, float]]:
    """For each tau: fraction of questions missing a needed source, average sources predicted, exact match."""
    rows = []
    for tau in TAUS:
        predicted = [threshold(p, tau) for p in probs]
        missed = sum(bool(r - p) for p, r in zip(predicted, required))
        exact = sum(p == set(r) for p, r in zip(predicted, required))
        rows.append(
            {
                "tau": tau,
                "missed_needed_source_fraction": round(missed / len(required), 4),
                "avg_sources_predicted": round(float(np.mean([len(p) for p in predicted])), 4),
                "exact_match": round(exact / len(required), 4),
            }
        )
    return rows


def cross_validate(
    questions: Sequence[WorkloadQuestion], featurizer: str, tau: float, n_folds: int = N_FOLDS
) -> dict[str, Any]:
    """Grouped k-fold on dev: no template appears in both a training and a held-out fold."""
    groups = [q.template_id for q in questions]
    n_splits = min(n_folds, len(set(groups)))
    if n_splits < 2:
        raise ValueError("cross-validation needs at least two templates")
    specs = [q.spec for q in questions]
    predicted: list[set[str] | None] = [None] * len(specs)
    encoder = None
    for train_idx, held_idx in GroupKFold(n_splits=n_splits).split(specs, groups=groups):
        clf = SourceClassifier(featurizer=featurizer)
        clf._encoder = encoder  # reuse a loaded bge model across folds
        clf.fit([specs[i] for i in train_idx])
        encoder = clf._encoder
        probs = clf.predict_proba_batch([specs[i] for i in held_idx])
        for i, p in zip(held_idx, probs):
            predicted[i] = threshold(p, tau)
    report = set_metrics([p or set() for p in predicted], [s.required_sources for s in specs])
    report["folds"] = n_splits
    return report


def latency_stats(clf: SourceClassifier, questions: Sequence[WorkloadQuestion]) -> tuple[list[dict[str, float]], dict[str, float]]:
    """Single-question predict_proba on each question (as a router calls it), after one warm-up call."""
    clf.predict_proba(questions[0].spec)  # warm-up: first call pays one-off setup (model load for bge)
    probs, times = [], []
    for q in questions:
        probs.append(clf.predict_proba(q.spec))
        times.append(clf.last_latency_ms)
    return probs, {
        "n": len(times),
        "mean_ms": round(float(np.mean(times)), 4),
        "p95_ms": round(float(np.percentile(times, 95)), 4),
    }


def train_and_report(workload_dir: Path, featurizer: str, tau: float = 0.5) -> tuple[SourceClassifier, dict[str, Any]]:
    dev = read_jsonl(workload_dir / "dev.jsonl")
    test = read_jsonl(workload_dir / "test.jsonl")

    cv = cross_validate(dev, featurizer, tau)

    start = time.perf_counter()
    clf = SourceClassifier(featurizer=featurizer).fit([q.spec for q in dev])
    train_s = time.perf_counter() - start

    test_probs, latency = latency_stats(clf, test)
    required = [q.spec.required_sources for q in test]
    test_report = set_metrics([threshold(p, tau) for p in test_probs], required)

    report = {
        "featurizer": featurizer,
        "tau": tau,
        "n_dev": len(dev),
        "n_test": len(test),
        "train_seconds": round(train_s, 3),
        "dev_cv": cv,
        "test": test_report,
        "predict_latency_ms": latency,
        "tau_sweep_test": tau_sweep(test_probs, required),
    }
    return clf, report


def default_report_path(out: Path) -> Path:
    return out.with_name(f"{out.stem}_report.json")


# ---- routing on held-out phrasings ---------------------------------------------------------

PARAPHRASE_REPORT = Path("outputs/routing_paraphrase_report.json")
TOP_FAMILIES = 5


def p1_route(text: str) -> set[str]:
    """P1's source decision: the keyword rules, or notes when no rule fires (as RulePlanner does)."""
    return rule_sources(text) or {NOTES}


def p2_route(probs: dict[str, float], tau: float) -> set[str]:
    """P2's source decision: sources with p >= tau, or the most probable one when none pass."""
    return threshold(probs, tau) or {max(probs, key=lambda s: probs[s])}


def routing_metrics(predicted: Sequence[set[str]], items: Sequence[Paraphrase]) -> dict[str, Any]:
    """Per-source P/R/F1, exact match, missed-source rate, per-class rates and the worst phrasing families."""
    required = [p.question.spec.required_sources for p in items]
    missed = [bool(r - s) for s, r in zip(predicted, required)]
    report = set_metrics(predicted, required)
    report["missed_source_rate"] = round(sum(missed) / len(items), 4)
    report["avg_sources_predicted"] = round(float(np.mean([len(s) for s in predicted])), 4)

    per_class: dict[str, dict[str, Any]] = {}
    for qclass in sorted({p.question.spec.qclass for p in items}):
        idx = [i for i, p in enumerate(items) if p.question.spec.qclass == qclass]
        per_class[qclass] = {
            "n": len(idx),
            "exact_match": round(sum(set(predicted[i]) == set(required[i]) for i in idx) / len(idx), 4),
            "missed_source_rate": round(sum(missed[i] for i in idx) / len(idx), 4),
        }
    report["per_class"] = per_class

    families: dict[str, dict[str, Any]] = {}
    for i, p in enumerate(items):
        row = families.setdefault(
            p.family, {"family": p.family, "qclass": p.question.spec.qclass, "n": 0, "missed": 0, "missed_sources": {}}
        )
        row["n"] += 1
        if missed[i]:
            row["missed"] += 1
            for source in sorted(required[i] - predicted[i]):
                row["missed_sources"][source] = row["missed_sources"].get(source, 0) + 1
            row.setdefault(
                "example",
                {"text": p.question.spec.text, "predicted": sorted(predicted[i]), "required": sorted(required[i])},
            )
    for row in families.values():
        row["miss_rate"] = round(row["missed"] / row["n"], 4)
    ranked = sorted(families.values(), key=lambda r: (-r["missed"], -r["miss_rate"], r["family"]))
    report["top_miss_families"] = [r for r in ranked[:TOP_FAMILIES] if r["missed"] > 0]
    return report


def routed_tau_sweep(probs: Sequence[dict[str, float]], items: Sequence[Paraphrase]) -> list[dict[str, float]]:
    """P2 decisions at tau = 0.1 .. 0.9: missed-source rate, average sources predicted, exact match."""
    required = [p.question.spec.required_sources for p in items]
    rows = []
    for tau in TAUS:
        predicted = [p2_route(p, tau) for p in probs]
        rows.append(
            {
                "tau": tau,
                "missed_source_rate": round(sum(bool(r - s) for s, r in zip(predicted, required)) / len(items), 4),
                "avg_sources_predicted": round(float(np.mean([len(s) for s in predicted])), 4),
                "exact_match": round(sum(s == set(r) for s, r in zip(predicted, required)) / len(items), 4),
            }
        )
    return rows


def _latency(times_ms: Sequence[float]) -> dict[str, float]:
    return {
        "n": len(times_ms),
        "mean_ms": round(float(np.mean(times_ms)), 4),
        "p95_ms": round(float(np.percentile(times_ms, 95)), 4),
    }


def paraphrase_report(
    workload_dir: Path,
    featurizers: Sequence[str] = ("tfidf", "bge"),
    seed: int = 0,
    tau: float = 0.5,
    model_out: Path | None = None,
) -> dict[str, Any]:
    """Compare P1 and P2 routers on test-template questions in held-out phrasing families.

    P2 trains on the original dev questions plus dev paraphrases in the train
    families; every router is evaluated on test questions in the test families
    only. Each router's decision includes its own default for an empty
    prediction. A featurizer whose dependencies are missing is reported as skipped.
    """
    dev = read_jsonl(workload_dir / "dev.jsonl")
    test = read_jsonl(workload_dir / "test.jsonl")
    split = split_families(seed)
    train_paraphrases = paraphrase_all(dev, split["train"], seed)
    eval_items = paraphrase_all(test, split["test"], seed)
    train_specs = [q.spec for q in dev] + [p.question.spec for p in train_paraphrases]

    routers: dict[str, Any] = {}

    times, p1_predicted = [], []
    for p in eval_items:
        start = time.perf_counter()
        p1_predicted.append(p1_route(p.question.spec.text))
        times.append((time.perf_counter() - start) * 1000.0)
    routers["P1_rules"] = {**routing_metrics(p1_predicted, eval_items), "latency_ms": _latency(times), "tau_sweep": None}

    for featurizer in featurizers:
        name = f"P2_{featurizer}"
        try:
            clf = SourceClassifier(featurizer=featurizer).fit(train_specs)
            clf.predict_proba(eval_items[0].question.spec)  # warm-up
        except ImportError as exc:
            routers[name] = {"skipped": f"{type(exc).__name__}: {exc}"}
            continue
        if model_out is not None:
            clf.save(model_out.with_name(f"{model_out.stem}_paraphrase_{featurizer}.joblib"))
        probs, times = [], []
        for p in eval_items:
            probs.append(clf.predict_proba(p.question.spec))
            times.append(clf.last_latency_ms)
        routers[name] = {
            **routing_metrics([p2_route(pr, tau) for pr in probs], eval_items),
            "latency_ms": _latency(times),
            "tau_sweep": routed_tau_sweep(probs, eval_items),
        }

    return {
        "seed": seed,
        "tau": tau,
        "families": {name: sorted(names) for name, names in split.items()},
        "n_train": {"original_dev": len(dev), "dev_paraphrases": len(train_paraphrases), "total": len(train_specs)},
        "n_eval": {
            "total": len(eval_items),
            "by_class": {c: sum(p.question.spec.qclass == c for p in eval_items) for c in "ABCD"},
            "source_test_questions": len(test),
        },
        "decision_rule": (
            "P1: keyword rules, notes when no rule fires. "
            "P2: sources with p >= tau, the most probable source when none pass."
        ),
        "routers": routers,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workload-dir", type=Path, default=Path("data/workload"))
    parser.add_argument("--out", type=Path, default=Path("outputs/classifier.joblib"))
    parser.add_argument("--featurizer", choices=["tfidf", "bge"], default="tfidf")
    parser.add_argument("--tau", type=float, default=0.5, help="threshold for the per-source and exact-match metrics")
    parser.add_argument("--report", type=Path, default=None, help="report path (default: <out stem>_report.json)")
    parser.add_argument(
        "--paraphrase",
        action="store_true",
        help="compare P1 and P2 on held-out phrasing families (report: outputs/routing_paraphrase_report.json)",
    )
    parser.add_argument(
        "--featurizers", nargs="+", choices=["tfidf", "bge"], default=["tfidf", "bge"],
        help="P2 featurizers to compare with --paraphrase",
    )
    parser.add_argument("--seed", type=int, default=0, help="paraphrase and family-split seed (with --paraphrase)")
    args = parser.parse_args(argv)

    if args.paraphrase:
        report = paraphrase_report(args.workload_dir, args.featurizers, args.seed, args.tau, model_out=args.out)
        report_path = args.report or PARAPHRASE_REPORT
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        print(f"report: {report_path}")
        return

    clf, report = train_and_report(args.workload_dir, args.featurizer, args.tau)
    clf.save(args.out)
    report_path = args.report or default_report_path(args.out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"model: {args.out}\nreport: {report_path}")


if __name__ == "__main__":
    main()
