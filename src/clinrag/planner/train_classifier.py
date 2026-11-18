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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workload-dir", type=Path, default=Path("data/workload"))
    parser.add_argument("--out", type=Path, default=Path("outputs/classifier.joblib"))
    parser.add_argument("--featurizer", choices=["tfidf", "bge"], default="tfidf")
    parser.add_argument("--tau", type=float, default=0.5, help="threshold for the per-source and exact-match metrics")
    parser.add_argument("--report", type=Path, default=None, help="report path (default: <out stem>_report.json)")
    args = parser.parse_args(argv)

    clf, report = train_and_report(args.workload_dir, args.featurizer, args.tau)
    clf.save(args.out)
    report_path = args.report or default_report_path(args.out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"model: {args.out}\nreport: {report_path}")


if __name__ == "__main__":
    main()
