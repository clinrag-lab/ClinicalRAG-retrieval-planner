"""Build the question workload: candidates per class, gold filtering, capping, dev/test split.

Usage:
    python -m clinrag.workload.build --tables-dir data/mimic-iv-demo --notes-dir data/generated \\
        --out-dir data/workload --seed 0
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clinrag.workload.labs import LABS
from clinrag.workload.schema import WorkloadQuestion, write_jsonl
from clinrag.workload.templates import (
    A_KINDS,
    WorkloadData,
    class_a,
    class_b,
    class_c,
    class_d,
    gold_is_usable,
    per_template_quota,
)

MAX_PER_CLASS = 100
DEV_FRACTION = 0.6
CLASSES = ("A", "B", "C", "D")
SELECTIVITY_BINS: tuple[tuple[float, float], ...] = (
    (0.0, 0.05),
    (0.05, 0.10),
    (0.10, 0.25),
    (0.25, 0.50),
    (0.50, 0.75),
    (0.75, 1.0),
)


@dataclass(frozen=True)
class Workload:
    dev: list[WorkloadQuestion]
    test: list[WorkloadQuestion]
    summary: dict[str, Any]


def build_workload(
    data: WorkloadData,
    seed: int,
    max_per_class: int = MAX_PER_CLASS,
    dev_fraction: float = DEV_FRACTION,
) -> Workload:
    n = data.n_admissions
    a_quota = per_template_quota(max_per_class, len(LABS) * len(A_KINDS))
    candidates = {
        "A": class_a(data, a_quota, random.Random(f"{seed}:A")),
        "B": class_b(data),
        "C": class_c(data),
        "D": class_d(data),
    }

    dev: list[WorkloadQuestion] = []
    test: list[WorkloadQuestion] = []
    class_summary: dict[str, dict[str, int]] = {}
    for qclass in CLASSES:
        pool = candidates[qclass]
        usable = _unique_by_text([q for q in pool if gold_is_usable(q, n)])
        kept = _cap(usable, max_per_class, random.Random(f"{seed}:cap:{qclass}"))
        kept = [q.with_qid(f"{qclass}{i:03d}") for i, q in enumerate(kept, start=1)]
        dev_templates = split_templates(
            [q.template_id for q in kept], dev_fraction, random.Random(f"{seed}:split:{qclass}")
        )
        class_dev = [q for q in kept if q.template_id in dev_templates]
        class_test = [q for q in kept if q.template_id not in dev_templates]
        dev += class_dev
        test += class_test
        class_summary[qclass] = {
            "candidates": len(pool),
            "skipped_by_gold_rule": len(pool) - len(usable),
            "kept": len(kept),
            "dev": len(class_dev),
            "test": len(class_test),
            "templates_dev": len({q.template_id for q in class_dev}),
            "templates_test": len({q.template_id for q in class_test}),
        }

    summary = {
        "seed": seed,
        "n_admissions": n,
        "max_per_class": max_per_class,
        "dev_fraction": dev_fraction,
        "classes": class_summary,
        "total": {"dev": len(dev), "test": len(test)},
        "selectivity": {c: spread([q.gold["selectivity"] for q in dev + test if q.spec.qclass == c]) for c in ("C", "D")},
        "gold_fraction": {
            c: spread([len(q.gold["answer"]) / n for q in dev + test if q.spec.qclass == c]) for c in ("B", "C", "D")
        },
    }
    return Workload(dev, test, summary)


def split_templates(template_ids: Sequence[str], dev_fraction: float, rng: random.Random) -> set[str]:
    """Pick the dev templates; the rest are test. Each split gets at least one when there are two or more."""
    unique = sorted(set(template_ids))
    rng.shuffle(unique)
    n_dev = round(dev_fraction * len(unique))
    if len(unique) >= 2:
        n_dev = min(max(n_dev, 1), len(unique) - 1)
    return set(unique[:n_dev])


def spread(values: Sequence[float]) -> dict[str, Any]:
    """min / quartiles / max and a histogram over SELECTIVITY_BINS."""
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    q = statistics.quantiles(ordered, n=4, method="inclusive") if len(ordered) > 1 else [ordered[0]] * 3
    bins = {}
    for lo, hi in SELECTIVITY_BINS:
        label = f"{lo:.2f}-{hi:.2f}"
        bins[label] = sum(1 for v in ordered if lo <= v < hi or (hi == 1.0 and v == 1.0))
    return {
        "n": len(ordered),
        "min": round(ordered[0], 4),
        "p25": round(q[0], 4),
        "median": round(q[1], 4),
        "p75": round(q[2], 4),
        "max": round(ordered[-1], 4),
        "bins": bins,
    }


def _unique_by_text(questions: list[WorkloadQuestion]) -> list[WorkloadQuestion]:
    seen: set[str] = set()
    out = []
    for q in questions:
        if q.spec.text not in seen:
            seen.add(q.spec.text)
            out.append(q)
    return out


def _cap(questions: list[WorkloadQuestion], limit: int, rng: random.Random) -> list[WorkloadQuestion]:
    """Keep at most limit questions, sampled at random but kept in their original order."""
    if len(questions) <= limit:
        return questions
    keep = set(rng.sample(range(len(questions)), limit))
    return [q for i, q in enumerate(questions) if i in keep]


def write_workload(workload: Workload, out_dir: str | Path) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "dev.jsonl", workload.dev)
    write_jsonl(out / "test.jsonl", workload.test)
    (out / "summary.json").write_text(json.dumps(workload.summary, indent=2) + "\n", encoding="utf-8")
    return [out / "dev.jsonl", out / "test.jsonl", out / "summary.json"]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tables-dir", required=True, help="hosp tables folder, or a folder containing hosp/")
    parser.add_argument("--notes-dir", required=True, help="folder with note_concepts.csv.gz from notes_gen")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    workload = build_workload(WorkloadData.load(args.tables_dir, args.notes_dir), args.seed)
    for path in write_workload(workload, args.out_dir):
        print(f"wrote {path}")
    print(json.dumps(workload.summary["total"]))


if __name__ == "__main__":
    main()
