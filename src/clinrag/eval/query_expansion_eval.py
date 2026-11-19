
import argparse
import csv
import json
from pathlib import Path
from statistics import mean

from clinrag.sources.faiss_notes import FaissNotesSource


def load_questions(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            if q["qclass"] == "B" and isinstance(q["gold"]["answer"], list):
                yield q


def ranked_admissions(result):
    """Get unique admission IDs in retrieval order."""
    seen = set()
    ids = []

    for item in result.items:
        hadm_id = int(item["hadm_id"])

        if hadm_id not in seen:
            seen.add(hadm_id)
            ids.append(hadm_id)

    return ids


def reciprocal_rank_fusion(rankings, k=60):
    """Combine multiple ranked admission lists."""
    scores = {}

    for ranking in rankings:
        for rank, hadm_id in enumerate(ranking, start=1):
            scores[hadm_id] = (
                scores.get(hadm_id, 0.0)
                + 1.0 / (k + rank)
            )

    return sorted(
        scores,
        key=lambda hadm_id: (-scores[hadm_id], hadm_id),
    )


def evaluate(notes, questions, top_k, candidate_k):
    rows = []

    for q in questions:
        semantic_query = q["note_query"]

        # This field was saved by semantic_build.py.
        original_qid = q["gold"]["original_qid"]

        # Map the original question to its clinical keyword.
        keyword_map = {
            "B001": "dialysis",
            "B002": "chest pain",
            "B003": "shortness of breath",
            "B004": "anticoagulation",
            "B005": "insulin",
            "B006": "sepsis",
        }

        keyword = keyword_map[original_qid]

        semantic_result = notes.search(
            semantic_query,
            k=candidate_k,
        )

        keyword_result = notes.search(
            keyword,
            k=candidate_k,
        )

        semantic_ids = ranked_admissions(semantic_result)
        keyword_ids = ranked_admissions(keyword_result)

        fused_ids = reciprocal_rank_fusion([
            semantic_ids,
            keyword_ids,
        ])[:top_k]

        gold = set(int(x) for x in q["gold"]["answer"])
        correct = gold.intersection(fused_ids)

        precision = (
            len(correct) / len(fused_ids)
            if fused_ids else 0.0
        )
        recall = (
            len(correct) / len(gold)
            if gold else 1.0
        )

        hit = int(bool(correct))

        mrr = 0.0
        for rank, hadm_id in enumerate(fused_ids, 1):
            if hadm_id in gold:
                mrr = 1.0 / rank
                break

        latency = (
            semantic_result.latency_ms
            + keyword_result.latency_ms
        )

        rows.append({
            "qid": q["qid"],
            "keyword": keyword,
            "precision": precision,
            "recall": recall,
            "hit": hit,
            "mrr": mrr,
            "latency_ms": latency,
            "retrieved": len(fused_ids),
            "correct": len(correct),
        })

        print(
            f"{q['qid']} | "
            f"Precision: {precision:.3f} | "
            f"Recall: {recall:.3f} | "
            f"Latency: {latency:.2f} ms"
        )

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workload",
        default="data/workload/semantic_dev.jsonl",
    )
    parser.add_argument(
        "--index-dir",
        default="data/faiss_index",
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=30)
    parser.add_argument(
        "--out",
        default="data/eval/query_expansion_k10.csv",
    )
    args = parser.parse_args()

    if args.k < 1 or args.candidate_k < args.k:
        parser.error("--candidate-k must be >= --k >= 1")

    notes = FaissNotesSource(args.index_dir)
    questions = list(load_questions(args.workload))

    rows = evaluate(
        notes,
        questions,
        args.k,
        args.candidate_k,
    )

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "qid", "keyword", "precision", "recall",
            "hit", "mrr", "latency_ms", "retrieved", "correct",
        ])
        writer.writeheader()
        writer.writerows(rows)

    print("\n--- Query Expansion Summary ---")
    print("Questions:", len(rows))

    for metric in ["precision", "recall", "hit", "mrr", "latency_ms"]:
        print(
            f"Average {metric}: "
            f"{mean(row[metric] for row in rows):.4f}"
        )

    print("Saved to:", output)


if __name__ == "__main__":
    main()
