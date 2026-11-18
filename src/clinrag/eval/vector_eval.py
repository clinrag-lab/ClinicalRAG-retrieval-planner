
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from clinrag.sources.faiss_notes import FaissNotesSource


def load_questions(path):
    """Load the evaluation questions from a JSONL file."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate_question(notes, question, k):
    """Run a vector search and compare it with the expected results."""

    query = question["note_query"]
    gold_answer = question["gold"]["answer"]

    # List questions have multiple correct admission IDs.
    if isinstance(gold_answer, list):
        question_type = "list"

        gold_ids = set(int(x) for x in gold_answer)
        result = notes.search(query, k=k)

        # FAISS returns chunks, but the gold answers are admission IDs.
        # We compare unique admissions so repeated chunks do not count twice.
        retrieved_ids = set(int(x) for x in result.ids)
        correct = retrieved_ids & gold_ids

        precision = (
            len(correct) / len(retrieved_ids)
            if retrieved_ids else 0.0
        )

        recall = (
            len(correct) / len(gold_ids)
            if gold_ids else 0.0
        )

        hit = 1 if correct else 0

        # Find the rank of the first relevant admission.
        reciprocal_rank = 0.0

        for rank, item in enumerate(result.items, start=1):
            hadm_id = int(item["hadm_id"])

            if hadm_id in gold_ids:
                reciprocal_rank = 1.0 / rank
                break

        gold_count = len(gold_ids)
        retrieved_count = len(retrieved_ids)
        correct_count = len(correct)

        # These metrics are only for the admission-list questions.
        evidence_found = None
        expected_answer = None

    else:
        # These questions ask yes/no about one specific admission.
        question_type = "admission"

        hadm_id = int(question["note_hadm_id"])
        expected_answer = int(gold_answer)

        # Only search notes belonging to this admission.
        result = notes.search(
            query,
            k=k,
            hadm_ids=[hadm_id]
        )

        retrieved_ids = set(int(x) for x in result.ids)

        # Check if we found notes from the requested admission.
        evidence_found = int(
            hadm_id in retrieved_ids and len(result.items) > 0
        )

        gold_count = 1
        retrieved_count = len(retrieved_ids)
        correct_count = evidence_found

        # Precision/recall/MRR for a list of admissions don't
        # apply to these yes/no questions.
        precision = None
        recall = None
        hit = None
        reciprocal_rank = None

    return {
        "qid": question["qid"],
        "qclass": question["qclass"],
        "question_type": question_type,
        "query": query,
        "k": k,
        "gold_count": gold_count,
        "retrieved_count": retrieved_count,
        "correct_count": correct_count,
        "precision": precision,
        "recall": recall,
        "hit": hit,
        "mrr": reciprocal_rank,
        "evidence_found": evidence_found,
        "expected_answer": expected_answer,
        "latency_ms": result.latency_ms,
        "tokens": result.tokens,
    }


def print_average(results, metric):
    """Print the average for a metric, skipping missing values."""

    values = [
        row[metric]
        for row in results
        if row[metric] is not None
    ]

    if values:
        average = statistics.mean(values)
        print(f"Average {metric}: {average:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workload",
        default="data/workload/dev.jsonl",
    )
    parser.add_argument(
        "--index-dir",
        default="data/faiss_index",
    )
    parser.add_argument(
        "--out",
        default="data/eval/vector_results.csv",
    )
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    if args.k <= 0:
        parser.error("--k must be greater than zero")

    questions = load_questions(args.workload)

    # Class B only needs the vector database.
    questions = [
        q for q in questions
        if q["qclass"] == "B"
        and q.get("note_query")
    ]

    if not questions:
        raise ValueError("No Class B questions found")

    notes = FaissNotesSource(args.index_dir)

    results = []

    for question in questions:
        row = evaluate_question(notes, question, args.k)
        results.append(row)

        if row["question_type"] == "list":
            print(
                f"{row['qid']} | "
                f"Precision: {row['precision']:.3f} | "
                f"Recall: {row['recall']:.3f} | "
                f"Latency: {row['latency_ms']:.2f} ms"
            )
        else:
            print(
                f"{row['qid']} | "
                f"Evidence found: {row['evidence_found']} | "
                f"Expected answer: {row['expected_answer']} | "
                f"Latency: {row['latency_ms']:.2f} ms"
            )

    # Save the results so we can compare them later.
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=results[0].keys()
        )
        writer.writeheader()
        writer.writerows(results)

    # Separate the two question types when calculating averages.
    list_results = [
        r for r in results
        if r["question_type"] == "list"
    ]

    admission_results = [
        r for r in results
        if r["question_type"] == "admission"
    ]

    print("\n--- Vector Evaluation Summary ---")
    print("Total questions:", len(results))

    print("\n--- List Questions ---")
    print("Questions:", len(list_results))

    for metric in [
        "precision",
        "recall",
        "hit",
        "mrr",
        "latency_ms",
        "tokens",
    ]:
        print_average(list_results, metric)

    print("\n--- Admission Questions ---")
    print("Questions:", len(admission_results))

    for metric in [
        "evidence_found",
        "latency_ms",
        "tokens",
    ]:
        print_average(admission_results, metric)

    print("\n--- Overall Performance ---")
    print_average(results, "latency_ms")
    print_average(results, "tokens")

    print("\nResults saved to:", output)


if __name__ == "__main__":
    main()
