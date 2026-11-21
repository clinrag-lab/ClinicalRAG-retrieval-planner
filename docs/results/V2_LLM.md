# v2 results: answer accuracy with Llama-3.1-8B-Instruct

Frozen as tag `results-demo-v2`. Same open MIMIC-IV demo (275 admissions), generated notes, planner
config (`config/results-demo-v1.yaml`) and router set as `results-demo-v1`, with an LLM answer step.

## What changed from results-demo-v1

- **Workload v2.** Diya's ICD-9/ICD-10 fix (commit 202698a) adds ICD-9 249 to the diabetes block, which
  changes 3 generated notes and the gold answers of 27 questions. Notes and workload are regenerated
  into `data/generated_v2` and `data/workload_v2` (seed 0); the test files' SHA-256 are below.
- **SQL pushdown for latest and max.** Class A "latest" and "max" questions are answered by the
  relational source in SQL, returning one row instead of every matching lab row.
- **LLM answer step.** Llama-3.1-8B-Instruct in bf16 on one NVIDIA A100-SXM4-80GB (Sol, Slurm job
  2110541, torch 2.5.1+cu121), behind `--answerer llama --llm-dtype bf16`.
- **Shared context budget.** Evidence is sent to the LLM in plan order up to 8,192 tokens, the same for
  every router; `truncated_rate` is the share of questions cut at that budget.
- **Scoring.** Answers are parsed against the type the question's template expects. A null or
  unparsed answer scores as wrong (0); `parse_failure_rate` reports how often that happened.

Retrieval metrics (recall, tokens, latency) are computed as in v1. Only aggregate tables, the
commands and file checksums are below: no per-question rows and no note text.

Reproduce: `scripts/sol/README.md` (Sol setup and job), commands also in `docs/experiments.md`.

---

# outputs/runs/v2/SUMMARY.md, as written by the job

- commit `68f9a89a5bcabd5551fe881006e8926cdd53beb7`, written 2026-11-20 12:34, Slurm job 2110541
- answerer: llama-3.1-8b-instruct
- recall: value row (A), gold admission's note (per-admission B), normalized recall@k (B list, C, D)
- tokens: evidence tokens retrieved; latency and overhead in ms (median over repeats, mean over questions)
- answer_accuracy: mean answer score; a null or unparseable answer scores 0
- parse_failure_rate: share of answers with no value of the expected type
- truncated_rate: share of questions whose evidence was cut to max_context_tokens

## Template test split

Questions: `data/workload_v2/test.jsonl` (171, sha256 `9d11e06aacb6b65bfdf4f64b2368e4aded83c4b139c459b2e0c0c8ac8eaaf11d`)

### All classes

| router | qclass | n | mean_recall | mean_tokens | mean_latency_ms | mean_overhead_ms | answer_accuracy | parse_failure_rate | truncated_rate |
|---|---|---|---|---|---|---|---|---|---|
| oracle | all | 171 | 0.909 | 693.105 | 2.892 | 2.391 | 0.672 | 0.000 | 0.000 |
| p1 | all | 171 | 0.913 | 700.825 | 3.336 | 0.061 | 0.672 | 0.000 | 0.000 |
| p2_tfidf | all | 171 | 0.909 | 693.105 | 3.285 | 4.031 | 0.672 | 0.000 | 0.000 |
| query_all | all | 171 | 0.628 | 3201.643 | 3.200 | 2.413 | 0.316 | 0.000 | 0.123 |
| always_filter_first | all | 171 | 0.913 | 700.825 | 2.877 | 2.336 | 0.672 | 0.000 | 0.000 |
| always_search_first | all | 171 | 0.858 | 605.626 | 3.096 | 2.511 | 0.649 | 0.000 | 0.000 |

### Per class

| router | qclass | n | mean_recall | mean_tokens | mean_latency_ms | mean_overhead_ms | answer_accuracy | parse_failure_rate | truncated_rate |
|---|---|---|---|---|---|---|---|---|---|
| oracle | A | 43 | 1.000 | 11.000 | 1.660 | 0.032 | 1.000 | 0.000 | 0.000 |
| oracle | B | 46 | 0.978 | 323.957 | 0.673 | 0.029 | 0.963 | 0.000 | 0.000 |
| oracle | C | 42 | 0.838 | 1253.119 | 5.357 | 4.592 | 0.366 | 0.000 | 0.000 |
| oracle | D | 40 | 0.805 | 1262.875 | 4.178 | 5.330 | 0.306 | 0.000 | 0.000 |
| p1 | A | 43 | 1.000 | 11.000 | 2.209 | 0.055 | 1.000 | 0.000 | 0.000 |
| p1 | B | 46 | 0.978 | 323.957 | 0.667 | 0.043 | 0.963 | 0.000 | 0.000 |
| p1 | C | 42 | 0.852 | 1278.524 | 6.142 | 0.071 | 0.366 | 0.000 | 0.000 |
| p1 | D | 40 | 0.810 | 1269.200 | 4.669 | 0.076 | 0.306 | 0.000 | 0.000 |
| p2_tfidf | A | 43 | 1.000 | 11.000 | 1.701 | 1.345 | 1.000 | 0.000 | 0.000 |
| p2_tfidf | B | 46 | 0.978 | 323.957 | 0.641 | 1.257 | 0.963 | 0.000 | 0.000 |
| p2_tfidf | C | 42 | 0.838 | 1253.119 | 5.895 | 6.302 | 0.366 | 0.000 | 0.000 |
| p2_tfidf | D | 40 | 0.805 | 1262.875 | 5.289 | 7.722 | 0.306 | 0.000 | 0.000 |
| query_all | A | 43 | 1.000 | 1479.000 | 2.571 | 0.032 | 0.070 | 0.000 | 0.000 |
| query_all | B | 46 | 0.978 | 323.957 | 0.650 | 0.029 | 0.963 | 0.000 | 0.000 |
| query_all | C | 42 | 0.233 | 7508.333 | 5.560 | 4.621 | 0.093 | 0.000 | 0.333 |
| query_all | D | 40 | 0.240 | 3840.800 | 4.333 | 5.397 | 0.070 | 0.000 | 0.175 |
| always_filter_first | A | 43 | 1.000 | 11.000 | 1.643 | 0.031 | 1.000 | 0.000 | 0.000 |
| always_filter_first | B | 46 | 0.978 | 323.957 | 0.661 | 0.028 | 0.963 | 0.000 | 0.000 |
| always_filter_first | C | 42 | 0.852 | 1278.524 | 5.274 | 4.532 | 0.366 | 0.000 | 0.000 |
| always_filter_first | D | 40 | 0.810 | 1269.200 | 4.237 | 5.160 | 0.306 | 0.000 | 0.000 |
| always_search_first | A | 43 | 1.000 | 11.000 | 1.616 | 0.031 | 1.000 | 0.000 | 0.000 |
| always_search_first | B | 46 | 0.978 | 323.957 | 0.652 | 0.028 | 0.963 | 0.000 | 0.000 |
| always_search_first | C | 42 | 0.748 | 1091.643 | 5.485 | 4.389 | 0.333 | 0.000 | 0.000 |
| always_search_first | D | 40 | 0.681 | 1058.450 | 4.992 | 6.059 | 0.242 | 0.000 | 0.000 |

### Commands

```
python -m clinrag.eval.runner --sources reference --tables-dir data/mimic-iv-demo --notes-dir data/generated_v2 --answerer llama --llm-dtype bf16 --config config/results-demo-v1.yaml --split test --no-analysis --routers oracle --out outputs/runs/v2/template-test/parts/oracle --workload data/workload_v2
python -m clinrag.eval.runner --sources reference --tables-dir data/mimic-iv-demo --notes-dir data/generated_v2 --answerer llama --llm-dtype bf16 --config config/results-demo-v1.yaml --split test --no-analysis --routers p1 --out outputs/runs/v2/template-test/parts/p1 --workload data/workload_v2
python -m clinrag.eval.runner --sources reference --tables-dir data/mimic-iv-demo --notes-dir data/generated_v2 --answerer llama --llm-dtype bf16 --config config/results-demo-v1.yaml --split test --no-analysis --routers p2_tfidf --out outputs/runs/v2/template-test/parts/p2_tfidf --workload data/workload_v2
python -m clinrag.eval.runner --sources reference --tables-dir data/mimic-iv-demo --notes-dir data/generated_v2 --answerer llama --llm-dtype bf16 --config config/results-demo-v1.yaml --split test --no-analysis --routers query_all --out outputs/runs/v2/template-test/parts/query_all --workload data/workload_v2
python -m clinrag.eval.runner --sources reference --tables-dir data/mimic-iv-demo --notes-dir data/generated_v2 --answerer llama --llm-dtype bf16 --config config/results-demo-v1.yaml --split test --no-analysis --routers always_filter_first --out outputs/runs/v2/template-test/parts/always_filter_first --workload data/workload_v2
python -m clinrag.eval.runner --sources reference --tables-dir data/mimic-iv-demo --notes-dir data/generated_v2 --answerer llama --llm-dtype bf16 --config config/results-demo-v1.yaml --split test --no-analysis --routers always_search_first --out outputs/runs/v2/template-test/parts/always_search_first --workload data/workload_v2
```

## Paraphrase test split (P2 trained on v2 dev + train-family paraphrases)

Questions: `data/workload_v2/paraphrase/test.jsonl` (684, sha256 `9c2bd0f4a67c9013d0010cabdb8d540059cca4a4b85a6dba65356747ed0fe654`)

### All classes

| router | qclass | n | mean_recall | mean_tokens | mean_latency_ms | mean_overhead_ms | answer_accuracy | parse_failure_rate | truncated_rate |
|---|---|---|---|---|---|---|---|---|---|
| p1 | all | 684 | 0.467 | 1742.034 | 1.907 | 0.048 | 0.437 | 0.000 | 0.057 |
| p2_tfidf | all | 684 | 0.875 | 741.098 | 3.699 | 4.633 | 0.614 | 0.000 | 0.000 |
| oracle | all | 684 | 0.909 | 693.105 | 3.845 | 3.051 | 0.648 | 0.000 | 0.000 |
| query_all | all | 684 | 0.628 | 3173.380 | 4.275 | 3.250 | 0.409 | 0.000 | 0.123 |

### Per class

| router | qclass | n | mean_recall | mean_tokens | mean_latency_ms | mean_overhead_ms | answer_accuracy | parse_failure_rate | truncated_rate |
|---|---|---|---|---|---|---|---|---|---|
| p1 | A | 172 | 0.523 | 664.680 | 1.518 | 0.051 | 0.523 | 0.000 | 0.000 |
| p1 | B | 184 | 0.978 | 323.957 | 0.659 | 0.041 | 0.924 | 0.000 | 0.000 |
| p1 | C | 168 | 0.064 | 4830.929 | 4.542 | 0.053 | 0.160 | 0.000 | 0.232 |
| p1 | D | 160 | 0.241 | 1287.638 | 0.993 | 0.047 | 0.075 | 0.000 | 0.000 |
| p2_tfidf | A | 172 | 0.866 | 201.855 | 1.674 | 1.782 | 0.866 | 0.000 | 0.000 |
| p2_tfidf | B | 184 | 0.978 | 323.957 | 0.666 | 1.289 | 0.924 | 0.000 | 0.000 |
| p2_tfidf | C | 168 | 0.838 | 1253.119 | 7.085 | 7.486 | 0.328 | 0.000 | 0.000 |
| p2_tfidf | D | 160 | 0.805 | 1262.875 | 5.807 | 8.548 | 0.288 | 0.000 | 0.000 |
| oracle | A | 172 | 1.000 | 11.000 | 2.518 | 0.047 | 1.000 | 0.000 | 0.000 |
| oracle | B | 184 | 0.978 | 323.957 | 0.774 | 0.041 | 0.924 | 0.000 | 0.000 |
| oracle | C | 168 | 0.838 | 1253.119 | 6.745 | 5.705 | 0.328 | 0.000 | 0.000 |
| oracle | D | 160 | 0.805 | 1262.875 | 5.757 | 6.956 | 0.288 | 0.000 | 0.000 |
| query_all | A | 172 | 1.000 | 1366.605 | 3.528 | 0.049 | 0.477 | 0.000 | 0.000 |
| query_all | B | 184 | 0.978 | 323.957 | 0.649 | 0.031 | 0.924 | 0.000 | 0.000 |
| query_all | C | 168 | 0.233 | 7508.333 | 7.385 | 6.322 | 0.097 | 0.000 | 0.333 |
| query_all | D | 160 | 0.240 | 3840.800 | 5.983 | 7.169 | 0.072 | 0.000 | 0.175 |

### Commands

```
python -m clinrag.eval.runner --sources reference --tables-dir data/mimic-iv-demo --notes-dir data/generated_v2 --answerer llama --llm-dtype bf16 --config config/results-demo-v1.yaml --split test --no-analysis --routers p1 --out outputs/runs/v2/paraphrase-test/parts/p1 --workload data/workload_v2/paraphrase --train-workload data/workload_v2 --p2-train dev+paraphrases --p2-model-dir outputs/runs/v2/paraphrase-test/models
python -m clinrag.eval.runner --sources reference --tables-dir data/mimic-iv-demo --notes-dir data/generated_v2 --answerer llama --llm-dtype bf16 --config config/results-demo-v1.yaml --split test --no-analysis --routers p2_tfidf --out outputs/runs/v2/paraphrase-test/parts/p2_tfidf --workload data/workload_v2/paraphrase --train-workload data/workload_v2 --p2-train dev+paraphrases --p2-model-dir outputs/runs/v2/paraphrase-test/models
python -m clinrag.eval.runner --sources reference --tables-dir data/mimic-iv-demo --notes-dir data/generated_v2 --answerer llama --llm-dtype bf16 --config config/results-demo-v1.yaml --split test --no-analysis --routers oracle --out outputs/runs/v2/paraphrase-test/parts/oracle --workload data/workload_v2/paraphrase --train-workload data/workload_v2 --p2-train dev+paraphrases --p2-model-dir outputs/runs/v2/paraphrase-test/models
python -m clinrag.eval.runner --sources reference --tables-dir data/mimic-iv-demo --notes-dir data/generated_v2 --answerer llama --llm-dtype bf16 --config config/results-demo-v1.yaml --split test --no-analysis --routers query_all --out outputs/runs/v2/paraphrase-test/parts/query_all --workload data/workload_v2/paraphrase --train-workload data/workload_v2 --p2-train dev+paraphrases --p2-model-dir outputs/runs/v2/paraphrase-test/models
```
