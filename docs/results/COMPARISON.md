# FAISS (BGE) vs TF-IDF notes, with and without the query-embedding cache

Three runs, all with `s = 0.5`, `beta = 1.0` and the same planner config:

| column | what | where | power |
|---|---|---|---|
| TF-IDF | `results-demo-v1` (5acdd60), `TfidfNotesSource`, one chunk per note | `outputs/runs/final-demo` | (reference) |
| FAISS before | `--sources faiss` at 0445b80, no query cache; this morning | `outputs/runs/faiss-demo` | template: AC; **paraphrase: BATTERY** |
| FAISS cache | `--sources faiss` at d8b0b4d, `CachingEncoder`, cleared before every question execution | `outputs/runs/faiss-demo/cached` | AC for every part |

FAISS = `DemoRelationalSource` + `FaissNotesSource` (Dylan's, unchanged) + `IcdGraph`; 275 notes -> 474 chunks (180 words, 30 overlap),
`BAAI/bge-base-en-v1.5`, `IndexFlatIP`. Both FAISS runs use `faiss-demo/config/calibrated_config.yaml` (calibrated on dev without the cache:
`note_search_fixed_ms` 28.87, `min_similarity` 0.583; the cache run was not recalibrated). One router per process; latency and overhead
are the median of 3 repeats per question, then the mean / p95 over questions.

**Power and machine state.** The cache run logged power at the start and end of every part (`cached/run.log`) and every 20 s
(`cached/power_poll.log`, 63 samples, 12:43-13:05): all AC. "FAISS before" paraphrase parts ran mostly on battery (BGE encode ~100 ms vs
~28 ms on AC). Even on AC the laptop's speed varied between processes this afternoon: a BGE encode benchmark gave a median of 50 ms at
13:06 (28 ms at this morning's calibration), and in the cache run the class-B note search (no probe, so the cache cannot affect it) took a
median of 19.6 ms in the oracle process and 63 ms in the p1 process. **Latency columns from different runs therefore mix the cache's
effect with machine speed; the paired A/B section isolates the cache.** Recall, tokens and fallbacks are identical per question in both
FAISS runs (1,026 template and 2,736 paraphrase rows compared), so they are shown once.

## Template test (171 questions)

| router | n | recall TF-IDF | recall FAISS | tokens TF-IDF | tokens FAISS | fallback TF-IDF | fallback FAISS | mean ms TF-IDF | mean ms FAISS before | mean ms FAISS cache | p95 ms TF-IDF | p95 ms FAISS before | p95 ms FAISS cache | overhead ms TF-IDF | overhead ms FAISS before | overhead ms FAISS cache |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| oracle | 171 | 0.906 | 0.914 | 719 | 840 | 0.000 | 0.070 | 6.95 | 31.62 | 11.24 | 16.43 | 72.26 | 21.36 | 6.63 | 19.57 | 28.22 |
| p1 | 171 | 0.911 | 0.916 | 726 | 843 | 0.000 | 0.070 | 10.19 | 29.58 | 58.54 | 23.04 | 68.27 | 97.03 | 0.16 | 0.12 | 0.10 |
| p2_tfidf | 171 | 0.906 | 0.914 | 719 | 840 | 0.000 | 0.070 | 7.87 | 37.37 | 23.98 | 19.68 | 82.73 | 67.89 | 8.97 | 22.31 | 37.48 |
| query_all | 171 | 0.625 | 0.678 | 3233 | 3088 | 0.000 | 0.000 | 6.90 | 37.24 | 42.58 | 15.28 | 50.38 | 84.15 | 6.19 | 18.72 | 35.62 |
| always_filter_first | 171 | 0.911 | 0.916 | 726 | 843 | 0.000 | 0.070 | 6.99 | 28.96 | 15.46 | 16.08 | 66.06 | 36.02 | 6.31 | 17.62 | 21.23 |
| always_search_first | 171 | 0.855 | 0.877 | 632 | 777 | 0.000 | 0.070 | 7.09 | 28.56 | 15.21 | 15.02 | 65.96 | 35.80 | 6.30 | 18.69 | 22.07 |

Recall and tokens are the same FAISS numbers as before (the cache changes no results); the timing columns moved in both directions
between runs (oracle, p2_tfidf and the fixed-order routers down 36-64%, p1 and query_all up, overhead up for every probing router) because this
afternoon's processes ran at different speeds, which is why the cache's effect is measured with the paired A/B below rather than read off this table.

## Paraphrase end to end (684 held-out test paraphrases)

| router | n | recall TF-IDF | recall FAISS | tokens TF-IDF | tokens FAISS | fallback TF-IDF | fallback FAISS | mean ms TF-IDF | mean ms FAISS before (BATTERY) | mean ms FAISS cache | p95 ms TF-IDF | p95 ms FAISS before (BATTERY) | p95 ms FAISS cache | overhead ms TF-IDF | overhead ms FAISS before (BATTERY) | overhead ms FAISS cache |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| p1 | 684 | 0.464 | 0.623 | 1759 | 1960 | 0.000 | 0.155 | 3.56 | 76.88 | 38.89 | 11.60 | 228.52 | 77.72 | 0.10 | 0.20 | 0.11 |
| p2_tfidf | 684 | 0.873 | 0.914 | 764 | 917 | 0.000 | 0.104 | 7.75 | 109.54 | 20.79 | 19.34 | 233.67 | 62.83 | 9.18 | 66.40 | 36.93 |
| oracle | 684 | 0.906 | 0.914 | 719 | 840 | 0.000 | 0.070 | 5.14 | 91.71 | 19.81 | 13.44 | 211.04 | 52.53 | 4.90 | 53.64 | 27.48 |
| query_all | 684 | 0.625 | 0.678 | 3205 | 3091 | 0.000 | 0.000 | 7.30 | 100.95 | 35.99 | 17.88 | 130.32 | 71.49 | 5.99 | 51.66 | 32.68 |

The "before" latency columns ran on battery and are not comparable; on AC with the cache, FAISS end-to-end latency is 20-39 ms mean
per question vs 3.6-7.8 ms for TF-IDF, with recall unchanged from the earlier FAISS run (p1 0.623, p2_tfidf 0.914, oracle 0.914,
query_all 0.678) because the cache never changes what is retrieved.

## Cache effect, paired A/B (template test, one process per router, same BGE model)

Each question ran uncached and cached back to back (alternating which arm went first), each with 3 repeats, so drift in machine speed
hits both arms equally (`cached/paired-ab/`, power AC at the start and end of every router). Means over the 171 questions, ms:

| router | class | n | overhead no cache | overhead cache | change | retrieval no cache | retrieval cache | change | total no cache | total cache | change | recall/tokens equal |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| oracle | all | 171 | 28.89 | 28.81 | -0.09 (-0%) | 48.06 | 20.09 | -27.97 (-58%) | 76.95 | 48.89 | -28.06 (-36%) | yes |
| p1 | all | 171 | 0.09 | 0.08 | -0.00 (-1%) | 36.30 | 32.57 | -3.73 (-10%) | 36.38 | 32.65 | -3.73 (-10%) | yes |
| p2_tfidf | all | 171 | 30.36 | 30.28 | -0.07 (-0%) | 45.63 | 19.78 | -25.84 (-57%) | 75.99 | 50.07 | -25.92 (-34%) | yes |
| query_all | all | 171 | 32.01 | 32.14 | +0.13 (+0%) | 62.51 | 35.64 | -26.87 (-43%) | 94.52 | 67.77 | -26.75 (-28%) | yes |
| always_filter_first | all | 171 | 31.18 | 31.94 | +0.76 (+2%) | 51.84 | 21.20 | -30.64 (-59%) | 83.02 | 53.14 | -29.88 (-36%) | yes |
| always_search_first | all | 171 | 32.07 | 32.45 | +0.38 (+1%) | 49.89 | 19.04 | -30.84 (-62%) | 81.96 | 51.50 | -30.46 (-37%) | yes |

Planning overhead does not change (within 2% for every router), because the first encode is still paid inside planning by the
probe; retrieval drops 43-62% for every router that probes, because the search reuses the probe's embedding (class C/D retrieval
falls 83-86%, e.g. oracle C 71.7 -> 12.1 ms), and p1, which never probes, still gains 14% on C/D because `distinct_search`'s
over-fetch loop calls `search` more than once per question; recall and tokens are identical in both arms for every question.

## Cost model check on the template run (every candidate plan executed once; Spearman)

| group | plans | rho cost TF-IDF | rho cost FAISS before | rho cost FAISS cache | rho latency TF-IDF | rho latency FAISS before | rho latency FAISS cache | rho tokens TF-IDF | rho tokens FAISS before | rho tokens FAISS cache | cheapest-plan agreement TF-IDF | cheapest-plan agreement FAISS before | cheapest-plan agreement FAISS cache |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| overall | 466 | 0.864 | 0.852 | 0.861 | 0.631 | 0.601 | 0.719 | 0.932 | 0.944 | 0.944 | 0.462 | 0.515 | 0.526 |
| A | 86 | 0.866 | 0.866 | 0.866 | 0.738 | 0.866 | 0.866 | 0.867 | 0.866 | 0.866 | 1.000 | 1.000 | 1.000 |
| B | 92 | 0.353 | 0.353 | 0.353 | 0.317 | 0.257 | 0.154 | 0.353 | 0.510 | 0.510 | 0.348 | 0.457 | 0.457 |
| C | 168 | 0.813 | 0.846 | 0.854 | 0.007 | 0.005 | 0.013 | 0.909 | 0.936 | 0.936 | 0.262 | 0.262 | 0.381 |
| D | 120 | 0.594 | 0.621 | 0.603 | -0.160 | 0.176 | -0.050 | 0.838 | 0.839 | 0.839 | 0.225 | 0.325 | 0.250 |

The cost model's ranking is unchanged in substance (rho cost 0.861 vs 0.852 before and 0.864 with TF-IDF): by design the cache is
cleared before every candidate execution in this check (d8b0b4d), so each plan pays its own encode as the cost model assumes, and the
remaining differences from "FAISS before" (e.g. class C cheapest-plan agreement 0.381 vs 0.262) are run-to-run timing noise.
An earlier analysis pass that cleared only once per question is kept in `cached/template-test/discarded-analysis-cache-per-question/`;
it let the first candidate carry the encode for the rest and gave class B a cheapest-plan agreement of 0.000.

## Peak memory (Windows peak working set per process, cache run)

| step | peak working set |
|---|---|
| template runner, per router | 1,008-1,022 MB |
| template analyses | 1,061 MB |
| paraphrase runner, per router | 1,009-1,042 MB |
| paired A/B, per router (two source sets, one model) | 1,091-1,111 MB |

## Reproduce

```
bash outputs/runs/faiss-demo/tools/run_cached.sh     # cache run: per-router template + paraphrase, merges, analyses
bash outputs/runs/faiss-demo/tools/run_paired.sh     # paired A/B
python outputs/runs/faiss-demo/tools/write_comparison.py
```
