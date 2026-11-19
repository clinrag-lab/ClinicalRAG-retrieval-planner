# Results: demo freeze (results-demo-v1)

Code: commit `5acdd6098c2e7c03f1758c26d797a560f315670d`. Data: open MIMIC-IV demo (275 admissions), generated notes, workload in data/workload. Only the dev split is used for calibration, tuning and training.
Config for every run: `outputs/runs/final-demo/config/calibrated_config.yaml` (sha256 `a7a2cd3e9c31`).

## Config: calibration and order-rule tuning

```
python -m clinrag.eval.calibrate --sources reference --workload data/workload --split dev --out outputs/runs/final-demo/config/calibrated_config.yaml
python -m clinrag.workload.notes_gen --tables-dir data/mimic-iv-demo --out-dir data/generated --seed 0 --scale 10
python -m clinrag.workload.notes_gen --tables-dir data/mimic-iv-demo --out-dir data/generated --seed 0 --scale 50
python -m clinrag.eval.tune_order --split dev --config outputs/runs/final-demo/config/calibrated_config.yaml --notes-dir data/generated/scale_10 --report-dir outputs/runs/final-demo/config
```

Tuned order rule (dev, class C/D, notes scale 10): s = 0.5, beta = 1.0; the rule picks the better order on 74.6% of 118 dev questions (default s = 0.1, beta = 1: 19.5%).

## Template test run (171 test questions, scale 1)

```
python -m clinrag.eval.runner --sources reference --workload data/workload --split test --routers oracle p1 p2_tfidf query_all always_filter_first always_search_first --config outputs/runs/final-demo/config/calibrated_config.yaml --out outputs/runs/final-demo/template-test
```

| router | recall | tokens | latency ms | p95 ms | overhead ms | fallback | missed sources |
|---|---|---|---|---|---|---|---|
| oracle | 0.906 | 718.6 | 6.95 | 16.43 | 6.63 | 0.000 | 0.000 |
| p1 | 0.911 | 726.4 | 10.19 | 23.04 | 0.16 | 0.000 | 0.000 |
| p2_tfidf | 0.906 | 718.6 | 7.87 | 19.68 | 8.97 | 0.000 | 0.000 |
| query_all | 0.625 | 3232.9 | 6.90 | 15.28 | 6.19 | 0.000 | 0.000 |
| always_filter_first | 0.911 | 726.4 | 6.99 | 16.08 | 6.31 | 0.000 | 0.000 |
| always_search_first | 0.855 | 631.7 | 7.09 | 15.02 | 6.30 | 0.000 | 0.000 |

## Cost model: Spearman, estimated vs measured (template test run, every candidate plan once)

| group | plans | cost | latency only | tokens only | cheapest-plan agreement |
|---|---|---|---|---|---|
| overall | 466 | 0.864 | 0.631 | 0.932 | 0.462 |
| A | 86 | 0.866 | 0.738 | 0.867 | 1.000 |
| B | 92 | 0.353 | 0.317 | 0.353 | 0.348 |
| C | 168 | 0.813 | 0.007 | 0.909 | 0.262 |
| D | 120 | 0.594 | -0.160 | 0.838 | 0.225 |

## Overhead share (template test run)

| router | overhead ms | retrieval ms | overhead share of total time |
|---|---|---|---|
| oracle | 6.63 | 6.95 | 0.488 |
| p1 | 0.16 | 10.19 | 0.016 |
| p2_tfidf | 8.97 | 7.87 | 0.533 |
| query_all | 6.19 | 6.90 | 0.473 |
| always_filter_first | 6.31 | 6.99 | 0.474 |
| always_search_first | 6.30 | 7.09 | 0.470 |

## Order check per scale (test, class C/D, re-calibrated per scale, s and beta fixed)

```
python -m clinrag.eval.scale_study --split test --scales 1 10 50 --config outputs/runs/final-demo/config/calibrated_config.yaml --out outputs/runs/final-demo/scale-study
```

| notes | filter_first better | policy | picks better order | mean recall | mean latency ms |
|---|---|---|---|---|---|
| 275 | 0.671 | selectivity_rule | 0.756 | 0.820 | 13.91 |
| 275 | 0.671 | cost_model | 0.671 | 0.829 | 14.08 |
| 275 | 0.671 | always_filter_first | 0.671 | 0.829 | 14.08 |
| 275 | 0.671 | always_search_first | 0.329 | 0.713 | 13.98 |
| 2,750 | 0.963 | selectivity_rule | 0.756 | 0.753 | 13.70 |
| 2,750 | 0.963 | cost_model | 0.963 | 0.860 | 15.05 |
| 2,750 | 0.963 | always_filter_first | 0.963 | 0.860 | 15.05 |
| 2,750 | 0.963 | always_search_first | 0.037 | 0.244 | 11.97 |
| 13,750 | 1.000 | selectivity_rule | 0.793 | 0.716 | 39.36 |
| 13,750 | 1.000 | cost_model | 1.000 | 0.834 | 51.85 |
| 13,750 | 1.000 | always_filter_first | 1.000 | 0.834 | 51.85 |
| 13,750 | 1.000 | always_search_first | 0.000 | 0.115 | 27.00 |

## End to end on held-out phrasings (684 test paraphrases, scale 1)

```
python -m clinrag.workload.paraphrase --workload-dir data/workload --out-dir data/workload/paraphrase --seed 0
# p1 rows: from this five-router run, which ran out of memory in p2_bge (p1 complete, p2_tfidf 680 of 684)
python -m clinrag.eval.runner --sources reference --workload data/workload/paraphrase --split test --train-workload data/workload --p2-train dev+paraphrases --p2-model-dir outputs/runs/final-demo/paraphrase-test/models --routers p1 p2_tfidf p2_bge oracle query_all --config outputs/runs/final-demo/config/calibrated_config.yaml --no-analysis --fig6 --out outputs/runs/final-demo/paraphrase-test
mkdir -p outputs/runs/final-demo/paraphrase-test/parts/partial && cp outputs/runs/final-demo/paraphrase-test/results.jsonl outputs/runs/final-demo/paraphrase-test/parts/partial/
# p2_tfidf, oracle and query_all: one router per run
python -m clinrag.eval.runner --sources reference --workload data/workload/paraphrase --split test --train-workload data/workload --p2-train dev+paraphrases --p2-model-dir outputs/runs/final-demo/paraphrase-test/models --routers p2_tfidf --config outputs/runs/final-demo/config/calibrated_config.yaml --no-analysis --out outputs/runs/final-demo/paraphrase-test/parts/p2_tfidf
python -m clinrag.eval.runner --sources reference --workload data/workload/paraphrase --split test --train-workload data/workload --p2-train dev+paraphrases --p2-model-dir outputs/runs/final-demo/paraphrase-test/models --routers oracle --config outputs/runs/final-demo/config/calibrated_config.yaml --no-analysis --out outputs/runs/final-demo/paraphrase-test/parts/oracle
python -m clinrag.eval.runner --sources reference --workload data/workload/paraphrase --split test --train-workload data/workload --p2-train dev+paraphrases --p2-model-dir outputs/runs/final-demo/paraphrase-test/models --routers query_all --config outputs/runs/final-demo/config/calibrated_config.yaml --no-analysis --out outputs/runs/final-demo/paraphrase-test/parts/query_all
python -m clinrag.eval.freeze --merge p1=outputs/runs/final-demo/paraphrase-test/parts/partial p2_tfidf=outputs/runs/final-demo/paraphrase-test/parts/p2_tfidf oracle=outputs/runs/final-demo/paraphrase-test/parts/oracle query_all=outputs/runs/final-demo/paraphrase-test/parts/query_all --questions data/workload/paraphrase/test.jsonl --into outputs/runs/final-demo/paraphrase-test
```

P2 with BGE embeddings (p2_bge) was evaluated for routing only, because the end-to-end run with BGE exceeded laptop memory.

| router | class | recall | tokens | latency ms | overhead ms | fallback | missed sources |
|---|---|---|---|---|---|---|---|
| p1 | A | 0.523 | 722.8 | 4.36 | 0.13 | 0.000 | 0.477 |
| p1 | B | 0.974 | 326.3 | 0.89 | 0.07 | 0.000 | 0.000 |
| p1 | C | 0.064 | 4837.6 | 7.69 | 0.10 | 0.000 | 1.000 |
| p1 | D | 0.233 | 1287.6 | 1.45 | 0.08 | 0.000 | 0.975 |
| p1 | all | 0.464 | 1758.9 | 3.56 | 0.10 | 0.000 | 0.594 |
| p2_tfidf | A | 0.866 | 293.1 | 4.52 | 3.40 | 0.000 | 0.134 |
| p2_tfidf | B | 0.974 | 326.3 | 1.13 | 2.07 | 0.000 | 0.000 |
| p2_tfidf | C | 0.833 | 1252.9 | 14.35 | 15.47 | 0.000 | 0.000 |
| p2_tfidf | D | 0.805 | 1258.1 | 11.89 | 16.95 | 0.000 | 0.000 |
| p2_tfidf | all | 0.873 | 763.5 | 7.75 | 9.18 | 0.000 | 0.034 |
| oracle | A | 1.000 | 114.6 | 2.72 | 0.04 | 0.000 | 0.000 |
| oracle | B | 0.974 | 326.3 | 0.58 | 0.03 | 0.000 | 0.000 |
| oracle | C | 0.833 | 1252.9 | 7.72 | 7.84 | 0.000 | 0.000 |
| oracle | D | 0.805 | 1258.1 | 10.27 | 12.64 | 0.000 | 0.000 |
| oracle | all | 0.906 | 718.6 | 5.14 | 4.90 | 0.000 | 0.000 |
| query_all | A | 1.000 | 1470.2 | 5.24 | 0.07 | 0.000 | 0.000 |
| query_all | B | 0.974 | 326.3 | 1.14 | 0.06 | 0.000 | 0.000 |
| query_all | C | 0.233 | 7530.3 | 13.13 | 11.96 | 0.000 | 0.000 |
| query_all | D | 0.230 | 3837.4 | 10.48 | 12.90 | 0.000 | 0.000 |
| query_all | all | 0.625 | 3204.7 | 7.30 | 5.99 | 0.000 | 0.000 |

## Classifier routing on held-out phrasings (684 test paraphrases, tau 0.5)

```
python -m clinrag.planner.train_classifier --workload-dir data/workload --paraphrase
cp outputs/routing_paraphrase_report.json outputs/runs/final-demo/routing/
```

This is the only evaluation of p2_bge in the freeze; see the note above.

| router | exact match | missed sources | F1 relational | F1 notes | F1 graph | predict ms |
|---|---|---|---|---|---|---|
| P1_rules | 0.406 | 0.594 | 0.621 | 0.781 | 0.129 | 0.04 |
| P2_tfidf | 0.890 | 0.098 | 0.927 | 0.933 | 0.976 | 3.11 |
| P2_bge | 0.936 | 0.051 | 0.955 | 0.967 | 0.976 | 112.02 |

## Figures

- [`docs/figures/fig1_tokens_vs_recall.png`](../figures/fig1_tokens_vs_recall.png) (from template-test/)
- [`docs/figures/fig2_order_vs_selectivity.png`](../figures/fig2_order_vs_selectivity.png) (from scale-study/)
- [`docs/figures/fig3_tau_sweep.png`](../figures/fig3_tau_sweep.png) (from template-test/)
- [`docs/figures/fig4_overhead.png`](../figures/fig4_overhead.png) (from template-test/)
- [`docs/figures/fig5_scale.png`](../figures/fig5_scale.png) (from scale-study/)
- [`docs/figures/fig6_paraphrase_e2e.png`](../figures/fig6_paraphrase_e2e.png) (from paraphrase-test/)
