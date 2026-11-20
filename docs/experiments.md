# Experiments

The experiment runner turns the planner into results. It runs every router on
every question of a workload split, executes each decision with the evidence
check and fallback, scores the retrieved evidence, and summarizes cost,
latency and recall. Analyses and figures follow. Use only the fake fixture
and the open MIMIC-IV demo with generated notes; never credentialed MIMIC.

```
calibrate ──> outputs/calibrated_config.yaml
                    │ --config
                    v
runner: for each router, for each question
          decide ─> run_with_fallback (x repeats) ─> retrieval_score ─> results.jsonl
        summarize ─> results.csv, results.md
        analyses  ─> analysis/ (cost model, order check, overhead, tau sweep)
        figures   ─> figures/fig1..fig4
tune_order (dev) ─> s, beta in calibrated_config.yaml
scale_study ─> order check at several note-index sizes, figures fig2 and fig5
```

## Sources

`--sources fake` uses the reference implementations over the bundled fixture
tables and notes (`tests/fixtures`), for smoke tests. `--sources reference`
uses them over the demo tables (`data/mimic-iv-demo`) and the generated notes
(`data/generated`).

| source | implementation (`sources/reference.py`) |
|---|---|
| relational | `DemoRelationalSource`: DuckDB over the hosp tables. Lab filters are applied to the same rows the gold answers use; ICD codes are matched on (icd_version, icd_code). |
| notes | `TfidfNotesSource`: TF-IDF, one chunk per note, brute-force cosine scoring. Filtered searches score only the given admissions' notes. |
| graph | `IcdGraph` |

These are reference implementations. The team's DuckDB and FAISS sources
replace them: anything implementing `sources/base.py` plugs in the same way.

## Routers

| name | router | sources from | plan |
|---|---|---|---|
| `oracle` | `OracleCostRouter` | true sources | cheapest by the cost model (upper bound) |
| `p1` | `RulePlanner` | keyword rules | fixed shape, filter_first |
| `p2_tfidf`, `p2_bge` | `ClassifierCostRouter` | classifier at `--tau` | cheapest by the cost model |
| `query_all` | `QueryAllRouter` | true sources | query_all |
| `always_filter_first` | `AlwaysFilterFirstRouter` | true sources | filter_first whenever a filter meets notes |
| `always_search_first` | `AlwaysSearchFirstRouter` | true sources | search_first whenever a filter meets notes |

The three fixed baselines (`baselines/fixed.py`) run the same probes and cost
model as the oracle and only force the final pick. Only order or coverage
differs from the oracle.

P2 classifiers are trained on `dev.jsonl` when the run starts. On the dev
split, P2 is therefore scored on its own training questions. Use `--split
test`, or the tau sweep on held-out phrasings, for a fair P2 number.

The LLM router and vector-only RAG baselines are left to the team. The answer
step is a slot: `llm/answerer.py` defines the `Answerer` protocol, and
`NullAnswerer`, the default (`--answerer none`), returns no answer, so only
retrieval is scored.

`--answerer llama` runs Llama-3.1-8B-Instruct (`llm/llama.py`; install with
`pip install -e ".[llm]"`, `--llm-dtype bf16` or `4bit`). The module is
imported only for that flag. Every router's evidence goes to the answerer in
plan order, cut to `max_context_tokens` (planner config, default 8192): the
first item that does not fit and everything after it are dropped. Lab rows are
sent one per line, oldest to newest; class A latest and max questions retrieve
only the row that answers them (`StructuredFilter.agg`). The output is parsed
by `eval/parse.py` against the answer type of the question's template. A null
or unparseable answer scores 0. Each `results.jsonl` row records `answer_raw`,
`answer`, `answer_score`, `parse_failed`, `prompt_tokens_sent` and `truncated`;
`results.csv` and `results.md` add `answer_accuracy`, `parse_failure_rate` and
`truncated_rate` per router and class.

## Calibration

```
python -m clinrag.eval.calibrate --sources reference --workload data/workload --split dev
```

Calibration times the actual sources and fits the cost constants. Every
timing is the median of `--repeats`.

| constant | measured as |
|---|---|
| `relational_fixed_ms` | intercept of `run` time against rows returned, over the workload's relational queries |
| `relational_ms` | median `run` time minus `relational_fixed_ms` |
| `graph_fixed_ms`, `graph_ms` | the same for `expand` against codes returned, over its categories |
| `note_search_fixed_ms` | intercept of the scan regression minus fetch(1): the fixed part of a search (embedding or vectorizing the query) |
| `id_filter_ms_per_1k_ids` | slope of a search restricted to m ids that match no note, vs m |
| `note_scan_ms_per_1k_notes` | slope of a search restricted to s admissions, vs their notes (id filter subtracted) |
| `note_fetch_ms_per_1k_candidates` | slope of an unfiltered search for n results, vs n |
| `avg_tokens_per_row`, `avg_tokens_per_chunk` | mean tokens of returned rows and note chunks |
| `min_similarity` | score threshold that best separates notes tagged with the queried concept from the rest (balanced accuracy) |

Slopes and intercepts are least squares; the report lists slope, intercept
and R^2 for each fit. The intercepts become the per-call fixed costs.
`--base-config` keeps the unfitted keys of an existing config, such as the
tuned `c` and `beta`.

`min_similarity` is fitted because its right value depends on the scoring
function. TF-IDF cosine scores here are far below the 0.3 default chosen for
BGE.

Calibration writes three files to `outputs/`:

- `calibrated_config.yaml`, used by the runner through `--config`;
- `calibration_report.json`;
- `calibration_report.md`.

## Runner

```
python -m clinrag.eval.runner --sources reference --workload data/workload --split dev \
    --routers oracle p1 p2_tfidf query_all always_filter_first always_search_first \
    --config outputs/calibrated_config.yaml --repeats 3
```

For every router and question, the runner decides and executes with fallback
`--repeats` times. Latency and planning overhead are medians over the
repeats. Tokens and evidence come from the first repeat.

`retrieval_score` (`eval/metrics.py`) scores the final evidence by question
type:

| question | metric |
|---|---|
| A latest / max | the gold admission's row with the gold value came back |
| A ever above | whether any returned row passes the filter agrees with gold |
| B per-admission | the gold admission's note came back |
| B list, C, D | normalized recall@k over the note hadm_ids, in rank order |

Output in `--out` (default `outputs/runs/<timestamp>/`):

| file | contents |
|---|---|
| `results.jsonl` | one line per (router, question): every `RoutingResult` field, recall, estimated cost, measured cost (latency + alpha * tokens), selectivity, fallback attempts, answer fields |
| `results.csv`, `results.md` | per router and class: mean recall, mean tokens, mean and p95 latency, mean overhead, fallback rate, sources queried |
| `run.json` | arguments and planner config |
| `analysis/` | the analyses below, as CSV, `summary.json` and `analysis.md` |
| `figures/` | four PNGs |

## Analyses

These live in `eval/analysis.py` and run at the end of the runner; skip them
with `--no-analysis`.

- **Cost model check.** Every candidate plan of every question is executed
  once. The analysis reports the Spearman rank correlation between estimated
  and measured cost, overall and per class. It also reports latency and
  tokens separately, and how often the estimated-cheapest plan is the
  measured-cheapest.
- **Order check (RQ2).** Every class C and D question runs both filter_first
  and search_first. The analysis records the faster order (median of repeats)
  and each order's recall. It reports how often the order rule's pick and the
  cost model's pick match the faster order, binned by selectivity.
- **Overhead share.** Planning overhead divided by overhead plus retrieval
  time, per router.
- **Tau sweep.** P2 TF-IDF at tau 0.1 to 0.9: missed-source rate (a needed
  source absent from the chosen plan), mean tokens and recall. By default it
  runs on the run's questions rendered in held-out phrasing families, with
  the classifier trained on dev plus the train-family paraphrases
  (`--sweep-on paraphrased`; also `original` or `none`).

## Figures

| file | shows |
|---|---|
| `fig1_tokens_vs_recall.png` | mean tokens vs mean recall per router, ±1 SE |
| `fig2_order_vs_selectivity.png` | latency of filter_first and search_first vs selectivity, with the rule's switch point; one panel per index size when written by `scale_study` |
| `fig3_tau_sweep.png` | P2 TF-IDF tau vs missed-source rate and mean tokens (two axes) |
| `fig4_overhead.png` | planning overhead stacked on retrieval latency per router |
| `fig5_scale.png` | notes in the index (log) vs the share of C/D questions where filter_first is the better order, and how often the rule and the cost model pick it (`scale_study`) |

Each title is computed from the data and states the takeaway.

## Full rerun

Tuning and calibration use the dev split only; results are reported on test.

```
# 1. timings and min_similarity on the scale-1 notes (dev queries)
python -m clinrag.eval.calibrate --sources reference --workload data/workload --split dev
# 2. order rule s and beta on class C/D dev questions at notes scale 10; written into the same config
python -m clinrag.eval.tune_order --split dev --config outputs/calibrated_config.yaml \
    --notes-dir data/generated/scale_10
# 3. every router on the test split, with analyses and figures
python -m clinrag.eval.runner --sources reference --workload data/workload --split test \
    --routers oracle p1 p2_tfidf query_all always_filter_first always_search_first \
    --config outputs/calibrated_config.yaml --out outputs/runs/<run>
# 4. larger note indexes: S notes per admission, same per-admission concepts and gold
python -m clinrag.workload.notes_gen --tables-dir data/mimic-iv-demo --out-dir data/generated --scale 10
python -m clinrag.workload.notes_gen --tables-dir data/mimic-iv-demo --out-dir data/generated --scale 50
# 5. order check at each scale, re-calibrated per scale (s and beta stay as tuned)
python -m clinrag.eval.scale_study --split test --scales 1 10 50 \
    --config outputs/calibrated_config.yaml --out outputs/runs/<run>
```

Step 5 overwrites the run's `fig2` with one panel per scale and adds `fig5`.
The runner and calibration also take `--notes-dir data/generated/scale_S` to
run everything on a scaled index.

## Results freeze

The frozen demo results (git tag `results-demo-v1`) live in
`outputs/runs/final-demo/`. Each part comes from one command, listed in
`clinrag.eval.freeze.COMMANDS` and repeated in `SUMMARY.md`:

| folder | contents |
|---|---|
| `config/` | calibration on dev (scale 1), and s and beta tuned on dev at notes scale 10 |
| `template-test/` | every router on the template test split, with analyses |
| `scale-study/` | order check on test at notes scales 1, 10 and 50 |
| `paraphrase-test/` | p1, p2_tfidf, oracle and query_all on the 684 held-out-phrasing test questions, one router per run under `parts/`, merged with `freeze --merge`; P2 trained on dev plus train-family paraphrases |
| `routing/` | classifier routing report on held-out phrasings (P1, P2 tfidf, P2 bge) |

After the runs, collect the headline numbers and figures:

```
python -m clinrag.eval.freeze --run outputs/runs/final-demo
```

This writes `SUMMARY.md` and copies every figure into `figures/`. P2 with
BGE embeddings is evaluated for routing only: the end-to-end run with BGE
exceeded laptop memory. `outputs/` is git-ignored, so the tag pins the code
and the commands, not the result files.

## Caveats on the demo

- **Fixed per-call costs dominate at scale 1.** The demo has 275 admissions
  and one note each. Every note search is cheap, so fixed per-call costs
  (query vectorization, DuckDB queries) dominate latency. Calibration fits
  them as `*_fixed_ms`, and the scale study grows the index to 2,750 and
  13,750 notes to separate the two orders.
- **The order rule switches at a selectivity, `f / N < s`.** It is tuned at
  scale 10 and applied unchanged at every scale. The earlier candidate-budget
  rule reduced to a fixed matching-note count; see docs/planner.md.
- **Single runs are noisy.** Latency differences between plans are often
  below a millisecond, so the cost model check executes each plan once and
  its latency correlations are noisy.
