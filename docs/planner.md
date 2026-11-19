# Planner design

The planner picks how to answer a question from three sources: relational
(DuckDB over hospital tables), notes (vector search over discharge note chunks)
and graph (ICD diagnosis hierarchy). It estimates each candidate plan's cost
from cheap probes, without running the plan, and executes the cheapest one.

```
QuestionSpec
    |
    v
 probes ------- counts only: relational rows and ids, notes per admission,
    |           top-k probe scores, ICD codes under the category
    v
 enumerate ---- every plan that covers the required sources, plus query_all
    |
    v
 cost --------- latency_ms + alpha * tokens, per plan   (order rule prunes one variant)
    |
    v
 pick cheapest  (ties go to the earlier plan, so targeted plans beat query_all)
    |
    v
 execute ------ run steps in order, pass ICD codes and admission ids forward
    |
    v
 evidence ----- items + measured tokens + per-step latency
```

## Probes

Probes are lazy (`planner/probes.py`). Only a question with an order choice,
a structured filter or category combined with a note search (classes C and
D), runs the relational and note-count probes and the trial note search:

| probe | how |
|---|---|
| graph codes | `GraphSource.expand`; `IcdGraph` caches expansions |
| rows, admissions, matching notes | `RelationalSource.count_matches` in one SQL query when the source implements it; else `count` + `matching_ids`, with notes from `count_notes` (SQL) or `NotesSource.count_notes_for(ids)` |
| total notes, trial scores | `NotesSource.total_notes`, `probe_search(note_query, probe_k)` |

`count_notes` and `count_matches` are optional on `RelationalSource`. The
reference DuckDB source implements them by loading the note index's metadata
(note_id and hadm_id, never text) as a table.

Other questions have no order to choose. They read only index metadata:
`total_notes`, plus the one admission's note count for a per-admission
question. Their plans are still costed with what is known. An unprobed
relational step costs its latency but no tokens, a lower bound.
`Probes.ran` records which probes ran.

## Cost formula

```
cost(plan) = est_latency_ms + alpha * est_tokens      (summed over steps)
```

| step | latency (ms) | tokens |
|---|---|---|
| graph expand | `graph_fixed_ms + graph_ms` | 0 (codes feed the relational step) |
| relational | `relational_fixed_ms + relational_ms` | `rows * avg_tokens_per_row` if rows are evidence, else 0 |
| note search | `note_search_fixed_ms` + see below | `min(k, available) * avg_tokens_per_chunk` |

Each step pays its fixed per-call cost once (`*_fixed_ms`, default 0): DuckDB
call overhead, or embedding the query for a note search. Calibration fits them
as regression intercepts, so a plan with more steps pays more fixed cost even
when each step touches little data.

Note-search latency, on top of `note_search_fixed_ms`, with `scan(n) = note_scan_ms_per_1k_notes * n / 1000`,
`fetch(n) = note_fetch_ms_per_1k_candidates * n / 1000` and
`id_filter(n) = id_filter_ms_per_1k_ids * n / 1000`:

| mode | latency |
|---|---|
| unfiltered | `scan(total_notes) + fetch(k)` |
| search_first | `scan(total_notes) + fetch(min(ceil(beta * k * total_notes / matching_notes), total_notes))` |
| filter_first | `id_filter(matching_admissions) + scan(matching_notes) + fetch(k)` |
| unfiltered, per-admission question (`note_hadm_id` set) | `id_filter(1) + scan(scoped_notes) + fetch(k)` |

Per-admission class B questions search only one admission's notes (the
executor restricts the search to `note_hadm_id`), so they are costed as
filter_first over that admission. The probes collect `scoped_notes` with
`count_notes_for([note_hadm_id])`.

**Distinct admissions.** Every note search returns the top k distinct
admissions, one note each (the admission's best-scoring note). Gold answers
are per admission, so k notes from one admission would waste the evidence
budget once admissions have several notes or chunks. The executor applies
this to any `NotesSource`:

- unfiltered, filter_first and per-admission searches over-fetch (k notes,
  then 4x more) until k distinct admissions are found or the searched notes
  run out;
- search_first fetches its candidate count in notes, filters, then keeps one
  note per admission.

Token estimates count one item per admission: `min(k, total_notes)`
unfiltered, 1 for a per-admission search, `min(k, matching_admissions)`
otherwise. Over-fetching is not in the latency model.

A flat vector index scores every note it searches, so unfiltered and
search-first queries pay for all notes regardless of k, while filter-first
scores only the matching notes but pays to build and apply the admission id
filter. In filter-first and search-first plans the relational rows only
filter, so they add no tokens. All formulas are in
`src/clinrag/planner/cost.py`. The constants in `config.example.yaml` are
starting guesses to be calibrated on real runs.

## Filter-order rule

When a structured filter restricts a note search, search-first needs about
`k * N / f` unfiltered candidates to keep k notes after filtering, where
`N = total_notes` and `f = matching_notes`. That holds only if the matching
notes are spread evenly through the ranking. If the filter and the query are
correlated, relevant notes cluster outside the cohort and more candidates are
needed, so the rule scales the count by `beta`:

```
candidates = ceil(beta * k * N / f)
```

- `f == 0` -> filter_first
- candidates `> c * N` -> filter_first
- otherwise -> search_first (the exact boundary counts as search_first)

The limit is relative to the index size: search-first may fetch at most a
fraction `c` of the index. Equivalently, it switches to search_first once
`f >= beta * k / c` matching notes exist. The executor's search-first step
fetches the same beta-corrected count, which the planner fills into the plan.

`c` and `beta` are `PlannerConfig` keys (defaults 0.25 and 1.0).
`python -m clinrag.eval.tune_order --split dev` grid-searches them on class C
and D dev questions:

- grid: `c` in {0.02, 0.05, 0.1, 0.25, 0.5, 1.0}, `beta` in {1, 1.5, 2, 3};
- for each question, the better order is the one with higher recall, ties
  broken by lower measured latency;
- the winner maximizes the share of questions where the rule picks the better
  order, ties broken by the picked orders' mean recall, then mean latency.

The tuned values are written into `outputs/calibrated_config.yaml`, with the
full grid in `outputs/tune_order_report.md`. The test split is never used.

The planner applies this rule (`planner/order.py`) to each mixed plan family and
prunes the variant it rules out. The pruned plan stays in `PlanDecision.candidates`
with its cost and a `pruned_reason`, for logging and plots. The cost model then
chooses between plan families. `PlanDecision.order_costs` gives both variants'
estimated costs, and `PlanDecision.cost_order` gives the order the cost model
alone would pick. Compare these with `PlanDecision.order.order` (the rule's pick)
and with the measured faster order.

## Candidate plans

| required sources | plans |
|---|---|
| relational | `relational_only` |
| notes | `notes_only` |
| relational, notes | `relational+notes:filter_first`, `:search_first`, `:unfiltered` |
| graph, relational | `graph>relational` |
| graph, relational, notes | `graph>relational+notes:filter_first`, `:search_first` |
| any | `query_all` |

`query_all` is the main baseline. It queries every source the question gives
an input for and never restricts the note search. It always searches notes,
using the question text if there is no note query. Graph codes still feed its
relational step. Without a category it has no graph step, so for a
relational + notes question it has the same steps as `:unfiltered`.

## Routing

The cost planner above needs a question's required sources. A router decides
them. Every router implements `Router` (`planner/router.py`): a `name` and
`decide(question) -> PlanDecision`. The experiment runner executes each
decision and logs one `RoutingResult` per (router, question), with these fields:
router, qid, class, chosen plan, predicted and required sources, planning
overhead, estimated cost, and the measured tokens, latency, sources queried
and fallbacks used.

| router | name | sources from | plan choice | overhead |
|---|---|---|---|---|
| `OracleCostRouter` | `oracle_cost` | true `required_sources` | CostPlanner | probes + costing |
| `RulePlanner` (P1) | `rules` | keyword rules | fixed shape | rules only |
| `ClassifierCostRouter` (P2) | `classifier_cost` | `SourceClassifier` at `tau` | CostPlanner | classifier + probes + costing |

`OracleCostRouter` is an upper bound for analysis, not a contender. No
deployable router knows the true sources.

**P1 rules** (`planner/rule_planner.py`):

- lab name or comparison word: relational
- ICD category name: graph + relational
- "summary", "mention" or "documented": notes
- nothing fires: notes

The plan runs graph, then relational, then notes. A relational filter combined
with notes is always filter_first: no search_first, no cost model, no probes.
Its `est_cost` is `None`.

**P2** (`ClassifierCostRouter`) predicts a probability per source, keeps
sources with `p >= tau`, and runs CostPlanner on a copy of the question
restricted to them. When the restriction drops a source, its `query_all`
candidate is replaced by `query_all` over every source the question has
inputs for, costed with probes of that wider scope. The fallback can then
still reach a source the classifier missed.

**Restricting a question** (`restrict_question`). Routers predict sources;
the question's parsed inputs (lab filter, ICD category, note query) are reused
as given. A predicted source is kept only if it can run:

- graph needs a category, and keeps relational, which consumes the codes;
- relational needs a lab filter, or graph codes;
- notes always runs, searching the question text when there is no note query.

A category is cleared when graph is dropped, and the lab filter when
relational is dropped. If nothing predicted can run, the router uses its most
preferred source that can: the most probable one for P2, notes for P1.

## Source classifier

`SourceClassifier` (`planner/classifier.py`) trains one logistic regression
per source on the question text only. Its features are:

- TF-IDF of word 1-2 grams, with admission ids replaced by `admid` and other
  numbers by `num`. With `featurizer="bge"`, BAAI/bge-base-en-v1.5 embeddings
  instead; sentence-transformers is imported only then.
- Five binary rule features: a known lab name, a number or comparison,
  discharge summary / mention / documented, a known ICD category name, and an
  admission id.

The classifier works like this:

- `predict_proba(question)` returns a probability per source, and
  `predict(question, tau)` returns the sources with `p >= tau`. Each call sets
  `last_latency_ms`, which P2 counts as planning overhead.
- A source whose training label never varies gets a constant probability.
- Misses are left to the fallback; the classifier never adds sources by default.

Train it and write its report:

```
python -m clinrag.planner.train_classifier --workload-dir data/workload \
    --out outputs/classifier.joblib --featurizer tfidf
```

The script trains on `dev.jsonl` only. The report, `outputs/classifier_report.json`
(named `<out stem>_report.json`), contains:

- per-source precision, recall and F1, and exact-match accuracy of the
  predicted source set, both from 5-fold cross-validation on dev grouped by
  `template_id` and on `test.jsonl`;
- mean and p95 single-question predict latency;
- for tau = 0.1 .. 0.9 on test, the fraction of questions missing a needed
  source and the average number of sources predicted. This is the routing
  half of the tau sweep.

## Held-out phrasings

Every workload question comes from a fixed template, so TF-IDF, BGE and the
P1 rules all route the template questions perfectly. `workload/paraphrase.py`
re-renders each question in eight phrasing families per class, so routers can
be compared on wording they have not seen:

| class | families |
|---|---|
| A | `A_abbrev`, `A_terse`, `A_formal`, `A_lay`, `A_draw`, `A_stay_first`, `A_chart`, `A_value` |
| B | `B_record`, `B_natural`, `B_terse`, `B_document`, `B_notes`, `B_course`, `B_command`, `B_writeup` |
| C | `C_implicit`, `C_terse`, `C_who`, `C_lay`, `C_records`, `C_question_first`, `C_notes`, `C_cohort` |
| D | `D_cohort`, `D_terse`, `D_coded`, `D_history`, `D_dx`, `D_question`, `D_known`, `D_problem_list` |

The families draw on:

- **lab abbreviations and synonyms:** Cr, serum creatinine, kidney function
  marker; K, K+, serum potassium; Hgb, Hb, blood count;
- **comparison verbs:** exceeded, came in over, was elevated past, topped;
- **ICD category synonyms:** cardiac, kidney or urinary, lung or breathing,
  CHF, CKD, AKI, lung infection, DM, and others;
- **implicit note cues:** for example "Was the patient anticoagulated during
  stay {id}?", with no "discharge summary" or "mention";
- **terse clinician forms:** for example "latest K for {id}?" and
  "max Hgb, adm {id}".

A paraphrase is rendered from the question's slots: template kind, lab,
threshold, admission id, concept and category. The QuestionSpec slots and the
gold answer are therefore unchanged. Only the text changes, and the qid gains
a `_p<k>` suffix, where k is the family's index in its class. Each paraphrase
records its `phrasing_family` and `source_qid`. Word choices inside a family
are seeded per (seed, qid, family).

`split_families(seed)` shuffles each class's families and assigns half to
train and half to test. To write the split and the paraphrases to
`data/workload/paraphrase/`:

```
python -m clinrag.workload.paraphrase --workload-dir data/workload --seed 0
```

The data has these splits:

- **Train:** the original dev questions, plus dev questions in the train families.
- **Evaluation:** test-template questions in the test families only. Neither
  the template nor the phrasing appears in training.

To compare the routers:

```
python -m clinrag.planner.train_classifier --workload-dir data/workload --paraphrase
```

This writes `outputs/routing_paraphrase_report.json`, covering P1 (rules) and
P2 with tfidf and with bge (`--featurizers` picks a subset). Each router is
scored on its actual source decision:

- P1 searches notes when no rule fires.
- P2 takes the most probable source when nothing passes tau.

For each router the report contains:

- per-source precision, recall and F1;
- exact match and missed-source rate;
- per-class exact match and missed-source rate;
- predict latency;
- for P2, the tau sweep from 0.1 to 0.9;
- the five test families with the most missed sources, each with an example.

## Evidence check and fallback

`check_evidence(result, question, config)` (`planner/fallback.py`) fails a
result in any of these cases:

- it has no evidence items;
- its note search returned no notes;
- its best note `score` is below `min_similarity`;
- its note search was filtered (filter_first, search_first, or scoped to
  `note_hadm_id`) and fewer than `min_surviving_notes` notes survived.

Note items carry their similarity as `score`.

`run_with_fallback(decision, sources, question, config)` runs the chosen plan.
While the check fails, it runs the next plan from `decision.candidates`:

- only feasible candidates, cheapest first (uncosted P1 candidates in their
  listed order);
- the next plan must query every source the failed plan did plus at least one
  more, or be `query_all`;
- a candidate with the same steps as a plan already run is skipped (for a
  notes-only question, `query_all` is the same search as `notes_only`).

It stops after `max_fallbacks` extra plans or when no candidate is left.
Tokens and latency are summed over every attempt. Each attempt is recorded
with its plan and the reason its check failed.

| key | default | meaning |
|---|---|---|
| `min_similarity` | 0.3 | best note score below this fails the check |
| `min_surviving_notes` | 1 | a filtered note search keeping fewer notes fails |
| `max_fallbacks` | 2 | extra plans run after the first |
