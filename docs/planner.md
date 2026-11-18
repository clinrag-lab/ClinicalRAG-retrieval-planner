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

## Cost formula

```
cost(plan) = est_latency_ms + alpha * est_tokens      (summed over steps)
```

| step | latency (ms) | tokens |
|---|---|---|
| graph expand | `graph_ms` | 0 (codes feed the relational step) |
| relational | `relational_ms` | `rows * avg_tokens_per_row` if rows are evidence, else 0 |
| note search | `note_search_ms_per_1k_candidates * candidates / 1000` | `min(k, available) * avg_tokens_per_chunk` |

Note-search candidates: `unfiltered` = k; `filter_first` = matching notes (the
restricted search scores all of them); `search_first` =
`min(ceil(k * total / matching), total)`. In filter-first and search-first
plans the relational rows only filter, so they add no tokens. All formulas are
in `src/clinrag/planner/cost.py`. The constants in `config.example.yaml` are
starting guesses to be calibrated on real runs.

## Filter-order rule

When a structured filter restricts a note search, search-first needs about
`k * total_notes / matching_notes` candidates to keep k notes after filtering.

- `matching_notes == 0` -> filter_first
- candidates `> max_candidates` -> filter_first
- otherwise -> search_first (the exact boundary counts as search_first)

The planner applies this rule (`planner/order.py`) to each mixed plan family and
prunes the variant it rules out. The pruned plan stays in `PlanDecision.candidates`
with its cost and a `pruned_reason`, for logging and plots. The cost model then
chooses between plan families.

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
