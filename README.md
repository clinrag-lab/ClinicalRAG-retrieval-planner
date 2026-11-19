# ClinicalRAG-retrieval-planner

Cost-aware retrieval planning for clinical retrieval-augmented generation (RAG) over MIMIC-IV. A question can be answered from three sources: DuckDB over the structured hospital tables, a FAISS index over embedded discharge notes, and an in-memory ICD diagnosis-code hierarchy. For each question, the planner decides which sources to query and whether to filter first (narrow the cohort with structured data, then search notes) or search first (retrieve notes, then filter). It picks the plan with the lowest estimated cost, where `alpha` weights the tokens sent to the model. A local Llama-3.1-8B-Instruct then answers from the retrieved context.

## Repo layout

```
src/clinrag/
  config.py        load settings from a YAML config
  question.py      QuestionSpec: a question and the inputs each source needs
  sources/         source interfaces (base.py), token counting, and implementations
  planner/         probes, plan enumeration, cost model, order rule, executor
  workload/        synthetic notes, question templates and workload builder
  baselines/       baseline retrieval strategies
  eval/            metrics and experiment runners
  llm/             local LLM answer generation
tests/
  fixtures/        tiny fake dataset with MIMIC-IV column names
  conftest.py      loads fixtures into an in-memory DuckDB
  fakes.py         deterministic fake sources for planner tests
docs/
  planner.md       planner design: cost formula, order rule, candidate plans
scripts/
  download_demo.sh fetch the open-access MIMIC-IV demo
  sol/             Slurm job scripts for ASU Sol
config.example.yaml
```

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp config.example.yaml config.yaml # then edit paths in config.yaml
```

To get the open-access MIMIC-IV Clinical Database Demo v2.2 (no credentials needed):

```bash
bash scripts/download_demo.sh      # writes to data/mimic-iv-demo/
```

On Sol, submit the example job from the repo root with `bash scripts/sol/example_job.sh` after confirming `PARTITION` at the top of the script.

## Building the question workload

The workload is built from the open MIMIC-IV demo only (ODbL). The demo has no
clinical notes, so step 1 generates one synthetic discharge note per admission
from that admission's diagnoses, labs, prescriptions and disposition. It also
writes `note_concepts.csv.gz`, which records exactly which concepts each note
mentions; negated mentions are logged in `note_negations.csv.gz` and never
tagged. Step 2 builds the questions and their gold answers. Both steps are
deterministic for a given `--seed`, and everything they write stays under the
git-ignored `data/` folder.

```bash
bash scripts/download_demo.sh   # if data/mimic-iv-demo/ is missing
python -m clinrag.workload.notes_gen --tables-dir data/mimic-iv-demo --out-dir data/generated --seed 0
python -m clinrag.workload.build --tables-dir data/mimic-iv-demo --notes-dir data/generated \
    --out-dir data/workload --seed 0
```

`--tables-dir` can be the hosp folder or any folder above it. The build writes
`dev.jsonl`, `test.jsonl` and `summary.json`; load the questions with
`clinrag.workload.schema.read_jsonl`. There are four question classes:

| class | sources | example | gold |
|---|---|---|---|
| A | relational | most recent / highest creatinine in an admission, was it ever above 2.0 | a number (1/0 for yes/no) |
| B | notes | which discharge summaries mention dialysis | sorted hadm_ids |
| B | notes (one admission) | according to the discharge summary for admission X, was dialysis performed | 1 / 0; about half are no, some with a negated mention |
| C | relational + notes | among admissions with creatinine > T, which mention dialysis | sorted hadm_ids, plus filter selectivity |
| D | graph + relational + notes | among admissions with any renal (chapter) or chronic kidney disease (block) diagnosis, which mention dialysis | sorted hadm_ids, plus category selectivity and level |

Questions whose gold list is empty or covers more than 95% of admissions are
skipped, and each class is capped at 100. Dev and test are split by
template_id (about 60/40), so no template appears in both. Concept
probabilities are in `CONCEPT_PROBS` in `workload/notes_gen.py`. Diagnosis
categories are in `workload/icd_categories.py` as a two-level hierarchy (4
chapters, 9 blocks), and `sources/icd_graph.py` expands either level into the
version-qualified codes present in the data (`"10:I509"`, `"9:4280"`).

## Running tests

```bash
pytest
```

Tests use only the fake fixtures in `tests/fixtures/`, so they run without any MIMIC data. CI runs them on every push and pull request.

## Data rules

- Never commit MIMIC data, derived tables, embeddings, indexes or model outputs. This includes the demo dataset.
- Real data lives only on approved storage (for example, your Sol storage), never in the repo or on personal cloud drives.
- `config.yaml` stays local; commit changes to `config.example.yaml` instead.
- The only data files in the repo are the made-up fixtures in `tests/fixtures/`. Keep them fake.
- `.gitignore` blocks common data formats, but check `git status` before every commit.
