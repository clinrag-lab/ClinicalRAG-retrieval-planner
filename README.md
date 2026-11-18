# ClinicalRAG-retrieval-planner

Cost-aware retrieval planning for clinical retrieval-augmented generation (RAG) over MIMIC-IV. A question can be answered from three sources: DuckDB over the structured hospital tables, a FAISS index over embedded discharge notes, and an in-memory ICD diagnosis-code hierarchy. For each question, the planner decides which sources to query and whether to filter first (narrow the cohort with structured data, then search notes) or search first (retrieve notes, then filter). It picks the plan with the lowest estimated cost, where `alpha` weights the tokens sent to the model. A local Llama-3.1-8B-Instruct then answers from the retrieved context.

## Repo layout

```
src/clinrag/
  config.py        load settings from a YAML config
  sources/         DuckDB, FAISS and ICD hierarchy sources
  planner/         plan selection (sources and filter-first vs search-first)
  workload/        query workload generation and loading
  baselines/       baseline retrieval strategies
  eval/            metrics and experiment runners
  llm/             local LLM answer generation
tests/
  fixtures/        tiny fake dataset with MIMIC-IV column names
  conftest.py      loads fixtures into an in-memory DuckDB
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
