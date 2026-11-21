#!/usr/bin/env bash
# v2 accuracy job: every router with the Llama answer step on the v2 template and paraphrase
# test splits, one runner command per router, then the paraphrase merge and SUMMARY.md.
#
# Submit from the repo root:              bash scripts/sol/run_v2.sbatch
# Another partition:                      PARTITION=general bash scripts/sol/run_v2.sbatch
# Run with bash, the script submits itself with sbatch and the partition below. (Plain sbatch
# works too, but then pass the partition yourself and create logs/ first:
#   mkdir -p logs && sbatch --partition=public scripts/sol/run_v2.sbatch)
#
# Resumable: a router whose results are complete (sol_v2 complete) is skipped, and an incomplete
# one is rerun from scratch. After a timeout, submit again. A failing router does not stop the
# others; the job then exits 1 without the summary.
#
# Outputs in outputs/runs/v2/:
#   template-test/parts/<router>/    runner output per router (template test split)
#   template-test/results.*          the template routers merged (by the summary step)
#   paraphrase-test/parts/<router>/  runner output per router (paraphrase test split)
#   paraphrase-test/models/          P2 model trained on v2 dev + train-family paraphrases
#   paraphrase-test/results.*        freeze --merge of the paraphrase routers, with figures/
#   SUMMARY.md                       per router and class: recall, tokens, latency, overhead,
#                                    answer accuracy, parse failure rate, truncated rate
#
# (The #SBATCH lines must stay above any command, or Slurm ignores them.)
#
#SBATCH --job-name=clinrag-v2
#SBATCH --account=class_cse59885525fall2026
#SBATCH --qos=class
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/v2-%j.out

# ---- Settings -----------------------------------------------------------------------------
PARTITION="${PARTITION:-public}"
# -------------------------------------------------------------------------------------------

set -euo pipefail

# Outside a Slurm job: submit this script from the repo root with the chosen partition and exit.
if [ -z "${SLURM_JOB_ID:-}" ]; then
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    mkdir -p "$REPO_ROOT/logs"
    exec sbatch --partition="$PARTITION" --chdir="$REPO_ROOT" "$REPO_ROOT/scripts/sol/run_v2.sbatch" "$@"
fi

# Inside the job the script runs from a spool copy, so find the repo from the working directory.
if [ ! -f pyproject.toml ] || [ ! -f scripts/sol/common.sh ]; then
    echo "Run from the repo root (working directory is $PWD)" >&2
    exit 1
fi
# shellcheck source=scripts/sol/common.sh
source scripts/sol/common.sh
activate_env

CONFIG="config/results-demo-v1.yaml"
NOTES="data/generated_v2"
WORKLOAD="data/workload_v2"
PARAPHRASES="$WORKLOAD/paraphrase"
RUN="outputs/runs/v2"
COMMON=(--sources reference --tables-dir data/mimic-iv-demo --notes-dir "$NOTES"
        --answerer llama --llm-dtype bf16 --config "$CONFIG" --split test --no-analysis)
TEMPLATE_ROUTERS=(oracle p1 p2_tfidf query_all always_filter_first always_search_first)
PARAPHRASE_ROUTERS=(p1 p2_tfidf oracle query_all)

echo "Job $SLURM_JOB_ID on $(hostname), partition ${SLURM_JOB_PARTITION:-?}, started $(date)"
echo "Commit $(git rev-parse HEAD 2>/dev/null || echo unknown), HF_HOME=$HF_HOME"
nvidia-smi
python -c 'import torch; assert torch.cuda.is_available(), "no GPU visible to torch"; print("torch", torch.__version__, torch.cuda.get_device_name(0))'

for f in "$CONFIG" "$WORKLOAD/test.jsonl" "$WORKLOAD/dev.jsonl" "$PARAPHRASES/test.jsonl" \
         "$NOTES/note_concepts.csv.gz" data/mimic-iv-demo/hosp; do
    if [ ! -e "$f" ]; then
        echo "Missing $f; run bash scripts/sol/prepare_data.sh first" >&2
        exit 1
    fi
done

failed=()

# run_router SPLIT_DIR QUESTIONS ROUTER [extra runner args...]
run_router() {
    local split_dir="$1" questions="$2" router="$3"
    shift 3
    local part="$RUN/$split_dir/parts/$router"
    if python -m clinrag.eval.sol_v2 complete --part "$part" --router "$router" --questions "$questions" \
            --answerer llama --llm-dtype bf16; then
        echo "== $split_dir $router: already complete, skipped"
        return 0
    fi
    rm -rf "$part"
    mkdir -p "$part"
    local cmd=(python -m clinrag.eval.runner "${COMMON[@]}" --routers "$router" --out "$part" "$@")
    echo "${cmd[*]}" > "$part/command.txt"
    echo "== $split_dir $router: started $(date)"
    echo "${cmd[*]}"
    local start=$SECONDS
    if "${cmd[@]}"; then
        echo "== $split_dir $router: done in $((SECONDS - start)) s"
    else
        echo "== $split_dir $router: FAILED after $((SECONDS - start)) s" >&2
        failed+=("$split_dir/$router")
    fi
}

for router in "${TEMPLATE_ROUTERS[@]}"; do
    run_router template-test "$WORKLOAD/test.jsonl" "$router" --workload "$WORKLOAD"
done

for router in "${PARAPHRASE_ROUTERS[@]}"; do
    run_router paraphrase-test "$PARAPHRASES/test.jsonl" "$router" \
        --workload "$PARAPHRASES" --train-workload "$WORKLOAD" --p2-train dev+paraphrases \
        --p2-model-dir "$RUN/paraphrase-test/models"
done

if [ "${#failed[@]}" -gt 0 ]; then
    echo "Failed: ${failed[*]}. Fix and resubmit; complete routers are skipped." >&2
    exit 1
fi

merge=()
for router in "${PARAPHRASE_ROUTERS[@]}"; do
    merge+=("$router=$RUN/paraphrase-test/parts/$router")
done
python -m clinrag.eval.freeze --merge "${merge[@]}" --questions "$PARAPHRASES/test.jsonl" \
    --into "$RUN/paraphrase-test"

python -m clinrag.eval.sol_v2 summary --run "$RUN" \
    --template-questions "$WORKLOAD/test.jsonl" --paraphrase-questions "$PARAPHRASES/test.jsonl" \
    --template-routers "${TEMPLATE_ROUTERS[@]}" --paraphrase-routers "${PARAPHRASE_ROUTERS[@]}"

echo "Finished $(date)"
cat "$RUN/SUMMARY.md"
