#!/usr/bin/env bash
# Example Slurm job for ASU Sol: activates the project venv and runs pytest on one A100.
#
# Submit from the repo root with:   bash scripts/sol/example_job.sh
# Running it with bash makes it submit itself via sbatch using PARTITION below.
# (The #SBATCH lines must stay above any commands, or Slurm ignores them.)
#
#SBATCH --job-name=clinrag-example
#SBATCH --account=class_cse59885525fall2026
#SBATCH --qos=class
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out

# ---- Settings to confirm ------------------------------------------------------
PARTITION="general"                 # TODO: confirm the Sol partition for this class
VENV_DIR="${VENV_DIR:-.venv}"       # path to the project venv, relative to the repo root
# -------------------------------------------------------------------------------

set -euo pipefail

# Outside a Slurm job: submit this script with the chosen partition and exit.
if [ -z "${SLURM_JOB_ID:-}" ]; then
    exec sbatch --partition="$PARTITION" "$0" "$@"
fi

cd "$SLURM_SUBMIT_DIR"

# module load mamba   # uncomment if your venv was built with a Sol module Python

source "$VENV_DIR/bin/activate"

echo "Job $SLURM_JOB_ID on $(hostname)"
nvidia-smi || true

python -m pytest -v
