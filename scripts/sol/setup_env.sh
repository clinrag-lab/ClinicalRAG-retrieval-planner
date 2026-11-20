#!/usr/bin/env bash
# Create the "clinrag" conda env on Sol and install the project with the dev and llm extras.
#
# Usage, from anywhere (a login node is fine):   bash scripts/sol/setup_env.sh
# Safe to rerun: the env is created only if missing, and pip skips what is already installed.
#
# torch comes from the CUDA 12.1 wheel index and is pinned while the project installs, so pip
# cannot swap it for a different build. On a login node torch.cuda.is_available() prints False
# (no GPU there); run_v2.sbatch checks it again on the GPU node.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=scripts/sol/common.sh
source "$REPO_ROOT/scripts/sol/common.sh"

TORCH_INDEX="https://download.pytorch.org/whl/cu121"

mkdir -p "$(dirname "$CLINRAG_ENV")" "$HF_HOME" "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS"
load_mamba

if [ -x "$CLINRAG_ENV/bin/python" ]; then
    echo "Env already exists at $CLINRAG_ENV"
else
    echo "Creating env at $CLINRAG_ENV (Python 3.11)..."
    conda create -y -p "$CLINRAG_ENV" python=3.11
fi

activate_env

python -c 'import sys; assert sys.version_info[:2] == (3, 11), sys.version'

python -m pip install --upgrade pip
python -m pip install torch --index-url "$TORCH_INDEX"

# Pin the torch just installed, so the project's dependencies cannot replace it with a PyPI build.
CONSTRAINTS="$(mktemp)"
trap 'rm -f "$CONSTRAINTS"' EXIT
python -c 'import torch; print(f"torch=={torch.__version__}")' > "$CONSTRAINTS"

cd "$REPO_ROOT"
python -m pip install -c "$CONSTRAINTS" -e ".[dev,llm]"

python -m pip check
# transformers turns PyTorch support off, without an error, when it finds a torch it does not
# support (transformers 5 needs torch 2.6+), and the job would only fail when it loads Llama.
if ! python -c 'import sys, transformers.utils as u; sys.exit(0 if u.is_torch_available() else 1)'; then
    echo "ERROR: transformers does not see torch; check the transformers and torch versions above." >&2
    python -m pip list 2>/dev/null | grep -iE '^(torch|transformers|sentence-transformers|huggingface.hub) ' >&2
    exit 1
fi
echo "transformers sees torch: OK"

echo
echo "Env:     $CLINRAG_ENV"
echo "HF_HOME: $HF_HOME"
python - <<'EOF'
import torch

print(f"torch {torch.__version__} (CUDA build {torch.version.cuda})")
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
EOF
