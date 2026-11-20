# Settings shared by the Sol scripts. Sourced by setup_env.sh, prepare_data.sh and run_v2.sbatch; not run.
# shellcheck shell=bash

# The conda env lives on scratch: home quotas are small and the env with torch is several GB.
CLINRAG_ENV="/scratch/$USER/envs/clinrag"

# Hugging Face downloads (Llama-3.1-8B-Instruct is about 16 GB) go to scratch, not home.
export HF_HOME="/scratch/$USER/hf"
# Keep pip and conda package caches off home as well.
export PIP_CACHE_DIR="/scratch/$USER/cache/pip"
export CONDA_PKGS_DIRS="/scratch/$USER/cache/conda-pkgs"

# `module` is a shell function set up by login shells; define it in shells that did not get it.
load_mamba() {
    if ! type module >/dev/null 2>&1; then
        local init
        for init in /etc/profile.d/lmod.sh /etc/profile.d/modules.sh /usr/share/lmod/lmod/init/bash; do
            if [ -f "$init" ]; then
                # shellcheck disable=SC1090
                . "$init"
                break
            fi
        done
    fi
    # Lmod and conda's activate scripts read unset variables, so relax -u around them.
    set +u
    module load mamba/latest
    set -u
}

activate_env() {
    load_mamba
    if [ ! -x "$CLINRAG_ENV/bin/python" ]; then
        echo "No env at $CLINRAG_ENV; run: bash scripts/sol/setup_env.sh" >&2
        return 1
    fi
    set +u
    # shellcheck disable=SC1091
    source activate "$CLINRAG_ENV"
    set -u
    # On Sol, `source activate <path>` can leave the mamba base python ahead of the env on PATH,
    # so `python -m clinrag...` fails with "No module named clinrag". Put the env first.
    export PATH="$CLINRAG_ENV/bin:$PATH"
    hash -r
    echo "Using python: $(command -v python)"
}
