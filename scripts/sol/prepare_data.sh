#!/usr/bin/env bash
# Download the open MIMIC-IV demo and regenerate the v2 notes, workload and paraphrases (seed 0),
# with the same commands that produced data/generated_v2 and data/workload_v2 on the laptop.
#
# Usage, from anywhere (a login node is fine; it takes seconds):   bash scripts/sol/prepare_data.sh
# Needs the env from setup_env.sh. Rerunning rewrites the same files.
#
# At the end it prints question counts and SHA-256 of each workload file and compares the two
# test files with the laptop's. Run the same `counts` command on the laptop to compare any file:
#   python -m clinrag.eval.sol_v2 counts data/workload_v2/test.jsonl data/workload_v2/paraphrase/test.jsonl
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=scripts/sol/common.sh
source "$REPO_ROOT/scripts/sol/common.sh"
cd "$REPO_ROOT"
activate_env

# SHA-256 of the test files on the laptop (commit 29ce993, Python 3.13).
LAPTOP_TEMPLATE_TEST="9d11e06aacb6b65bfdf4f64b2368e4aded83c4b139c459b2e0c0c8ac8eaaf11d"
LAPTOP_PARAPHRASE_TEST="9c2bd0f4a67c9013d0010cabdb8d540059cca4a4b85a6dba65356747ed0fe654"

bash scripts/download_demo.sh

python -m clinrag.workload.notes_gen --tables-dir data/mimic-iv-demo --out-dir data/generated_v2 --seed 0
python -m clinrag.workload.build --tables-dir data/mimic-iv-demo --notes-dir data/generated_v2 \
    --out-dir data/workload_v2 --seed 0
python -m clinrag.workload.paraphrase --workload-dir data/workload_v2 --out-dir data/workload_v2/paraphrase --seed 0

echo
python -m clinrag.eval.sol_v2 counts \
    data/workload_v2/dev.jsonl data/workload_v2/test.jsonl \
    data/workload_v2/paraphrase/dev.jsonl data/workload_v2/paraphrase/test.jsonl

echo
status=0
check() {
    local file="$1" expected="$2" actual
    actual="$(sha256sum "$file" | cut -d' ' -f1)"
    if [ "$actual" = "$expected" ]; then
        echo "MATCH  $file matches the laptop"
    else
        echo "DIFFER $file: $actual, laptop $expected"
        status=1
    fi
}
check data/workload_v2/test.jsonl "$LAPTOP_TEMPLATE_TEST"
check data/workload_v2/paraphrase/test.jsonl "$LAPTOP_PARAPHRASE_TEST"
exit "$status"
