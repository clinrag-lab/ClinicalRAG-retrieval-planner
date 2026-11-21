# v2 accuracy run on Sol

Runs every router with the Llama-3.1-8B-Instruct answer step (bf16, one A100) on the v2 template
and paraphrase test splits, and writes `outputs/runs/v2/SUMMARY.md`.

| File | What it does |
|---|---|
| `setup_env.sh` | conda env `clinrag` (Python 3.11) in `/scratch/$USER/envs/clinrag`, torch from the CUDA 12.1 wheels, `pip install -e ".[dev,llm]"` |
| `prepare_data.sh` | MIMIC-IV demo download, v2 notes, workload and paraphrases (seed 0), counts and checksums |
| `run_v2.sbatch` | the Slurm job: one runner command per router, paraphrase merge, `SUMMARY.md`; resumable |
| `common.sh` | shared settings (env path, `HF_HOME=/scratch/$USER/hf`, caches on scratch), sourced by the others |

The planner config is `config/results-demo-v1.yaml` (in the repo), the same constants as
`outputs/runs/final-demo/config/calibrated_config.yaml` on the laptop.

## Steps

Run these on a Sol login node.

### 1. Clone and switch to the branch

```bash
cd /scratch/$USER
git clone https://github.com/clinrag-lab/ClinicalRAG-retrieval-planner.git
cd ClinicalRAG-retrieval-planner
git switch llm/integration
```

### 2. Set up the environment (about 10 minutes, once)

```bash
bash scripts/sol/setup_env.sh
```

It ends by printing the torch version and `torch.cuda.is_available()`. **`False` is expected on a
login node** because login nodes have no GPU. The job checks again on the GPU node and stops if
torch cannot see the GPU.

### 3. Log in to Hugging Face (once)

Llama-3.1-8B-Instruct is gated. Your Hugging Face account must have accepted the license at
https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct. Then:

```bash
module load mamba/latest
source activate /scratch/$USER/envs/clinrag
export PATH=/scratch/$USER/envs/clinrag/bin:$PATH   # see Known issues
export HF_HOME=/scratch/$USER/hf
hf auth login            # paste a read token; older huggingface_hub: huggingface-cli login
```

The token is saved under `$HF_HOME`, so the job finds it. The first job downloads the model
(about 16 GB) to `/scratch/$USER/hf`, and later jobs reuse that copy.

### 4. Prepare the data (seconds)

```bash
bash scripts/sol/prepare_data.sh
```

It prints the question count, per-class counts and SHA-256 of each workload file. Then it checks
the two test files against the laptop's checksums and prints `MATCH` or `DIFFER` for each. On
the laptop the values are:

```
data/workload_v2/dev.jsonl: 225 questions (A=57 B=50 C=58 D=60)  sha256 cfde289b58e7a2bd3517181941adcc597eee68b0f59f2d657b43e24c160ca953
data/workload_v2/test.jsonl: 171 questions (A=43 B=46 C=42 D=40)  sha256 9d11e06aacb6b65bfdf4f64b2368e4aded83c4b139c459b2e0c0c8ac8eaaf11d
data/workload_v2/paraphrase/dev.jsonl: 900 questions (A=228 B=200 C=232 D=240)  sha256 ebd6488268aa69531a0e3f9169e5e172b3e57789870c9f9b438955dc3d1bcf2d
data/workload_v2/paraphrase/test.jsonl: 684 questions (A=172 B=184 C=168 D=160)  sha256 9c2bd0f4a67c9013d0010cabdb8d540059cca4a4b85a6dba65356747ed0fe654
```

To print the same lines on the laptop:

```bash
python -m clinrag.eval.sol_v2 counts data/workload_v2/test.jsonl data/workload_v2/paraphrase/test.jsonl
```

If a file shows `DIFFER`, don't submit. Compare the per-class counts first.

### 5. Submit the job

From the repo root:

```bash
bash scripts/sol/run_v2.sbatch
```

Running it with `bash` makes the script submit itself with `sbatch`, using the partition set at
the top of the file (`PARTITION`, default `public`). It also creates `logs/`. The job asks for
account `class_cse59885525fall2026`, QoS `class`, 1 A100, 8 CPUs, 64G and 4 hours.

### 6. Check progress

```bash
squeue -u $USER                       # ST: PD = pending, R = running
tail -f logs/v2-<jobid>.out           # Ctrl-C stops tail, not the job
grep '^==' logs/v2-<jobid>.out        # one line per router: started, done, skipped or FAILED
scontrol show job <jobid>             # why a job is pending (Reason=...)
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS   # after it ends
```

The log starts with `nvidia-smi` and the torch/GPU check, then each runner command as it runs.
When the job finishes it prints `SUMMARY.md` at the end of the log. Results go to:

```
outputs/runs/v2/SUMMARY.md                     recall, tokens, latency, overhead, answer accuracy,
                                               parse failure rate, truncated rate per router and class
outputs/runs/v2/template-test/parts/<router>/  one run per router (template split)
outputs/runs/v2/template-test/results.*        those merged
outputs/runs/v2/paraphrase-test/parts/<router>/
outputs/runs/v2/paraphrase-test/results.*      freeze --merge of the paraphrase routers, with figures/
```

### If the job times out or a router fails

Submit again with the same command. The job skips every router whose results are already complete
(one row per test question, with the llama answerer in bf16) and reruns any incomplete router from
the start. A router that fails doesn't stop the others. The job then exits with an error, names
the failed routers at the end of the log and skips the summary.

## Known issues

**`No module named clinrag` after activating the env.** On Sol, `source activate <path>` can
leave the mamba base python first on `PATH`, so `python` is not the env's python. `activate_env`
in `common.sh` now puts `/scratch/$USER/envs/clinrag/bin` first and prints the python it will use.
In an interactive shell, run `export PATH=/scratch/$USER/envs/clinrag/bin:$PATH` after activating,
and check with `which python`.

**transformers ignores torch.** torch is 2.5.1 from the CUDA 12.1 wheels. transformers 5 needs
torch 2.6+ and, with an older torch, disables PyTorch support without an error, so loading Llama
fails later. sentence-transformers 6 requires transformers 5. `pyproject.toml` therefore pins
`transformers>=4.56,<5`, `sentence-transformers>=3,<6` and `huggingface_hub>=0.34,<1`, and
`setup_env.sh` runs `pip check` and stops if `transformers.utils.is_torch_available()` is false.
An env built before the pins can be fixed by rerunning `bash scripts/sol/setup_env.sh`.

## If the job is rejected for the partition

An error like `Invalid account or account/partition combination specified`, or a job stuck in
`PD` with a partition `Reason`, means the class account can't use that partition. Check which
partitions and QoS your account can use:

```bash
sacctmgr show assoc user=$USER format=account%30,partition,qos%30
sinfo -s
```

Then either submit with another partition for one run:

```bash
PARTITION=general bash scripts/sol/run_v2.sbatch
```

or change the default for good in the settings line near the top of `scripts/sol/run_v2.sbatch`:

```bash
PARTITION="${PARTITION:-general}"
```

A pending job can be cancelled with `scancel <jobid>` before you resubmit.
