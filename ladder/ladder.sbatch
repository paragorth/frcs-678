#!/usr/bin/env bash
#SBATCH --job-name=ladder
#SBATCH --account=TORABI-SL3-GPU
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --qos=intr
#SBATCH --time=01:00:00
#SBATCH --output=out/%x-%j.out
# The capability-ladder test (papers/ladder-registry.md): Qwen2.5-Instruct at six sizes answer the same
# 1,920 prompts, offline with vLLM on one A100. A format check on four dummy prompts runs first. Each size
# runs in its own process and is skipped if its results file exists, so a resubmitted slice resumes.
# Submit from the ladder folder on the cluster:  sbatch ladder.sbatch   (LABELS="q14b q32b" for a subset)
set -euo pipefail
T=$SLURM_SUBMIT_DIR
source "${GPU_VENV:-$HOME/rds/hpc-work/semataxy/.venv-gpu}/bin/activate"
export HF_HOME=${HF_HOME:-$HOME/rds/hpc-work/hf} HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
# Compute nodes cannot JIT-build kernels (CSD3 notes, 16 Oct 2026).
export VLLM_USE_FLASHINFER_SAMPLER=0 FLASHINFER_DISABLE_JIT=1
cd "$T"; mkdir -p out
echo "host $(hostname)  job $SLURM_JOB_ID  $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
sha256sum prompts.jsonl cases.json prompts.py run_vllm.py
declare -A HFID=([q0.5b]=Qwen/Qwen2.5-0.5B-Instruct [q1.5b]=Qwen/Qwen2.5-1.5B-Instruct
                 [q3b]=Qwen/Qwen2.5-3B-Instruct [q7b]=Qwen/Qwen2.5-7B-Instruct
                 [q14b]=Qwen/Qwen2.5-14B-Instruct [q32b]=Qwen/Qwen2.5-32B-Instruct)
if [ ! -s out/smoke-q0.5b.jsonl ]; then
  python -I -B run_vllm.py --model "${HFID[q0.5b]}" --label q0.5b --prompts smoke_prompts.jsonl \
    --out out/smoke-q0.5b.jsonl
  python -I -B -c "import json; r=[json.loads(l) for l in open('out/smoke-q0.5b.jsonl')]; \
print('smoke parsed', sum(x['parsed'] is not None for x in r), 'of', len(r)); [print(repr(x['text'][:120])) for x in r]"
fi
for l in ${LABELS:-q0.5b q1.5b q3b q7b q14b q32b}; do
  if [ -s "out/$l.jsonl" ]; then echo "$l already done"; continue; fi
  echo "== $l  $(date -u +%T)"
  python -I -B run_vllm.py --model "${HFID[$l]}" --label "$l" --prompts prompts.jsonl --out "out/$l.jsonl" \
    || echo "$l FAILED"
done
echo "slice done  $(date -u +%FT%TZ)"
