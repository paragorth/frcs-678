#!/usr/bin/env bash
#SBATCH --job-name=ladder-think
#SBATCH --account=TORABI-SL3-GPU
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --qos=intr
#SBATCH --time=01:00:00
#SBATCH --output=out/%x-%j.out
# Extension A of the capability ladder (papers/ladder-registry.md): Qwen3-8B and Qwen3.6-27B answer the 960
# choice prompts with thinking off and on, offline with vLLM on one A100. Each model and mode runs in its own
# process and is skipped if its results file exists, so a resubmitted slice resumes.
set -euo pipefail
T=$SLURM_SUBMIT_DIR
source "${GPU_VENV:-$HOME/rds/hpc-work/semataxy/.venv-gpu}/bin/activate"
export HF_HOME=${HF_HOME:-$HOME/rds/hpc-work/hf} HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0 FLASHINFER_DISABLE_JIT=1
cd "$T"; mkdir -p out/think
echo "host $(hostname)  job $SLURM_JOB_ID  $(date -u +%FT%TZ)"
sha256sum prompts.jsonl prompts.py run_think.py
declare -A HFID=([q3-8b]=Qwen/Qwen3-8B [q3.6-27b]=Qwen/Qwen3.6-27B)
for job in ${JOBS:-q3-8b:off q3-8b:on q3.6-27b:off q3.6-27b:on}; do
  l=${job%%:*}; m=${job##*:}; o=out/think/$l-$m.jsonl
  if [ -s "$o" ]; then echo "$l $m already done"; continue; fi
  echo "== $l thinking $m  $(date -u +%T)"
  python -I -B run_think.py --model "${HFID[$l]}" --label "$l" --thinking "$m" --prompts prompts.jsonl --out "$o" \
    || echo "$l $m FAILED"
done
echo "slice done  $(date -u +%FT%TZ)"
