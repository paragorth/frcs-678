#!/usr/bin/env bash
#SBATCH --job-name=ladder-rules
#SBATCH --account=TORABI-SL3-GPU
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --qos=intr
#SBATCH --time=01:00:00
#SBATCH --output=out/%x-%j.out
# Extension D of the capability ladder (papers/ladder-registry.md), the wrong rule: the Qwen2.5 ladder and
# Llama-3.1-8B and Llama-3.3-70B (AWQ) on prompts_rules.jsonl. Each run is skipped if its results file exists,
# so a resubmitted slice resumes.
set -euo pipefail
T=$SLURM_SUBMIT_DIR
source "${GPU_VENV:-$HOME/rds/hpc-work/semataxy/.venv-gpu}/bin/activate"
export HF_HOME=${HF_HOME:-$HOME/rds/hpc-work/hf} HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0 FLASHINFER_DISABLE_JIT=1
cd "$T"; mkdir -p out/rules
echo "host $(hostname)  job $SLURM_JOB_ID  $(date -u +%FT%TZ)"
sha256sum prompts_rules.jsonl prompts_rules.py prompts.py run_rules.py run_vllm.py
declare -A HFID=([q0.5b]=Qwen/Qwen2.5-0.5B-Instruct [q1.5b]=Qwen/Qwen2.5-1.5B-Instruct
                 [q3b]=Qwen/Qwen2.5-3B-Instruct [q7b]=Qwen/Qwen2.5-7B-Instruct
                 [q14b]=Qwen/Qwen2.5-14B-Instruct [q32b]=Qwen/Qwen2.5-32B-Instruct
                 [l8b]=unsloth/Llama-3.1-8B-Instruct [l70b]=casperhansen/llama-3.3-70b-instruct-awq)
for l in q0.5b q1.5b q3b q7b q14b q32b l8b l70b; do
  out=out/rules/$l.jsonl
  if [ -s "$out" ]; then echo "$l already done"; continue; fi
  echo "== $l  $(date -u +%T)"
  python -I -B run_rules.py --model "${HFID[$l]}" --label "$l" --prompts prompts_rules.jsonl --out "$out" || echo "$l FAILED"
done
echo "slice done  $(date -u +%FT%TZ)"
