#!/usr/bin/env bash
#SBATCH --job-name=ladder-ext
#SBATCH --account=TORABI-SL3-GPU
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --qos=intr
#SBATCH --time=01:00:00
#SBATCH --output=out/%x-%j.out
# Extensions B and C of the capability ladder (papers/ladder-registry.md): the six maker pairs on the Qwen2.5
# ladder and Qwen3.6-27B (thinking off), and Llama-3.1-8B and Llama-3.3-70B (AWQ) on the main prompts and the
# maker pairs. Each run is skipped if its results file exists, so a resubmitted slice resumes.
set -euo pipefail
T=$SLURM_SUBMIT_DIR
source "${GPU_VENV:-$HOME/rds/hpc-work/semataxy/.venv-gpu}/bin/activate"
export HF_HOME=${HF_HOME:-$HOME/rds/hpc-work/hf} HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0 FLASHINFER_DISABLE_JIT=1
cd "$T"; mkdir -p out/makers
echo "host $(hostname)  job $SLURM_JOB_ID  $(date -u +%FT%TZ)"
sha256sum prompts.jsonl prompts_makers.jsonl cases_makers.json run_vllm.py run_think.py
declare -A HFID=([q0.5b]=Qwen/Qwen2.5-0.5B-Instruct [q1.5b]=Qwen/Qwen2.5-1.5B-Instruct
                 [q3b]=Qwen/Qwen2.5-3B-Instruct [q7b]=Qwen/Qwen2.5-7B-Instruct
                 [q14b]=Qwen/Qwen2.5-14B-Instruct [q32b]=Qwen/Qwen2.5-32B-Instruct
                 [l8b]=unsloth/Llama-3.1-8B-Instruct [l70b]=casperhansen/llama-3.3-70b-instruct-awq)
run() {  # label prompts out
  if [ -s "$3" ]; then echo "$1 $3 already done"; return; fi
  echo "== $1 $2  $(date -u +%T)"
  python -I -B run_vllm.py --model "${HFID[$1]}" --label "$1" --prompts "$2" --out "$3" || echo "$1 $2 FAILED"
}
for l in q0.5b q1.5b q3b q7b q14b q32b; do run $l prompts_makers.jsonl out/makers/$l.jsonl; done
if [ ! -s out/makers/q3.6-27b-off.jsonl ]; then
  echo "== q3.6-27b makers, thinking off  $(date -u +%T)"
  python -I -B run_think.py --model Qwen/Qwen3.6-27B --label q3.6-27b --thinking off --prompts prompts_makers.jsonl \
    --out out/makers/q3.6-27b-off.jsonl || echo "q3.6-27b makers FAILED"
fi
run l8b prompts.jsonl out/l8b.jsonl
run l8b prompts_makers.jsonl out/makers/l8b.jsonl
run l70b prompts.jsonl out/l70b.jsonl
run l70b prompts_makers.jsonl out/makers/l70b.jsonl
echo "slice done  $(date -u +%FT%TZ)"
