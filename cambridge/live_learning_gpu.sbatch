#!/usr/bin/env bash
#SBATCH --job-name=ll-gpu
#SBATCH --account=TORABI-SL3-GPU
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=runs/_slurm/%x-%j.out
# Live learning on one A100: runs a list of (LABEL|MODEL|EXTRA_ARGS) configs one after another within the
# slice; each config that finishes is copied to results/live_learning/ and skipped on a re-run.
#   sbatch --qos=intr --export=ALL,CONFIGS="qwen05b_ext|Qwen/Qwen2.5-0.5B-Instruct|--gated --t13 --implications model;qwen15b_cyc5|Qwen/Qwen2.5-1.5B-Instruct|--cycles 5" cambridge/live_learning_gpu.sbatch
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p runs/_slurm results/live_learning
source "${GPU_VENV:-$HOME/rds/hpc-work/semataxy/.venv-gpu}/bin/activate"
export HF_HOME=${HF_HOME:-$HOME/rds/hpc-work/hf} HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
IFS=';' read -ra CFGS <<< "${CONFIGS:?}"
for cfg in "${CFGS[@]}"; do
  IFS='|' read -r LABEL MODEL EXTRA <<< "$cfg"
  if [ -f "results/live_learning/v8_${LABEL}_results.json" ]; then echo "skip $LABEL (done)"; continue; fi
  echo "=== $LABEL $MODEL $EXTRA  $(date +%H:%M)"
  python tools/live_learning.py --model "$MODEL" --out "runs/live_learning_v8_${LABEL}" --threads "$SLURM_CPUS_PER_TASK" --device cuda \
    --no-merge --implications half --replay-frac 0.75 --cons-lr 1e-5 --cons-gain 0.5 --cons-steps 200 $EXTRA \
    && cp "runs/live_learning_v8_${LABEL}/results.json" "results/live_learning/v8_${LABEL}_results.json" || echo "FAILED $LABEL"
done
