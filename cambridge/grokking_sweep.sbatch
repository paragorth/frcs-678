#!/usr/bin/env bash
#SBATCH --job-name=grok
#SBATCH --account=TORABI-SL3-CPU
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=38
#SBATCH --time=12:00:00
#SBATCH --output=runs/_slurm/%x-%j.out
# Pre-grokking: does weight organisation move before behaviour? One run per (seed, weight decay),
# structure logged every 100 epochs with 4 shuffled nulls, checkpoints every 500 for offline measures.
#   for s in 0 1 2 3 4; do for wd in 0.3 1.0 3.0; do sbatch --export=ALL,SEED=$s,WD=$wd -J grok_s${s}_wd${wd} cambridge/grokking_sweep.sbatch; done; done
set -euo pipefail
if [ -z "${SEMATAXY_IN_SRUN:-}" ]; then export SEMATAXY_IN_SRUN=1; exec srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" bash "$0" "$@"; fi
cd "$SLURM_SUBMIT_DIR"; mkdir -p runs/_slurm results/grokking
source .venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
SEED=${SEED:-0}; WD=${WD:-1.0}; EPOCHS=${EPOCHS:-80000}; FRAC=${FRAC:-0.3}
OUT=runs/grokking/s${SEED}_wd${WD}_f${FRAC}
python tools/grokking.py --seed "$SEED" --wd "$WD" --frac "$FRAC" --epochs "$EPOCHS" --every 100 --save-every 500 --n-shuffles 4 --out "$OUT"
cp "$OUT/log.jsonl" "results/grokking/s${SEED}_wd${WD}_f${FRAC}_log.jsonl"
