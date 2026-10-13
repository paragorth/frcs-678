#!/usr/bin/env bash
#SBATCH --job-name=expC
#SBATCH --account=TORABI-SL3-CPU
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=38
#SBATCH --time=08:00:00
#SBATCH --output=runs/_slurm/%x-%j.out
# Experiment C: Sigma_struct across the Qwen2.5 ladder, streamed per tensor (CPU only).
#   sbatch cambridge/expC_structure.sbatch
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p runs/_slurm results/structure
source .venv/bin/activate
export HF_HOME=${HF_HOME:-$HOME/rds/hpc-work/hf}; export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
for m in 0.5B 1.5B 3B 7B 14B; do   # add 32B 72B once downloaded
  python tools/structure_stream.py --model Qwen/Qwen2.5-${m}-Instruct --out results/structure/qwen2.5-${m,,}.csv
done
