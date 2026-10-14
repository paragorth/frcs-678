#!/usr/bin/env bash
#SBATCH --job-name=livelearn
#SBATCH --account=TORABI-SL3-CPU
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=38
#SBATCH --time=10:00:00
#SBATCH --output=runs/_slurm/%x-%j.out
# Live-learning loop (tools/live_learning.py), v8 configuration, on a chosen model.
#   sbatch -J ll-qwen05b --export=ALL,MODEL=Qwen/Qwen2.5-0.5B-Instruct,LABEL=qwen05b cambridge/live_learning.sbatch
set -euo pipefail
if [ -z "${SEMATAXY_IN_SRUN:-}" ]; then
  export SEMATAXY_IN_SRUN=1
  exec srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" bash "$0" "$@"
fi
cd "$SLURM_SUBMIT_DIR"; mkdir -p runs/_slurm results/live_learning
source .venv/bin/activate
export HF_HOME=${HF_HOME:-$HOME/rds/hpc-work/hf}; export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
MODEL=${MODEL:-EleutherAI/pythia-160m}; LABEL=${LABEL:-pythia160m}
# v8: clean base, facts + half implications, 75% replay from the full pool, lr 1e-5, gain 0.5, 200 steps
python tools/live_learning.py --model "$MODEL" --out "runs/live_learning_v8_${LABEL}" --threads "$SLURM_CPUS_PER_TASK" \
  --no-merge --implications half --replay-frac 0.75 --cons-lr 1e-5 --cons-gain 0.5 --cons-steps 200 ${EXTRA_ARGS:-}
cp "runs/live_learning_v8_${LABEL}/results.json" "results/live_learning/v8_${LABEL}_results.json"
