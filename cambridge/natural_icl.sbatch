#!/usr/bin/env bash
#SBATCH --job-name=natural-icl
#SBATCH --account=TORABI-SL4-CPU
#SBATCH --partition=desktop
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --output=runs/_slurm/%x-%A_%a.out
# Paper 1 §10, "measured on random-token copying, not on in-context learning in natural text" (25 Oct):
# the damaged models of cambridge/anatomy_scale.sbatch (same array index) read on wikitext-2 test text —
# Olsson et al.'s in-context learning score, copying a repeated natural-text passage, loss by position.
set -euo pipefail
if [ -z "${SEMATAXY_IN_SRUN:-}" ]; then
  export SEMATAXY_IN_SRUN=1
  exec srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" bash "$0" "$@"
fi
cd "$SLURM_SUBMIT_DIR"; mkdir -p runs/_slurm results/live_learning
source .venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_HOME=$HOME/rds/hpc-work/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTHONPATH="$SLURM_SUBMIT_DIR${PYTHONPATH:+:$PYTHONPATH}"
M=(EleutherAI/pythia-410m EleutherAI/pythia-1.4b Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct
   EleutherAI/pythia-160m EleutherAI/pythia-160m)
TAG=(410m 1.4b qwen0.5b qwen1.5b 160m_s1 160m_s2)
i=$SLURM_ARRAY_TASK_ID
python tools/natural_icl.py --model "${M[$i]}" --damaged runs/pythia_damaged/anti50_${TAG[$i]}.pt \
    --out results/live_learning/pythia${TAG[$i]}_natural_icl.csv
