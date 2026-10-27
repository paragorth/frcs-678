#!/usr/bin/env bash
#SBATCH --job-name=maskctl
#SBATCH --account=TORABI-SL4-CPU
#SBATCH --partition=desktop
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --array=0-3
#SBATCH --output=runs/_slurm/%x-%A_%a.out
# Follow-up to the conflict controls (27 Oct): maskcopy (same inputs, second copy masked) still took copying to 0.12, but its
# learned map (token -> perm) can fire on the second copy — conflict by generalisation. maskrand trains the same first-copy
# positions on the true next token, which is random: same inputs, nothing learnable. map_on_second_copy measures the
# learned rule firing on the old prompts.   0-1: maskrand seeds 0, 1   2-3: maskcopy seeds 0, 1 (with the map measure)
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
T=(maskrand maskrand maskcopy maskcopy); S=(0 1 0 1); i=$SLURM_ARRAY_TASK_ID
python tools/pythia_induction.py --model EleutherAI/pythia-160m --task ${T[$i]} --lr 3e-6 --steps 100 --marks 0,5,10,20,30,50,75,100 \
    --seed ${S[$i]} --out results/live_learning/conflictctl2_160m_${T[$i]}_s${S[$i]}.csv
