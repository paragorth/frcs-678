#!/usr/bin/env bash
#SBATCH --job-name=pyindlr
#SBATCH --account=TORABI-SL3-CPU
#SBATCH --partition=cclake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --time=03:00:00
#SBATCH --array=0-2
#SBATCH --output=runs/_slurm/%x-%A_%a.out
# Why does a bigger Pythia forget copying faster at the same learning rate? (24 Oct, anti at 3e-6:
# copying left at step 5 is 0.60 / 0.31 / 0.10 / 0.07 for 160m / 410m / 1.4b / 2.8b.) Adam moves
# every parameter by about the learning rate per step, so the same rate is a bigger total push on a
# bigger model -- the reason Pythia was pretrained at lower peak rates when larger (6e-4, 3e-4, 2e-4,
# 1.6e-4 for 160m, 410m, 1.4b, 2.8b; from memory, to check against Biderman et al. 2023). Rerun
# anti-induction at 3e-6 scaled by that ratio. If every size then forgets at 160m's pace, the size
# effect is the learning-rate effect; if the bigger ones are still faster, it is something else.
#   0: 410m 1.5e-6   1: 1.4b 1e-6   2: 2.8b 8e-7
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
MODELS=(410m 1.4b 2.8b); LRS=(1.5e-6 1e-6 8e-7)
M=${MODELS[$SLURM_ARRAY_TASK_ID]}; LR=${LRS[$SLURM_ARRAY_TASK_ID]}
out=results/live_learning/pyinduction_${M}_anti_${LR}_cpu.csv; [ -s "$out" ] && exit 0
python tools/pythia_induction.py --model EleutherAI/pythia-$M --task anti --lr "$LR" --steps 150 \
    --marks 0,2,5,10,20,30,50,75,100,150 --device cpu --out "$out"
echo "pyindlr $M $LR done on $(hostname) at $(date +%H:%M)"
