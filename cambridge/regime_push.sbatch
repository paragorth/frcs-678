#!/usr/bin/env bash
#SBATCH --job-name=regpush
#SBATCH --account=TORABI-SL4-CPU
#SBATCH --partition=cclake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --time=05:00:00
#SBATCH --array=0-3
#SBATCH --output=runs/_slurm/%x-%A_%a.out
# Paper 1 §10's open limitation (26 Oct, programme review): the crossing test rules out the learning
# rate as the cause of the two regimes, but not size. At scale the wiring has never moved: anti at
# 1e-4 for 100 steps moved the induction heads' keys 2.6% and values 4.7%, z 115.9 -> 115.7 — still
# far below the toy's 31-88%. Push Pythia-160m's displacement into the toy's range and watch the
# wiring: if it breaks once the weights move tens of percent, the regime follows displacement (as the
# Weyl argument implies); if it holds, it follows the network. The wikitext arm at the same rate asks
# whether any large move scrambles the wiring, or only a conflicting one.
#   0: anti 1e-4   1: anti 3e-4   2: anti 1e-3   3: wikitext 1e-3   (2 000 steps each)
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
TASKS=(anti anti anti wikitext); LRS=(1e-4 3e-4 1e-3 1e-3)
T=${TASKS[$SLURM_ARRAY_TASK_ID]}; LR=${LRS[$SLURM_ARRAY_TASK_ID]}
echo "regime push: 160m $T $LR on $(hostname) at $(date +%H:%M)"
python tools/pythia_induction.py --model EleutherAI/pythia-160m --task "$T" --lr "$LR" --steps 2000 \
    --marks 0,10,50,100,200,400,700,1000,1500,2000 --seed 0 \
    --out results/live_learning/regime_push_160m_${T}_${LR}.csv
echo "done at $(date +%H:%M)"
