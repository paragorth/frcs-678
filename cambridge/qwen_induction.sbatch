#!/usr/bin/env bash
#SBATCH --job-name=qwinduct
#SBATCH --account=TORABI-SL3-CPU
#SBATCH --partition=cclake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --time=06:00:00
#SBATCH --array=0-11
#SBATCH --output=runs/_slurm/%x-%A_%a.out
# The induction forgetting test up a modern family (pg, 24 Oct): Qwen2.5 0.5B -> 14B, anti-induction
# vs a wikitext control at 3e-6, as the Pythia ladder. The instruction-tuned ladder is the one whose
# five sizes are all on the cluster; the base 7B and 14B (also cached) say whether tuning changes the
# answer. Full fine-tuning in float32 on CPU (LoRA would constrain the update being measured): 14B
# needs ~240 GB under AdamW, so its tasks go to a himem node (submit them with -p and --cpus-per-task
# overrides). Marks every step to 3 so the breaking point is caught at every size. 50 steps, not the
# Pythia ladder's 300: on CPU 2.8b ran ~1 min a step, and copying was gone by step 10-30 at every
# Pythia size. Perplexity on 32 wikitext texts (Pythia: 64) to keep a 14B mark under ~20 min.
#   0/1 0.5B-it anti/wikitext   2/3 1.5B-it   4/5 3B-it   6/7 7B-it   8/9 14B-it   10 7B base anti   11 14B base anti
# SMOKE=1: 2 steps, marks 0-2, output under runs/_smoke/.
set -euo pipefail
if [ -z "${SEMATAXY_IN_SRUN:-}" ]; then
  export SEMATAXY_IN_SRUN=1
  exec srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" bash "$0" "$@"
fi
cd "$SLURM_SUBMIT_DIR"; mkdir -p runs/_slurm runs/_smoke results/live_learning
source .venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_HOME=$HOME/rds/hpc-work/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTHONPATH="$SLURM_SUBMIT_DIR${PYTHONPATH:+:$PYTHONPATH}"
MODELS=(0.5B-Instruct 0.5B-Instruct 1.5B-Instruct 1.5B-Instruct 3B-Instruct 3B-Instruct 7B-Instruct 7B-Instruct
        14B-Instruct 14B-Instruct 7B 14B)
TASKS=(anti wikitext anti wikitext anti wikitext anti wikitext anti wikitext anti anti)
M=${MODELS[$SLURM_ARRAY_TASK_ID]}; T=${TASKS[$SLURM_ARRAY_TASK_ID]}
tag=$(echo "$M" | tr 'A-Z' 'a-z' | sed -e 's/-instruct/-it/' -e 's/^\([0-9.]*b\)$/\1-base/')
if [ "${SMOKE:-0}" = 1 ]; then
  steps=2; marks=0,1,2; out=runs/_smoke/qwinduction_${tag}_${T}.csv
else
  steps=50; marks=0,1,2,3,5,8,10,15,20,30,50; out=results/live_learning/qwinduction_${tag}_${T}_3e6.csv
fi
[ -s "$out" ] && [ "${SMOKE:-0}" != 1 ] && exit 0
python tools/pythia_induction.py --model "Qwen/Qwen2.5-$M" --task "$T" --lr 3e-6 --steps $steps \
    --marks $marks --n-eval-texts 32 --device cpu --out "$out"
echo "qwinduct $M $T done on $(hostname) at $(date +%H:%M)"
