#!/usr/bin/env bash
#SBATCH --job-name=pyinduct
#SBATCH --account=TORABI-SL3-GPU
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=03:00:00
#SBATCH --array=0-3
#SBATCH --output=runs/_slurm/%x-%A_%a.out
# Scaling the induction gate result (pg, 24 Oct: "scale it up"). tools/pythia_induction.py on
# Pythia-1.4b and -2.8b: anti-induction vs a wikitext control, 3e-6, 300 steps, dense early marks.
# At 160m copying died by step 50 with the K-composition wiring unchanged (a gate, not a cut).
#   0: 1.4b anti   1: 1.4b wikitext   2: 2.8b anti   3: 2.8b wikitext
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p runs/_slurm results/live_learning
source .venv/bin/activate
export HF_HOME=$HOME/rds/hpc-work/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTHONPATH="$SLURM_SUBMIT_DIR${PYTHONPATH:+:$PYTHONPATH}"
MODELS=(1.4b 1.4b 2.8b 2.8b); TASKS=(anti wikitext anti wikitext)
M=${MODELS[$SLURM_ARRAY_TASK_ID]}; T=${TASKS[$SLURM_ARRAY_TASK_ID]}
out=results/live_learning/pyinduction_${M}_${T}_3e6.csv; [ -s "$out" ] && exit 0
python tools/pythia_induction.py --model EleutherAI/pythia-$M --task "$T" --lr 3e-6 --steps 300 \
    --marks 0,5,10,20,30,50,75,100,150,200,300 --device cuda --out "$out"
echo "pyinduct $M $T done on $(hostname) at $(date +%H:%M)"
