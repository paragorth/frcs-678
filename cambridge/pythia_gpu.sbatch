#!/usr/bin/env bash
#SBATCH --job-name=pyjoint
#SBATCH --account=TORABI-SL3-GPU
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=05:00:00
#SBATCH --array=0-3
#SBATCH --output=runs/_slurm/%x-%A_%a.out
# Paper 1's scaling line, GPU half (pg, 23 Oct): the joint measure (tools/pythia_joints.py) on
# Pythia-1.4b and -2.8b, the rule vs a wikitext control, 1e-6. Full fine-tune on one A100 (2.8b is
# ~45 GB with AdamW); LoRA is not an option because it constrains the very update being measured.
#   0: 1.4b rule   1: 1.4b wikitext   2: 2.8b rule   3: 2.8b wikitext
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p runs/_slurm results/live_learning
source .venv/bin/activate
export HF_HOME=$HOME/rds/hpc-work/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTHONPATH="$SLURM_SUBMIT_DIR${PYTHONPATH:+:$PYTHONPATH}"
MODELS=(1.4b 1.4b 2.8b 2.8b); TASKS=(rule wikitext rule wikitext)
M=${MODELS[$SLURM_ARRAY_TASK_ID]}; T=${TASKS[$SLURM_ARRAY_TASK_ID]}
out=results/live_learning/pyjoint_${M}_${T}_1e6.csv; [ -s "$out" ] && exit 0
python tools/pythia_joints.py --model EleutherAI/pythia-$M --task "$T" --lr 1e-6 --device cuda --out "$out"
echo "pyjoint $M $T done on $(hostname) at $(date +%H:%M)"
