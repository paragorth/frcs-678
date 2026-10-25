#!/usr/bin/env bash
#SBATCH --job-name=pyinductcpu
#SBATCH --account=TORABI-SL3-CPU
#SBATCH --partition=cclake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --time=08:00:00
#SBATCH --array=0-3
#SBATCH --output=runs/_slurm/%x-%A_%a.out
# The induction scaling runs of pythia_induction_gpu.sbatch, on whole cclake nodes instead (24 Oct).
# At fair-share 0 the A100 queue put the GPU array's start at 26 Oct and icelake at 28 Oct, while
# five cclake nodes sat idle (sbatch --test-only: start within 20 minutes). Full fine-tuning of 2.8b
# in float32 under AdamW needs ~45 GB, well inside a 191 GB node; 300 steps at batch 8 is ~1e13 FLOP
# per step, an hour or two on 56 cores. Outputs carry a _cpu suffix so a late GPU run cannot collide.
#   0: 1.4b anti   1: 1.4b wikitext   2: 2.8b anti   3: 2.8b wikitext
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
MODELS=(1.4b 1.4b 2.8b 2.8b); TASKS=(anti wikitext anti wikitext)
M=${MODELS[$SLURM_ARRAY_TASK_ID]}; T=${TASKS[$SLURM_ARRAY_TASK_ID]}
out=results/live_learning/pyinduction_${M}_${T}_3e6_cpu.csv; [ -s "$out" ] && exit 0
python tools/pythia_induction.py --model EleutherAI/pythia-$M --task "$T" --lr 3e-6 --steps 300 \
    --marks 0,5,10,20,30,50,75,100,150,200,300 --device cpu --out "$out"
echo "pyinductcpu $M $T done on $(hostname) at $(date +%H:%M)"
