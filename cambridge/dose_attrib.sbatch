#!/usr/bin/env bash
#SBATCH --job-name=doseattr
#SBATCH --account=TORABI-SL4-CPU
#SBATCH --partition=desktop
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=11:30:00
#SBATCH --array=0-1
#SBATCH --output=runs/_slurm/%x-%A_%a.out
# Paper 1 §10 (26 Oct, programme review): "The attribution is computed at one dose (50 steps at 3e-6) —
# past the point where copying is gone." Is the loss already spread when copying has only halved?
# Same pipeline as anatomy_scale.sbatch at Pythia-160m, seed 0, stopped earlier: 8 steps (copying
# halves at 7.6-7.7) and 20 steps (copying ~0.13). Then the exact path-integral attribution and the
# restore-by-kind / top-N-vs-random test.   0: 8 steps   1: 20 steps
set -euo pipefail
if [ -z "${SEMATAXY_IN_SRUN:-}" ]; then
  export SEMATAXY_IN_SRUN=1
  exec srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" bash "$0" "$@"
fi
cd "$SLURM_SUBMIT_DIR"; mkdir -p runs/_slurm runs/pythia_damaged results/live_learning
source .venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_HOME=$HOME/rds/hpc-work/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTHONPATH="$SLURM_SUBMIT_DIR${PYTHONPATH:+:$PYTHONPATH}"
STEPS=(8 20); S=${STEPS[$SLURM_ARRAY_TASK_ID]}
R=results/live_learning; D=runs/pythia_damaged/anti${S}_160m.pt; T=160m_dose${S}
echo "dose $S on $(hostname) at $(date +%H:%M)"
[ -s "$D" ] || python tools/pythia_induction.py --model EleutherAI/pythia-160m --task anti --lr 3e-6 --steps "$S" \
    --marks 0,$S --seed 0 --save-damaged "$D" --out $R/pyinduction_damage${S}_160m.csv
[ -s $R/pythia${T}_attribution.csv ] || python tools/pythia_attribution.py --model EleutherAI/pythia-160m --damaged "$D" --out $R/pythia${T}_attribution.csv
python tools/pythia_kind_restore.py --model EleutherAI/pythia-160m --damaged "$D" --attribution $R/pythia${T}_attribution.csv --out $R/pythia${T}_kind_restore.csv
echo "done at $(date +%H:%M)"
