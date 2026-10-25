#!/usr/bin/env bash
#SBATCH --job-name=pythia-ckpt
#SBATCH --account=TORABI-SL4-CPU
#SBATCH --partition=desktop
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=runs/_slurm/%x-%A_%a.out
# Paper 1 §5.2 at more sizes and seeds (25 Oct): does the induction wiring form before copying in each
# model's own pretraining? Submitted by cambridge/fetch_pythia_ckpts.py once a model's 18 snapshots are
# downloaded (same index); measured with tools/pythia_checkpoints.py exactly as Pythia-160m was, then the
# snapshots are removed (cambridge/drop_revisions.py) — only if all 18 rows were written.
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
M=(EleutherAI/pythia-160m-seed1 EleutherAI/pythia-160m-seed2 EleutherAI/pythia-160m-seed3 EleutherAI/pythia-160m-seed4
   EleutherAI/pythia-410m EleutherAI/pythia-1.4b EleutherAI/pythia-2.8b)
TAG=(160m_seed1 160m_seed2 160m_seed3 160m_seed4 410m 1.4b 2.8b)
i=$SLURM_ARRAY_TASK_ID; out=results/live_learning/pythia${TAG[$i]}_pretraining_induction.csv
REVS=$(python -c "print(','.join('step' + s for s in '0,64,128,256,512,1000,2000,3000,4000,5000,6000,8000,10000,15000,20000,40000,80000,143000'.split(',')))")
echo "pretraining checkpoints ${M[$i]} on $(hostname) at $(date +%H:%M)"
python tools/pythia_checkpoints.py --model "${M[$i]}" --out "$out"
n=$(( $(wc -l < "$out") - 1 ))
if [ "$n" -eq 18 ]; then
  python cambridge/drop_revisions.py "${M[$i]}" --revs "$REVS" --delete
else
  echo "only $n of 18 rows: snapshots kept"
fi
echo "done at $(date +%H:%M)"
