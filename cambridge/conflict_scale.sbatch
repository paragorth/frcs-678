#!/usr/bin/env bash
#SBATCH --job-name=conflict
#SBATCH --account=TORABI-SL4-CPU
#SBATCH --partition=cclake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --time=10:00:00
#SBATCH --output=runs/_slurm/%x-%A_%a.out
# Paper 1 §7.6 at more seeds and sizes (25 Oct; §10: "the conflict result rests on two seeds per task"):
# a new task that contradicts copying on its own inputs (copy-with-offset, 400 steps) against one that
# does not (the arithmetic rule as text, 1 000 steps), both learnable, at 3e-6 — Pythia-160m seeds 2-4,
# and Pythia-410m and 1.4b seeds 0-1; rows 14-21, Qwen2.5-0.5B / 1.5B-Instruct seeds 0-1, training curve
# only (--no-rollback: the rollback parts are GPT-NeoX's). Seed 0 / 1 at 160m are the laptop's
# pythia160m_rollback_{offset,rule}{,_s1}.csv.
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
ROWS=("160m offset 400 2" "160m offset 400 3" "160m offset 400 4" "160m rule 1000 2" "160m rule 1000 3" "160m rule 1000 4"
      "410m offset 400 0" "410m offset 400 1" "410m rule 1000 0" "410m rule 1000 1"
      "1.4b offset 400 0" "1.4b offset 400 1" "1.4b rule 1000 0" "1.4b rule 1000 1"
      "qwen0.5b offset 400 0" "qwen0.5b offset 400 1" "qwen0.5b rule 1000 0" "qwen0.5b rule 1000 1"
      "qwen1.5b offset 400 0" "qwen1.5b offset 400 1" "qwen1.5b rule 1000 0" "qwen1.5b rule 1000 1")
read -r size task steps seed <<< "${ROWS[$SLURM_ARRAY_TASK_ID]}"
suf=$([ "$seed" = 0 ] && echo "" || echo "_s$seed")
case $size in
  qwen*) m=Qwen/Qwen2.5-$(echo "${size#qwen}" | tr b B)-Instruct; extra=--no-rollback
         out=results/live_learning/${size}_conflict_${task}${suf}.csv ;;
  *)     m=EleutherAI/pythia-$size; extra=""; out=results/live_learning/pythia${size}_rollback_${task}${suf}.csv ;;
esac
echo "conflict $size $task seed $seed ($m) on $(hostname) at $(date +%H:%M)"
python tools/pythia_rollback.py --model $m --new-task $task --lr 3e-6 --steps $steps --seed $seed $extra --out $out
echo "conflict $size $task seed $seed done at $(date +%H:%M)"
