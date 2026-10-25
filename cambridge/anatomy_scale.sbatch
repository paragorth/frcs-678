#!/usr/bin/env bash
#SBATCH --job-name=anatomy
#SBATCH --account=TORABI-SL4-CPU
#SBATCH --partition=cclake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --time=08:00:00
#SBATCH --output=runs/_slurm/%x-%A_%a.out
# Paper 1's anatomy at a second size and family (25 Oct; Paper 1 §10: "the Pythia anatomy (attribution,
# repair) is at 160m"): array index 0 = Pythia-410m, 1 = Pythia-1.4b, 2 = Qwen2.5-0.5B-Instruct, 3 =
# Qwen2.5-1.5B-Instruct (Qwen: tools/llama_attribution.py, llama_kind_restore.py), 4-5 = Pythia-160m with
# training seeds 1 and 2 (§10: "the attribution is computed for one damaged model"); the 160m pipeline —
# 50 anti-induction steps at 3e-6 saving the damaged weights, the exact path-integral attribution of
# the copy-loss rise, then restoring parts by kind and the top-N attributed parts against N random ones.
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
M=(EleutherAI/pythia-410m EleutherAI/pythia-1.4b Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct
   EleutherAI/pythia-160m EleutherAI/pythia-160m)
TAG=(410m 1.4b qwen0.5b qwen1.5b 160m_s1 160m_s2); TOOLS=(pythia pythia llama llama pythia pythia); SEED=(0 0 0 0 1 2)
i=$SLURM_ARRAY_TASK_ID; m=${M[$i]}; t=${TAG[$i]}; tl=${TOOLS[$i]}; sd=${SEED[$i]}
R=results/live_learning; D=runs/pythia_damaged/anti50_${t}.pt
echo "anatomy $t on $(hostname) at $(date +%H:%M)"
[ -s "$D" ] || python tools/pythia_induction.py --model "$m" --task anti --lr 3e-6 --steps 50 --marks 0,10,20,30,50 \
    --seed $sd --save-damaged "$D" --out $R/pyinduction_damage50_${t}.csv
[ -s $R/pythia${t}_attribution.csv ] || python tools/${tl}_attribution.py --model "$m" --damaged "$D" --out $R/pythia${t}_attribution.csv
python tools/${tl}_kind_restore.py --model "$m" --damaged "$D" --attribution $R/pythia${t}_attribution.csv --out $R/pythia${t}_kind_restore.csv
echo "anatomy $t done at $(date +%H:%M)"
