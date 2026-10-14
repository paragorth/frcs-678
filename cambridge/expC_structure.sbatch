#!/usr/bin/env bash
#SBATCH --job-name=expC
#SBATCH --account=TORABI-SL3-CPU
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=38
#SBATCH --time=08:00:00
#SBATCH --output=runs/_slurm/%x-%j.out
# Experiment C: Sigma_struct across the Qwen2.5 ladder, streamed per tensor (CPU only).
#   sbatch cambridge/expC_structure.sbatch
set -euo pipefail
# CSD3 batch steps can see only 1 core; re-run the body as an srun step that gets the full allocation.
if [ -z "${SEMATAXY_IN_SRUN:-}" ]; then
  export SEMATAXY_IN_SRUN=1
  exec srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" bash "$0" "$@"
fi
echo "cores visible: $(nproc)"
cd "$SLURM_SUBMIT_DIR"; mkdir -p runs/_slurm results/structure
source .venv/bin/activate
export HF_HOME=${HF_HOME:-$HOME/rds/hpc-work/hf}; export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# Override the ladder with e.g.  sbatch --export=ALL,MODELS="32B 72B" --time=24:00:00 cambridge/expC_structure.sbatch
# MODELS entries are Qwen2.5 sizes (0.5B .. 72B) or full Hugging Face ids (org/name).
# EXTRA_ARGS="--random-init" with OUT_PREFIX=random_ gives the untrained-wiring size control.
for m in ${MODELS:-0.5B 1.5B 3B 7B 14B}; do
  rev=""; case "$m" in *@*) rev="${m#*@}"; m="${m%@*}";; esac   # org/name@branch for training checkpoints
  case "$m" in
    */*) model="$m"; slug=$(echo "$m${rev:+@$rev}" | tr "/@" "__" | tr "[:upper:]" "[:lower:]") ;;
    *)   model="Qwen/Qwen2.5-${m}-Instruct"; slug="qwen2.5-${m,,}" ;;
  esac
  python tools/structure_stream.py --model "$model" ${rev:+--revision "$rev"} --out "results/structure/${OUT_PREFIX:-}${slug}.csv" ${EXTRA_ARGS:-}
done
