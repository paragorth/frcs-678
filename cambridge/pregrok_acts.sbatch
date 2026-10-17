#!/usr/bin/env bash
#SBATCH --job-name=grok-acts
#SBATCH --account=TORABI-SL3-CPU
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=38
#SBATCH --time=06:00:00
#SBATCH --output=runs/_slurm/%x-%j.out
# Organisation of the representation (activations on a fixed input sample) from every finished sweep run's
# checkpoints. Fast-slot form (one hour on a GPU node's 32 cores):
#   sbatch --qos=intr --account=TORABI-SL3-GPU --partition=ampere --gres=gpu:1 --cpus-per-task=32 --time=01:00:00 cambridge/pregrok_acts.sbatch
set -euo pipefail
if [ -z "${SEMATAXY_IN_SRUN:-}" ]; then export SEMATAXY_IN_SRUN=1; exec srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" bash "$0" "$@"; fi
cd "$SLURM_SUBMIT_DIR"; mkdir -p results/grokking
source "${GPU_VENV:-$HOME/rds/hpc-work/semataxy/.venv-gpu}/bin/activate" 2>/dev/null || source .venv/bin/activate
export OMP_NUM_THREADS=3
RUNS=${RUNS:-$(for d in runs/grokking/s*_f0.*/; do [ -f "$d/final.pt" ] && echo "${d%/}"; done)}
echo "$RUNS" | xargs -P 10 -I{} bash -c 'r={}; [ -f "$r/acts_log.jsonl" ] && [ "$r/acts_log.jsonl" -nt "$r/final.pt" ] && exit 0; python tools/pregrok_acts.py "$r" --n-inputs 2048 --n-shuffles 2 > "$r/acts.out" 2>&1 && cp "$r/acts_log.jsonl" "results/grokking/$(basename $r)_acts_log.jsonl" && echo "done $r" || echo "FAILED $r"'
echo "acts logs: $(ls results/grokking/*_acts_log.jsonl 2>/dev/null | wc -l)"
python tools/pregrok_analysis.py results/grokking/*_acts_log.jsonl --out results/grokking/pregrok_acts_summary.csv | tail -40
