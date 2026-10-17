#!/usr/bin/env bash
#SBATCH --job-name=grok-topk
#SBATCH --account=TORABI-SL3-CPU
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=38
#SBATCH --time=02:00:00
#SBATCH --output=runs/_slurm/%x-%j.out
# Amendment 5: top-k subspace persistence from every finished sweep run's checkpoints, then the analysis.
#   sbatch --qos=intr --account=TORABI-SL3-GPU --partition=ampere --gres=gpu:1 --cpus-per-task=32 --time=00:40:00 cambridge/pregrok_topk.sbatch
set -euo pipefail
if [ -z "${SEMATAXY_IN_SRUN:-}" ]; then export SEMATAXY_IN_SRUN=1; unset SLURM_MEM_PER_CPU SLURM_MEM_PER_NODE; exec srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" bash "$0" "$@"; fi
cd "$SLURM_SUBMIT_DIR"; mkdir -p results/grokking
source "${GPU_VENV:-$HOME/rds/hpc-work/semataxy/.venv-gpu}/bin/activate" 2>/dev/null || source .venv/bin/activate
export OMP_NUM_THREADS=3
RUNS=$(for d in runs/grokking/s*_f0.*/; do [ -f "$d/final.pt" ] && echo "${d%/}"; done)
echo "$RUNS" | xargs -P 10 -I{} bash -c 'r={}; [ -f "$r/persist.jsonl" ] && [ "$r/persist.jsonl" -nt "$r/final.pt" ] && exit 0; python tools/pregrok_topk.py persist "$r" --k 4 8 > "$r/persist.out" 2>&1 && cp "$r/persist.jsonl" "results/grokking/$(basename $r)_persist.jsonl" && echo "done $r" || echo "FAILED $r"'
echo "persist files: $(ls results/grokking/*_persist.jsonl 2>/dev/null | wc -l)"
python tools/pregrok_topk.py analyse results/grokking | tail -60
