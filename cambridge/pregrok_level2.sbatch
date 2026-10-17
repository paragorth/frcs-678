#!/usr/bin/env bash
#SBATCH --job-name=grok-l2
#SBATCH --account=TORABI-SL3-CPU
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=38
#SBATCH --time=06:00:00
#SBATCH --output=runs/_slurm/%x-%j.out
# Level-2 (between-matrix) organisation along the circuit, from every finished sweep run's checkpoints.
#   sbatch cambridge/pregrok_level2.sbatch            # all runs under runs/grokking with a final.pt
#   sbatch --export=ALL,RUNS="runs/grokking/s0_wd1.0_f0.3" cambridge/pregrok_level2.sbatch
set -euo pipefail
if [ -z "${SEMATAXY_IN_SRUN:-}" ]; then export SEMATAXY_IN_SRUN=1; exec srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" bash "$0" "$@"; fi
cd "$SLURM_SUBMIT_DIR"; mkdir -p results/grokking
source .venv/bin/activate; export OMP_NUM_THREADS=4
RUNS=${RUNS:-$(for d in runs/grokking/*/; do [ -f "$d/final.pt" ] && echo "${d%/}"; done)}
echo "$RUNS" | xargs -P 8 -I{} bash -c 'r={}; [ -f "$r/level2_log.jsonl" ] && [ "$r/level2_log.jsonl" -nt "$r/final.pt" ] && exit 0; python tools/pregrok_level2.py "$r" --n-shuffles 4 > "$r/level2.out" 2>&1 && cp "$r/level2_log.jsonl" "results/grokking/$(basename $r)_level2_log.jsonl" && echo "done $r"'
python tools/pregrok_analysis.py results/grokking/*_f*_log.jsonl --out results/grokking/pregrok_level1_summary.csv | tail -40
python tools/pregrok_analysis.py results/grokking/*_level2_log.jsonl --out results/grokking/pregrok_level2_summary.csv | tail -40
