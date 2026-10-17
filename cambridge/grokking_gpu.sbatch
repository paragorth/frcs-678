#!/usr/bin/env bash
#SBATCH --job-name=grok-gpu
#SBATCH --account=TORABI-SL3-GPU
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --output=runs/_slurm/%x-%j.out
# The whole pre-grokking sweep on one A100 in one slice: 21 small runs in parallel (the model is one layer,
# 128-wide); structure logged every 100 epochs on the CPU side of each process.
#   sbatch --qos=intr cambridge/grokking_gpu.sbatch
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p runs/_slurm results/grokking runs/grokking
source "${GPU_VENV:-$HOME/rds/hpc-work/semataxy/.venv-gpu}/bin/activate"
export OMP_NUM_THREADS=1 EPOCHS=${EPOCHS:-80000}
python tools/grokking.py --seed 0 --wd 1.0 --frac 0.3 --epochs 300 --every 100 --save-every 500 --n-shuffles 1 --device cuda --out runs/grokking/_selfcheck > runs/grokking/_selfcheck.log 2>&1 \
  || { echo "SELFCHECK FAILED"; tail -5 runs/grokking/_selfcheck.log; exit 1; }
python -c "import torch; assert torch.cuda.is_available(), 'no cuda'"; echo "selfcheck ok on $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
run() {  # seed wd frac
  local out=runs/grokking/s$1_wd$2_f$3
  [ -f "$out/final.pt" ] && { echo "skip $out (done)"; return; }
  local resume=""; local last=$(ls "$out"/ckpt/ep*.pt 2>/dev/null | grep -v '\.opt$' | sort | tail -1)
  if [ -n "$last" ]; then ep=$(basename "$last" .pt); ep=${ep#ep}; resume="--resume $last --start-epoch $((10#$ep))"; echo "resume $out from epoch $((10#$ep))"; fi
  python tools/grokking.py --seed "$1" --wd "$2" --frac "$3" --epochs "$EPOCHS" --every 100 --save-every 500 --n-shuffles 4 --device cuda --out "$out" $resume >> "$out.log" 2>&1 \
    && cp "$out/log.jsonl" "results/grokking/s$1_wd$2_f$3_log.jsonl" && echo "done $out" || echo "FAILED $out (see $out.log)"
}
for s in 0 1 2 3 4; do for wd in 0.3 1.0 3.0; do run $s $wd 0.3 & done; done
for s in 0 1 2; do run $s 0.0 0.3 & run $s 1.0 0.6 & done
wait
echo "finished: $(ls runs/grokking/*/final.pt 2>/dev/null | wc -l) runs"
