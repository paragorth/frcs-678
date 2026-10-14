#!/usr/bin/env bash
#SBATCH --job-name=capbench
#SBATCH --account=TORABI-SL3-CPU
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=38
#SBATCH --time=06:00:00
#SBATCH --output=runs/_slurm/%x-%j.out
# Independent capability score (tools/capability_bench.py) for one model served by llama-server.
#   sbatch -J cap-q7b --export=ALL,LABEL=q7b cambridge/capability_bench.sbatch
# LABEL is a ladder label (q0.5b .. q72b, l8b, g9b) or NAME=<org/model> with GGUF=<file> for others.
set -euo pipefail
if [ -z "${SEMATAXY_IN_SRUN:-}" ]; then
  export SEMATAXY_IN_SRUN=1
  exec srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" bash "$0" "$@"
fi
cd "$SLURM_SUBMIT_DIR"; mkdir -p runs/_slurm runs/_servers results/capability
source .venv/bin/activate
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export GGUF_DIR=${GGUF_DIR:-$HOME/rds/hpc-work/semataxy/gguf}
export LLAMA_SERVER=${LLAMA_SERVER:-$HOME/rds/hpc-work/semataxy/llama.cpp/build/bin/llama-server}
export VLLM_HOST=http://127.0.0.1 SEMATAXY_PREFILL=0
LABEL=${LABEL:?}
declare -A NAME=( [q0.5b]=Qwen/Qwen2.5-0.5B-Instruct [q1.5b]=Qwen/Qwen2.5-1.5B-Instruct [q3b]=Qwen/Qwen2.5-3B-Instruct
                  [q7b]=Qwen/Qwen2.5-7B-Instruct [q14b]=Qwen/Qwen2.5-14B-Instruct [q32b]=Qwen/Qwen2.5-32B-Instruct
                  [q72b]=Qwen/Qwen2.5-72B-Instruct [l8b]=Llama-3.1-8B-Instruct [g9b]=Gemma-2-9B-it )
gguf_for() {
  case "$1" in
    q*)  ls "$GGUF_DIR"/qwen2.5-${1#q}-instruct-q4_k_m*.gguf 2>/dev/null | sort | head -1 ;;
    l8b) ls "$GGUF_DIR"/Meta-Llama-3.1-8B-Instruct-Q4_K_M*.gguf 2>/dev/null | head -1 ;;
    g9b) ls "$GGUF_DIR"/gemma-2-9b-it-Q4_K_M*.gguf 2>/dev/null | head -1 ;;
  esac
}
f=${GGUF:-$(gguf_for "$LABEL")}; [ -n "$f" ] || { echo "no GGUF for $LABEL"; exit 1; }
PORT=8020
echo "serving $LABEL from $(basename "$f") on :$PORT with $SLURM_CPUS_PER_TASK threads"
nohup "$LLAMA_SERVER" -m "$f" --port $PORT --host 127.0.0.1 -t "$SLURM_CPUS_PER_TASK" -c 8192 --parallel 4 --temp 0 --seed 0 \
    > "runs/_servers/cap-${LABEL}-${SLURM_JOB_ID}.log" 2>&1 &
until curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null; do sleep 5; done; echo "ready"
python tools/capability_bench.py --label "$LABEL" --model "${NAME[$LABEL]:-$LABEL}" --port $PORT --cache-dir runs/_cache_capability
