#!/usr/bin/env bash
#SBATCH --job-name=govsim-cpu
#SBATCH --account=TORABI-SL3-CPU
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=76
#SBATCH --time=12:00:00
#SBATCH --output=runs/_slurm/%x-%j.out
# GovSim on CPU: one llama.cpp server per model on this node, simulations against localhost.
#   sbatch cambridge/govsim_cpu.sbatch manifests/expA/A_q3b_h12_selffw.yaml [more manifests]
# Models served are those named in the manifests' `models:` blocks that exist in $GGUF_DIR.
set -euo pipefail
# CSD3 batch steps can see only 1 core; re-run the body as an srun step that gets the full allocation.
if [ -z "${SEMATAXY_IN_SRUN:-}" ]; then
  export SEMATAXY_IN_SRUN=1
  exec srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" bash "$0" "$@"
fi
echo "cores visible: $(nproc)"
cd "$SLURM_SUBMIT_DIR"; mkdir -p runs/_slurm runs/_servers
source .venv/bin/activate
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export GGUF_DIR=${GGUF_DIR:-$HOME/rds/hpc-work/semataxy/gguf}
export LLAMA_SERVER=${LLAMA_SERVER:-$HOME/rds/hpc-work/semataxy/llama.cpp/build/bin/llama-server}
export VLLM_HOST=http://127.0.0.1
export SEMATAXY_PREFILL=0            # llama-server cannot continue a partial assistant turn
N_SEEDS=${N_SEEDS:-5}; PARALLEL=${PARALLEL:-4}

# label -> gguf file, port (ports match manifests/models_qwen25.yaml)
# Larger models are sharded on Hugging Face; llama-server loads all shards when given the first,
# so resolve each label to its first (or only) q4_k_m file by glob.
gguf_for() { ls "$GGUF_DIR"/qwen2.5-${1#q}-instruct-q4_k_m*.gguf 2>/dev/null | sort | head -1; }
declare -A PORT=( [q0.5b]=8000 [q1.5b]=8001 [q3b]=8002 [q7b]=8003 [q14b]=8004 [q32b]=8005 [q72b]=8006 )

needed=$(grep -ho "^  q[0-9.]*b:" "$@" | tr -d ' :' | sort -u)
n=$(echo "$needed" | wc -w); threads=$(( SLURM_CPUS_PER_TASK / n ))
for label in $needed; do
  f=$(gguf_for "$label"); [ -n "$f" ] || { echo "no GGUF for $label in $GGUF_DIR"; exit 1; }
  echo "serving $label on :${PORT[$label]} with $threads threads from $(basename "$f")"
  nohup "$LLAMA_SERVER" -m "$f" --port "${PORT[$label]}" --host 127.0.0.1 \
      -t "$threads" -c 8192 --parallel "$PARALLEL" --temp 0 --seed 0 \
      > "runs/_servers/${label}-${SLURM_JOB_ID}.log" 2>&1 &
done
for label in $needed; do
  until curl -sf "http://127.0.0.1:${PORT[$label]}/v1/models" >/dev/null; do sleep 5; done; echo "$label ready"
done

python -m semataxy sweep "$@" --n-seeds "$N_SEEDS" --seed-start "${SEED_START:-1}" --out runs > "runs/_slurm/cmds-$SLURM_JOB_ID.txt"
echo "$(wc -l < runs/_slurm/cmds-$SLURM_JOB_ID.txt) runs queued"
xargs -P "$PARALLEL" -I{} bash -c "{}" < "runs/_slurm/cmds-$SLURM_JOB_ID.txt"
# only real run directories (runs/ also holds _cache, _servers, _slurm)
python -m semataxy analyze $(ls -d runs/*/ | grep -v "/_") --table --csv "runs/summary-$SLURM_JOB_ID.csv"
