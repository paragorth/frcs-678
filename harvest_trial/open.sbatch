#!/usr/bin/env bash
#SBATCH --job-name=hb-open
#SBATCH --account=TORABI-SL3-GPU
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --qos=intr
#SBATCH --time=01:00:00
#SBATCH --output=out/%x-%j.out
# One slice of the fine-tuning test, step 1 (run_open.py): vLLM serves one family's base model and its
# adapters on this node's GPUs, and the runner sends what has no answer yet until 9 minutes before the
# hour. The interactive QOS starts within seconds but allows one job per user, for an hour: submit slices
# one after another until the runner reports nothing left to answer (each resumes from the results file).
# Submit from the trial folder on the cluster:
#   sbatch --export=ALL,FAMILY=qwen,OUTDIR=out/open-qwen open.sbatch
#   sbatch --export=ALL,FAMILY=llama,OUTDIR=out/open-llama open.sbatch
#   sbatch --gres=gpu:4 --export=ALL,FAMILY=llama,LLAMA_WEIGHTS=bf16,OUTDIR=out/open-llama open.sbatch
# The weights must already be in the Hugging Face cache (run_open.py --fetch on the login node, with pg's
# approval): the job runs offline.
set -euo pipefail
T=$SLURM_SUBMIT_DIR; FAMILY=${FAMILY:?}; OUTDIR=${OUTDIR:?}; LLAMA_WEIGHTS=${LLAMA_WEIGHTS:-awq}
START=$(date +%s); MINUTES=${MINUTES:-60}
source "${GPU_VENV:-$HOME/rds/hpc-work/semataxy/.venv-gpu}/bin/activate"
export HF_HOME=${HF_HOME:-$HOME/rds/hpc-work/hf} HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
# compute nodes cannot JIT-compile flashinfer kernels (govsim_gpu.sbatch, 16 Oct 2026)
export VLLM_USE_FLASHINFER_SAMPLER=0 FLASHINFER_DISABLE_JIT=1
# per-job loopback address, since two jobs may share a node (govsim_gpu.sbatch)
LOOP_HOST="127.$(( (SLURM_JOB_ID / 256) % 256 )).$(( SLURM_JOB_ID % 256 )).1"; PORT=8000
cd "$T"; mkdir -p "$OUTDIR"
eval "$(python -I -B run_open.py --family "$FAMILY" --llama-weights "$LLAMA_WEIGHTS" --print-serve)"
echo "host $(hostname)  job $SLURM_JOB_ID  family $FAMILY  base $BASE  GPUs $GPUS  $(date -u +%FT%TZ)"
vllm serve "$BASE" --served-model-name base --host "$LOOP_HOST" --port $PORT \
  --tensor-parallel-size "$GPUS" --gpu-memory-utilization 0.90 --generation-config auto \
  --enable-lora --max-lora-rank "$MAX_RANK" --lora-modules $LORA_MODULES $SERVE_EXTRA \
  > "$OUTDIR/vllm-$SLURM_JOB_ID.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
for _ in $(seq 1 180); do
  curl -sf "http://$LOOP_HOST:$PORT/v1/models" >/dev/null && break
  if ! kill -0 $SERVER 2>/dev/null; then
    echo "vLLM exited:"; tail -40 "$OUTDIR/vllm-$SLURM_JOB_ID.log"; exit 2
  fi
  sleep 5
done
curl -sf "http://$LOOP_HOST:$PORT/v1/models" >/dev/null || { echo "vLLM not up after 15 minutes"; exit 2; }
echo "server up after $(( $(date +%s) - START )) s"
python -I -B run_open.py --family "$FAMILY" --llama-weights "$LLAMA_WEIGHTS" --mode live \
  --base-url "http://$LOOP_HOST:$PORT/v1" --out "$OUTDIR/results.jsonl" \
  --stop-at $(( START + MINUTES * 60 - 9 * 60 )) --grace 240 --parallel "${PARALLEL:-48}"
echo "slice done $(date -u +%FT%TZ)"
