#!/usr/bin/env bash
#SBATCH --job-name=govsim-gpu
#SBATCH --account=TORABI-SL3-GPU
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --output=runs/_slurm/%x-%j.out
# GovSim on GPU: vLLM serves every model the manifests name, on the GPUs of this one node; the
# simulations run as CPU processes on the same node against localhost. Same manifests, same runner,
# same cache as the CPU path. One A100-80GB holds 14B or 32B in bf16; 72B needs --gres=gpu:2 (TP=2).
#   sbatch cambridge/govsim_gpu.sbatch manifests/expB/B_res-q14b_in-q32b_r4_neutral_t07.yaml
#   sbatch --gres=gpu:2 --export=ALL,TP_q72b=2 cambridge/govsim_gpu.sbatch manifests/expB/B_res-q14b_in-q72b_*.yaml
# Env: N_SEEDS (default 10), SEED_START (1), PARALLEL (simultaneous games; vLLM batches them, default 8),
#      TP_<label> tensor-parallel size per label (default 1), GPU_UTIL (0.90).
# The interactive QOS (--qos=intr, 1 h, up to 3 nodes) schedules in seconds while the normal GPU QOS waits
# a day; run cells as repeated 1 h slices: every finished call is in the shared cache, so a re-run of the
# same manifest and seeds replays what was done and finishes the rest.
#   sbatch --qos=intr --time=01:00:00 --gres=gpu:3 --export=ALL,TP_q72b=2,PARALLEL=10 cambridge/govsim_gpu.sbatch manifests/expB/B_res-q14b_in-q72b_r4_neutral_t07.yaml
#   sbatch --qos=intr --time=01:00:00 --gres=gpu:1 --export=ALL,SHARE_GPU=1,PARALLEL=10 cambridge/govsim_gpu.sbatch manifests/expB/B_res-q14b_in-q7bb_r4_neutral_t07.yaml
# The intr QOS allows one submitted job per user at a time.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p runs/_slurm runs/_servers
source "${GPU_VENV:-$HOME/rds/hpc-work/semataxy/.venv-gpu}/bin/activate"
export HF_HOME=${HF_HOME:-$HOME/rds/hpc-work/hf} HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
# per-job loopback address (two jobs may share a node; see govsim_cpu.sbatch)
LOOP_HOST="127.$(( (SLURM_JOB_ID / 256) % 256 )).$(( SLURM_JOB_ID % 256 )).1"
export VLLM_HOST="http://$LOOP_HOST"
# Same prompting as the CPU path: no continuation of a partial assistant turn. With prefill on, vLLM
# returned empty text for every continued call (16 Oct 2026: a whole cell of zeros); and the ladder
# must be prompted identically on both paths to be comparable.
export SEMATAXY_PREFILL=0
# Compute nodes cannot JIT-compile kernels (the 72B slice died in a flashinfer ninja build, 16 Oct 2026):
# use the prebuilt sampler and attention kernels only.
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN FLASHINFER_DISABLE_JIT=1
N_SEEDS=${N_SEEDS:-10}; PARALLEL=${PARALLEL:-8}; GPU_UTIL=${GPU_UTIL:-0.90}
# LIVE=1: in-game live learning (semataxy/livegame.py). The serving engine is started with LoRA
# enabled and runtime adapter loading; a consolidation worker (tools/lora_worker.py) holds the
# stranger's base model on the LAST GPU of the allocation. One game at a time per worker.
if [ "${LIVE:-0}" = "1" ]; then
  export VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 SEMATAXY_LIVE_QUEUE="$PWD/runs_gpu/_live_queue"
  LORA_FLAGS="--enable-lora --max-lora-rank ${LORA_RANK:-16} --max-loras 4"
  PARALLEL=${LIVE_PARALLEL:-1}
  rm -rf "$SEMATAXY_LIVE_QUEUE"; mkdir -p "$SEMATAXY_LIVE_QUEUE"
else
  LORA_FLAGS=""
fi

declare -A HFID=( [q0.5b]=Qwen/Qwen2.5-0.5B-Instruct [q1.5b]=Qwen/Qwen2.5-1.5B-Instruct [q3b]=Qwen/Qwen2.5-3B-Instruct
                  [q7b]=Qwen/Qwen2.5-7B-Instruct [q14b]=Qwen/Qwen2.5-14B-Instruct [q32b]=Qwen/Qwen2.5-32B-Instruct
                  [q72b]=Qwen/Qwen2.5-72B-Instruct [q7bb]=Qwen/Qwen2.5-7B [q14bb]=Qwen/Qwen2.5-14B [q32bb]=Qwen/Qwen2.5-32B
                  [l8b]=meta-llama/Llama-3.1-8B-Instruct [g9b]=google/gemma-2-9b-it
                  [r1_70b]=deepseek-ai/DeepSeek-R1-Distill-Llama-70B [qwen3_32b]=Qwen/Qwen3-32B )
# WEIGHTS: where a label's weights differ from the name the manifests use (4-bit AWQ builds for the
# 70B-class models: a 145 GB bf16 72B needs three A100s on one node, which the queue rarely frees; the
# AWQ build fits one. The CPU rungs ran 4-bit GGUF, so the ladder stays 4-bit throughout; recorded in
# the server log and the job output).
declare -A WEIGHTS=( [q72b]=Qwen/Qwen2.5-72B-Instruct-AWQ [r1_70b]=casperhansen/deepseek-r1-distill-llama-70b-awq )
declare -A PORT=( [q0.5b]=8000 [q1.5b]=8001 [q3b]=8002 [q7b]=8003 [q14b]=8004 [q32b]=8005 [q72b]=8006
                  [l8b]=8010 [g9b]=8011 [q7bb]=8012 [q14bb]=8013 [q32bb]=8014 [r1_70b]=8015 [qwen3_32b]=8016 )
for m in "$@"; do [ -f "$m" ] || { echo "manifest not found on the cluster: $m (sync before submitting)"; exit 1; }; done
needed=$(grep -ho "^  [a-z][a-z0-9._]*:$" "$@" | tr -d ' :' | sort -u)
ngpu=$(nvidia-smi -L | wc -l); gpu=0; nlab=$(echo "$needed" | wc -w)
if [ "${LIVE:-0}" = "1" ]; then
  # reserve the last GPU for the worker; the servers use the rest
  WORKER_GPU=$((ngpu - 1)); ngpu=$WORKER_GPU
  LIVE_MODEL=${LIVE_MODEL:?set LIVE_MODEL, the HF id of the live slot model}
  CUDA_VISIBLE_DEVICES=$WORKER_GPU nohup python tools/lora_worker.py --model "$LIVE_MODEL" --queue "$SEMATAXY_LIVE_QUEUE" \
      --device cuda:0 --rank "${LORA_RANK:-16}" > "runs/_servers/lora_worker-${SLURM_JOB_ID}.log" 2>&1 &
fi
for label in $needed; do  # weights must already be in the offline hub (compute nodes have no internet)
  id=${WEIGHTS[$label]:-${HFID[$label]:?unknown label $label}}; snap="$HF_HOME/hub/models--${id//\//--}/snapshots"
  ls "$snap"/*/*.safetensors >/dev/null 2>&1 || { echo "weights for $label ($id) not in the hub yet"; exit 1; }
done
# SHARE_GPU=1: every model on GPU 0, memory split evenly (7B + 14B in bf16 fit one A100-80GB with room
# for KV cache). A one-GPU job schedules far sooner than a three-GPU one.
if [ "${SHARE_GPU:-0}" = "1" ]; then GPU_UTIL=$(python -c "print(round(0.88/$nlab,2))"); fi
wait_ready() {  # $1 label
  for i in $(seq 1 180); do curl -sf "http://$LOOP_HOST:${PORT[$1]}/v1/models" >/dev/null && return 0; sleep 10
    grep -q "Engine core initialization failed\|RuntimeError\|ValueError" "runs/_servers/$1-${SLURM_JOB_ID}.log" 2>/dev/null && return 1; done
  return 1
}
for label in $needed; do
  tpvar="TP_$label"; tp=${!tpvar:-1}
  if [ "${SHARE_GPU:-0}" = "1" ]; then devs=0; tp=1; else
    devs=$(seq -s, $gpu $((gpu + tp - 1))); gpu=$((gpu + tp))
    [ "$gpu" -le "$ngpu" ] || { echo "not enough GPUs for $needed (have $ngpu)"; exit 1; }
  fi
  w=${WEIGHTS[$label]:-${HFID[$label]}}
  echo "serving $label as ${HFID[$label]} from weights $w on GPU $devs tp=$tp port ${PORT[$label]} util $GPU_UTIL"
  CUDA_VISIBLE_DEVICES=$devs nohup vllm serve "$w" \
      --served-model-name "${HFID[$label]}" --host "$LOOP_HOST" --port "${PORT[$label]}" \
      --tensor-parallel-size "$tp" --dtype bfloat16 --max-model-len 8192 --seed 0 $LORA_FLAGS \
      --gpu-memory-utilization "$GPU_UTIL" \
      > "runs/_servers/${label}-${SLURM_JOB_ID}.log" 2>&1 &
  # Sharing one GPU: vLLM sizes its cache from the memory free at start-up, so two engines starting
  # together each see the other's allocation and both abort. Bring them up one at a time.
  if [ "${SHARE_GPU:-0}" = "1" ]; then
    wait_ready "$label" || { echo "server $label never came up"; grep -iE "error" "runs/_servers/${label}-${SLURM_JOB_ID}.log" | tail -5; exit 1; }
    echo "$label ready"
  fi
done
if [ "${SHARE_GPU:-0}" != "1" ]; then
  for label in $needed; do
    wait_ready "$label" || { echo "server $label never came up"; grep -iE "error" "runs/_servers/${label}-${SLURM_JOB_ID}.log" | tail -5; exit 1; }
    echo "$label ready"
  done
fi
# SMOKE=1: two seeds, three months, into runs_smoke/, then the validity check. Every new model, server
# or world runs this before a full cell (three GPU cells were lost to first-use faults on 16 Oct 2026).
# GPU-served games live in runs_gpu/ (own cache): the same manifest served by vLLM bf16/AWQ and by
# llama.cpp Q4 gives different resident trajectories, so the two stacks are never mixed in one cell.
# SEMATAXY_RUNS_TAG=voice -> runs_gpu_voice/ with its own cache: games after the own-voice fix (17 Oct 2026)
# are a different game from the ones before it and must never share a directory or a cache with them.
OUT=runs_gpu${SEMATAXY_RUNS_TAG:+_$SEMATAXY_RUNS_TAG}; EXTRA=""; mkdir -p "$OUT"
if [ "${SMOKE:-0}" = "1" ]; then N_SEEDS=2; OUT=runs_smoke; EXTRA="-o experiment.env.max_num_rounds=3"; mkdir -p runs_smoke; fi
if [ "${LIVE:-0}" = "1" ]; then
  for i in $(seq 1 240); do [ -f "$SEMATAXY_LIVE_QUEUE/READY" ] && break; sleep 10; done
  [ -f "$SEMATAXY_LIVE_QUEUE/READY" ] || { echo "lora worker never came up"; tail -20 "runs/_servers/lora_worker-${SLURM_JOB_ID}.log"; exit 1; }
  echo "lora worker ready: $(cat "$SEMATAXY_LIVE_QUEUE/READY")"
fi
python -m semataxy sweep "$@" --n-seeds "$N_SEEDS" --seed-start "${SEED_START:-1}" --out "$OUT" | sed "s|\$| $EXTRA|" > "runs/_slurm/cmds-$SLURM_JOB_ID.txt"
echo "$(wc -l < runs/_slurm/cmds-$SLURM_JOB_ID.txt) runs queued, $PARALLEL in flight"
xargs -P "$PARALLEL" -I{} bash -c "{}" < "runs/_slurm/cmds-$SLURM_JOB_ID.txt"
if [ "${SMOKE:-0}" = "1" ]; then
  python tools/check_cell.py $(ls -dt runs_smoke/*/ | head -n "$N_SEEDS") --job "$SLURM_JOB_ID"; exit $?
fi
python tools/check_cell.py $(ls -dt "$OUT"/*_s*_*/ | head -n "$N_SEEDS") --job "$SLURM_JOB_ID" || echo "VALIDITY CHECK FAILED: do not read this cell"
python -m semataxy analyze $(ls -d "$OUT"/*/ | grep -v "/_" | while read d; do [ -f "$d/run.json" ] && echo "$d"; done) --table --csv "$OUT/summary-$SLURM_JOB_ID.csv" || echo "analyze step failed (runs are intact)"
