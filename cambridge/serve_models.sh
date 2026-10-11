#!/usr/bin/env bash
# Start one vLLM OpenAI-compatible server per model on the GPUs of this node.
# Ports match manifests/models_qwen25.yaml. Edit MODELS/GPUS for the node you get.
#
# Usage (inside an allocation):   bash cambridge/serve_models.sh
# Then:  export VLLM_HOST=http://$(hostname)   and run the manifests.
#
# Nothing here has been run on the Cambridge cluster yet. Check first:
#   - which vLLM version the module system or your conda env provides
#   - whether the node has outbound access to huggingface.co (else pre-download
#     to $HF_HOME on a login node and pass --download-dir / HF_HUB_OFFLINE=1)
#   - GPU memory per model (72B needs 2+ A100-80GB or 4 A100-40GB, see TP below)
set -euo pipefail

: "${HF_HOME:=$HOME/.cache/huggingface}"
export HF_HOME
LOGDIR=${LOGDIR:-$PWD/runs/_servers}
mkdir -p "$LOGDIR"

# label | model id | port | gpu ids (comma) | tensor-parallel
read -r -d '' MODELS <<'EOF' || true
q3b   Qwen/Qwen2.5-3B-Instruct   8002 0   1
q7b   Qwen/Qwen2.5-7B-Instruct   8003 1   1
q14b  Qwen/Qwen2.5-14B-Instruct  8004 2   1
q32b  Qwen/Qwen2.5-32B-Instruct  8005 3   1
EOF

while read -r label model port gpus tp; do
  [ -z "$label" ] && continue
  echo "starting $label ($model) on GPU(s) $gpus port $port"
  CUDA_VISIBLE_DEVICES=$gpus nohup python -m vllm.entrypoints.openai.api_server \
      --model "$model" --port "$port" --tensor-parallel-size "$tp" \
      --dtype bfloat16 --max-model-len 8192 --seed 0 \
      --gpu-memory-utilization 0.90 --disable-log-requests \
      > "$LOGDIR/$label.log" 2>&1 &
  echo $! > "$LOGDIR/$label.pid"
done <<< "$MODELS"

echo "waiting for servers..."
while read -r label model port gpus tp; do
  [ -z "$label" ] && continue
  until curl -sf "http://127.0.0.1:$port/v1/models" >/dev/null; do sleep 10; done
  echo "$label ready on :$port"
done <<< "$MODELS"
