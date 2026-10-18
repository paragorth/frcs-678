#!/bin/bash
# Depth sweep per PREREG-depth-sweep.md. Fits only; no training, no install runs.
set -u
# Single-instance lock. Earlier attempts left three sweep generations running
# concurrently on the same tag, contending for the GPU and racing on the same
# checkpoint file, which silently corrupted the fit and made wall-times
# meaningless. Refuse to start if another instance holds the lock.
LOCK=/workspace/out/.depth_sweep.lock
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "REFUSING TO START: $LOCK exists -- another sweep is running."
  ps -eo pid,etimes,args | grep -E "[f]it_lens|[d]epth_sweep" || true
  exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT
cd /root/jlens
export JLENS_OUT=/workspace/out HF_HOME=/workspace/hf
NP=${NP:-1000}
SB=${SB:-6}      # reference 7B fit meta: seed_batch=6 (script default 32 thrashes VRAM)
XC=${XC:-8}      # full-J cross-check prompts: 64 cost more than the fit itself
ML=${ML:-192}    # reference 7B fit meta: max_seq_len=192 (script default 256).
                 # This is a CORRECTNESS setting, not just speed: the control
                 # depth must be fitted exactly as the reference it reproduces.
for I in 4 6 7 10 13 15 18 21 24; do
  TAG="_d${I}"
  if [ -f "/workspace/out/gate_multi_7b${TAG}.json" ]; then
    echo "=== inject $I already done, skipping"; continue
  fi
  T0=$(date +%s)
  echo "=== FIT inject=$I  ($(date +%H:%M))"
  /usr/bin/python3 -u scripts/fit_lens.py --model 7b --inject "$I" --tag "$TAG" \
      --n-prompts "$NP" --seed-batch "$SB" --max-seq-len "$ML" --crosscheck "$XC"
  echo "=== FITSECS inject=$I $(( $(date +%s) - T0 ))"
  echo "=== GATE inject=$I ($(date +%H:%M))"
  /usr/bin/python3 -u scripts/stage2_verify.py --model 7b --inject "$I" --tag "$TAG"
  echo "=== GATE192 inject=$I ($(date +%H:%M))"
  /usr/bin/python3 -u scripts/gate_multi.py --model 7b --inject "$I" --tag "$TAG" \
     
  echo "=== DEPTHSECS inject=$I $(( $(date +%s) - T0 ))"
done
echo "DEPTH SWEEP COMPLETE"
