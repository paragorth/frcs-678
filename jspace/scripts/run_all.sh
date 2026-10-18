#!/usr/bin/env bash
# Full pipeline, run on the pod. Results land in $JLENS_OUT (network volume).
set -uo pipefail
MODEL="${MODEL:-1.5b}"
NP="${NP:-500}"
export JLENS_OUT="${JLENS_OUT:-/workspace/out}"
export PYTHONPATH="$(pwd)/scripts:${PYTHONPATH:-}"
mkdir -p "$JLENS_OUT"
LOG="$JLENS_OUT/run_all.log"
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
step() { say "=== $* ==="; }

verdict() { python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['verdict'])" "$1" 2>/dev/null || echo MISSING; }

step "environment"
python3 -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),
torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" 2>&1 | tee -a "$LOG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | tee -a "$LOG"

step "token set"
python3 scripts/tokens.py "$MODEL" 2>&1 | tee -a "$LOG"

step "stage 1: restricted fit ($NP prompts)"
python3 scripts/fit_lens.py --model "$MODEL" --n-prompts "$NP" --crosscheck 3 2>&1 | tee -a "$LOG"

step "stage 2 attempt 1"
python3 scripts/stage2_verify.py --model "$MODEL" --attempt 1 2>&1 | tee -a "$LOG"
V1=$(verdict "$JLENS_OUT/stage2_verdict_attempt1.json")
say "stage 2 attempt 1 verdict: $V1"

TAG=""
if [ "$V1" != "PASS" ]; then
  # Pre-committed fallback: one refit at 3x prompts, one retry. Not a
  # conclusion about the model either way.
  step "stage 2 failed -> refit at 3x ($((NP*3)) prompts)"
  python3 scripts/fit_lens.py --model "$MODEL" --n-prompts "$((NP*3))" --tag _3x 2>&1 | tee -a "$LOG"
  step "stage 2 attempt 2 (3x lens)"
  python3 scripts/stage2_verify.py --model "$MODEL" --tag _3x --attempt 2 2>&1 | tee -a "$LOG"
  V2=$(verdict "$JLENS_OUT/stage2_verdict_attempt2.json")
  say "stage 2 attempt 2 verdict: $V2"
  TAG="_3x"
  if [ "$V2" != "PASS" ]; then
    say "!!! STAGE 2 FAILED AFTER REFIT (attempt1=$V1 attempt2=$V2)"
    say "!!! Stage 3 will still be run for completeness, but its output is NOT"
    say "!!! interpretable as ignition: the lens has not been shown to carry"
    say "!!! the multi-fact structure the sweep assumes."
    echo "{\"stage2_failed\":true,\"attempt1\":\"$V1\",\"attempt2\":\"$V2\"}" \
      > "$JLENS_OUT/STAGE2_FAILED.json"
  fi
fi

step "stage 3: alpha sweep (lens tag='${TAG:-<base>}')"
python3 scripts/stage3_sweep.py --model "$MODEL" --tag "$TAG" 2>&1 | tee -a "$LOG"

step "done"
ls -la "$JLENS_OUT" | tee -a "$LOG"
