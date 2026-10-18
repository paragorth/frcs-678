#!/usr/bin/env bash
# Job 1: fit a lens on a larger corpus, then run the Stage 2 validation gate.
# Pre-committed: on failure, refit ONCE at a larger corpus, retry, then stop.
set -u
MODEL=${MODEL:?}; NP=${NP:-3000}; NP2=${NP2:-6000}; SB=${SB:-24}; SEQ=${SEQ:-256}
export JLENS_OUT=${JLENS_OUT:-/workspace/out}
export PYTHONPATH=/root/jlens/scripts
cd /root/jlens
log(){ echo "[$(date -u +%H:%M:%S)] $*"; }

log "=== job1 $MODEL: prompts ==="
python3 - <<PY
import json, os
from jlens.examples import load_wikitext_prompts
out=os.environ["JLENS_OUT"]; p=os.path.join(out,"fit_prompts.json")
need=$NP2
cur=json.load(open(p)) if os.path.exists(p) else []
if len(cur)<need:
    cur=load_wikitext_prompts(need); json.dump(cur,open(p,"w"))
print("fit corpus:",len(cur))
PY

log "=== token set ==="
python3 scripts/tokens.py $MODEL

log "=== stage 1 fit ($NP prompts) ==="
python3 scripts/fit_lens.py --model $MODEL --n-prompts $NP --seed-batch $SB --max-seq-len $SEQ --crosscheck 2 || exit 1

log "=== stage 2 attempt 1 ==="
python3 scripts/stage2_verify.py --model $MODEL 2>&1 | tee /tmp/s2a.txt
V=$(python3 -c "import json,os;print(json.load(open(os.path.join(os.environ['JLENS_OUT'],'stage2_verdict_${MODEL}.json')))['verdict'])")
log "attempt 1 verdict: $V"
if [ -z "$V" ]; then log "=== STAGE 2 DID NOT PRODUCE A VERDICT ($MODEL) -- aborting rather than refitting on a crash ==="; exit 2; fi
if [ "$V" = "PASS" ]; then log "=== GATE PASSED ($MODEL, $NP prompts) ==="; exit 0; fi

log "=== refit at $NP2 prompts (pre-committed single retry) ==="
python3 scripts/fit_lens.py --model $MODEL --n-prompts $NP2 --seed-batch $SB --max-seq-len $SEQ --tag _big || exit 1
log "=== stage 2 attempt 2 ==="
python3 scripts/stage2_verify.py --model $MODEL --tag _big 2>&1 | tee /tmp/s2b.txt
V2=$(python3 -c "import json,os;print(json.load(open(os.path.join(os.environ['JLENS_OUT'],'stage2_verdict_${MODEL}_big.json')))['verdict'])")
log "attempt 2 verdict: $V2"
if [ "$V2" = "PASS" ]; then log "=== GATE PASSED ($MODEL, $NP2 prompts) ==="; else
  log "=== GATE FAILED ($MODEL): $V then $V2 -- stopping on this model, criterion not relaxed ==="; fi
