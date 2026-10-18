#!/bin/bash
# SAFE PULL. Written after attempt 2 destroyed 8.4h of completed results.
#
# What went wrong: `rsync ... root@$IP:/path/file_d*.json out/` used an UNQUOTED
# remote glob, so the LOCAL shell tried to expand it, matched nothing, and the
# rsync failed -- and `runpod_ctl down` ran in the SAME command block without
# checking. The pod was destroyed with the only copy of the data on it.
#
# This script pulls and verifies ONLY. It never destroys. Teardown is a separate
#, deliberate step taken after the results have been READ.
set -euo pipefail
JOB="${1:?usage: pull_verify.sh <job> <expected_count>}"
EXPECT="${2:?expected number of verdict files}"
ST="out/runpod_state_${JOB}.json"
IP=$(python3 -c "import json;print(json.load(open('$ST'))['ssh_ip'])")
PT=$(python3 -c "import json;print(json.load(open('$ST'))['ssh_port'])")
RSH="ssh -i $HOME/.ssh/id_ed25519_runpod -o StrictHostKeyChecking=no -p $PT"

echo "== remote inventory (quoted glob -- REMOTE shell expands it)"
$RSH "root@$IP" 'ls -la /workspace/out/stage2_verdict_7b_d*.json' || {
  echo "FATAL: no verdicts on the pod. DO NOT DESTROY."; exit 1; }

echo "== pulling whole out/ directory (no globs at all -- nothing to mis-expand)"
rsync -az -e "$RSH" "root@$IP:/workspace/out/" out/pulled_${JOB}/

N=$(ls out/pulled_${JOB}/stage2_verdict_7b_d*.json 2>/dev/null | wc -l | tr -d ' ')
echo "== verified locally: $N verdict files (expected $EXPECT)"
if [ "$N" -lt "$EXPECT" ]; then
  echo "FATAL: pulled $N < $EXPECT. DO NOT DESTROY THE POD."; exit 1
fi
echo "PULL VERIFIED. Results are local. Read them BEFORE any teardown."
