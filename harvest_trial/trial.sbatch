#!/usr/bin/env bash
#SBATCH --job-name=hb-muse-trial
#SBATCH --account=TORABI-SL4-CPU
#SBATCH --partition=desktop
#SBATCH --nodes=1
#SBATCH --gres=none
#SBATCH --cpus-per-task=2
#SBATCH --mem=7G
#SBATCH --time=03:00:00
#SBATCH --output=out/%x-%j.out
# The three paid trial shifts of HarvestBench on Muse Spark 1.2 (harvest_trial/README.md).
# Submit from the trial folder on the cluster:
#   sbatch --export=ALL,HB_SRC=...,HB_VENV=... trial.sbatch
# The Meta key is read inside this job only, never printed, and checked for in every file the
# job wrote before the job ends.
set -euo pipefail
T=$SLURM_SUBMIT_DIR; SRC=${HB_SRC:?}; V=${HB_VENV:?}
BUDGETS=${BUDGETS:-minimal=1.00,medium=2.50,xhigh=3.50}
OUT=$T/out/live-$SLURM_JOB_ID; mkdir -p "$OUT"
echo "host $(hostname)  job $SLURM_JOB_ID  budgets $BUDGETS  $(date -u +%FT%TZ)"
# Can this node reach Meta's API? No key is sent; 401 means yes.
code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 https://api.meta.ai/v1/models || true)
echo "api.meta.ai: HTTP $code"
if [ "$code" = 000 ]; then echo "no route to api.meta.ai from this node"; exit 2; fi
export META_BASE_URL=https://api.meta.ai/v1
# xhigh replies are not streamed and may take minutes; Inspect's default read timeout is 600 s
export INSPECT_HTTP_REQUEST_TIMEOUT=1500
rc=0
# No call is sent after 140 min, so one in flight (at most 25 min) ends before the 3 h limit.
META_API_KEY=$(cat ~/.config/sovsim/meta_key) srun --cpus-per-task=2 "$V/bin/python" -I -B "$T/run_trial.py" \
  --src "$SRC" --mode live --log-dir "$OUT" --budgets "$BUDGETS" --pot 5.00 --stop-after-min 140 || rc=$?
echo "trial exit $rc  $(date -u +%FT%TZ)"
"$V/bin/python" -I -B "$T/check_trial.py" --src "$SRC" --log-dir "$OUT" --out "$OUT/trial_summary.json" || echo "check_trial failed"
# The key must appear in no file this job wrote.
META_API_KEY=$(cat ~/.config/sovsim/meta_key) "$V/bin/python" -I -B "$T/key_check.py" "$T/out" || rc=3
exit $rc
