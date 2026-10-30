#!/usr/bin/env bash
#SBATCH --job-name=hb-sonnet-structure
#SBATCH --account=TORABI-SL4-CPU
#SBATCH --partition=desktop
#SBATCH --nodes=1
#SBATCH --gres=none
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=11:00:00
#SBATCH --output=out/%x-%j.out
# The structure test on Sonnet 5 and Sonnet 5.5 (docs/DESIGN_harvestbench_muse.md), through Anthropic's
# batch service, then the registered analysis. Submit from the trial folder on the cluster:
#   sbatch --export=ALL,HB_SRC=...,HB_VENV=...,ANTHROPIC_VENV=...,POT=... sonnet.sbatch
# The Anthropic key is read inside this job only, never printed, and checked for in every file written.
set -euo pipefail
T=$SLURM_SUBMIT_DIR; SRC=${HB_SRC:?}; V=${HB_VENV:?}; A=${ANTHROPIC_VENV:?}; POT=${POT:?}
OUT=$T/out/sonnet-$SLURM_JOB_ID; mkdir -p "$OUT"
echo "host $(hostname)  job $SLURM_JOB_ID  $(date -u +%FT%TZ)"
# Can this node reach Anthropic's API? No key is sent; 401 means yes.
code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 https://api.anthropic.com/v1/models || true)
echo "api.anthropic.com: HTTP $code"
if [ "$code" = 000 ]; then echo "no route to api.anthropic.com from this node"; exit 2; fi
rc=0
ANTHROPIC_API_KEY=$(cat ~/.config/sovsim/anthropic_key) srun --cpus-per-task=1 "$A/bin/python" -I -B "$T/run_sonnet.py" \
  --mode live --templates "$T/structure_templates.json" --rocks "$T/rock_templates.json" --out-dir "$OUT" --pot "$POT" || rc=$?
echo "sonnet exit $rc  $(date -u +%FT%TZ)"
"$V/bin/python" -I -B "$T/analyse_sonnet.py" --src "$SRC" --templates "$T/structure_templates.json" \
  --rocks "$T/rock_templates.json" --results "$OUT/results_live.jsonl" \
  --muse "$T/out/structure-39710878/analysis.json" --out "$OUT/analysis.json" || echo "analysis failed"
# The key must appear in no file this job wrote.
ANTHROPIC_API_KEY=$(cat ~/.config/sovsim/anthropic_key) "$V/bin/python" -I -B "$T/key_check.py" "$T/out" ANTHROPIC_API_KEY || rc=3
exit $rc
