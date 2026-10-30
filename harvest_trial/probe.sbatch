#!/usr/bin/env bash
#SBATCH --job-name=hb-probe-thinkers
#SBATCH --account=TORABI-SL4-CPU
#SBATCH --partition=desktop
#SBATCH --nodes=1
#SBATCH --gres=none
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=03:00:00
#SBATCH --output=out/%x-%j.out
# The thinking probe: 7 models through OpenRouter (probe_thinkers.py), or with ROUTE=direct 4 models
# through Google's and xAI's own APIs (probe_direct.py); 12 calls a model.
# Submit from the trial folder on the cluster:
#   sbatch --export=ALL,HB_VENV=...,POT=...[,ROUTE=direct][,MODELS=a,b] probe.sbatch
# The OpenRouter key is read inside this job only, never printed, and checked for in every file written.
set -euo pipefail
T=$SLURM_SUBMIT_DIR; V=${HB_VENV:?}; POT=${POT:?}
OUT=$T/out/probe-$SLURM_JOB_ID; mkdir -p "$OUT"
echo "host $(hostname)  job $SLURM_JOB_ID  $(date -u +%FT%TZ)"
code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 https://openrouter.ai/api/v1/models || true)
echo "openrouter.ai: HTTP $code"
if [ "$code" = 000 ]; then echo "no route to openrouter.ai from this node"; exit 2; fi
rc=0
if [ "${ROUTE:-openrouter}" = direct ]; then
  # Google and xAI with the project's own keys (probe_direct.py)
  GEMINI_API_KEY=$(cat ~/.config/sovsim/gemini_key) XAI_API_KEY=$(cat ~/.config/sovsim/xai_key) \
    srun --cpus-per-task=1 "$V/bin/python" -I -B "$T/probe_direct.py" \
    --mode live --templates "$T/structure_templates.json" --out "$OUT/probe.jsonl" --pot "$POT" || rc=$?
  echo "probe exit $rc  $(date -u +%FT%TZ)"
  GEMINI_API_KEY=$(cat ~/.config/sovsim/gemini_key) "$V/bin/python" -I -B "$T/key_check.py" "$OUT" GEMINI_API_KEY || rc=3
  XAI_API_KEY=$(cat ~/.config/sovsim/xai_key) "$V/bin/python" -I -B "$T/key_check.py" "$OUT" XAI_API_KEY || rc=3
  exit $rc
fi
OPENROUTER_API_KEY=$(cat ~/.config/sovsim/openrouter_key) srun --cpus-per-task=1 "$V/bin/python" -I -B "$T/probe_thinkers.py" \
  --mode live --templates "$T/structure_templates.json" --out "$OUT/probe.jsonl" --pot "$POT" ${MODELS:+--models "$MODELS"} || rc=$?
echo "probe exit $rc  $(date -u +%FT%TZ)"
OPENROUTER_API_KEY=$(cat ~/.config/sovsim/openrouter_key) "$V/bin/python" -I -B "$T/key_check.py" "$OUT" OPENROUTER_API_KEY || rc=3
exit $rc
