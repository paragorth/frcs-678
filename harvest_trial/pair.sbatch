#!/usr/bin/env bash
#SBATCH --job-name=hb-pair-deepseek
#SBATCH --account=TORABI-SL4-CPU
#SBATCH --partition=desktop
#SBATCH --nodes=1
#SBATCH --gres=none
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=03:00:00
#SBATCH --output=out/%x-%j.out
# The pair test (docs/DESIGN_harvestbench_muse.md): DeepSeek V4 Pro and V4 Flash through OpenRouter,
# thinking off, on the structure test's 2,100 prompts and 30 rock prompts each (run_pair.py), then the
# registered analysis (analyse_pair.py). Submit from the trial folder on the cluster:
#   sbatch --export=ALL,HB_SRC=...,HB_VENV=...,POT=...[,PROVIDER=deepinfra|baidu] pair.sbatch
# The OpenRouter key is read inside this job only, never printed, and checked for in every file written.
set -euo pipefail
T=$SLURM_SUBMIT_DIR; SRC=${HB_SRC:?}; V=${HB_VENV:?}; POT=${POT:?}
OUT=$T/out/pair-$SLURM_JOB_ID; mkdir -p "$OUT"
echo "host $(hostname)  job $SLURM_JOB_ID  $(date -u +%FT%TZ)"
code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 https://openrouter.ai/api/v1/models || true)
echo "openrouter.ai: HTTP $code"
if [ "$code" = 000 ]; then echo "no route to openrouter.ai from this node"; exit 2; fi
rc=0
OPENROUTER_API_KEY=$(cat ~/.config/sovsim/openrouter_key) srun --cpus-per-task=1 "$V/bin/python" -I -B "$T/run_pair.py" \
  --mode live --templates "$T/structure_templates.json" --rocks "$T/rock_templates.json" \
  --out-dir "$OUT" --pot "$POT" --provider "${PROVIDER:-deepinfra}" || rc=$?
echo "pair exit $rc  $(date -u +%FT%TZ)"
"$V/bin/python" -I -B "$T/analyse_pair.py" --src "$SRC" --templates "$T/structure_templates.json" \
  --rocks "$T/rock_templates.json" --rows "$OUT/pair_live.jsonl" --out "$OUT/analysis.json" || echo "analysis failed"
OPENROUTER_API_KEY=$(cat ~/.config/sovsim/openrouter_key) "$V/bin/python" -I -B "$T/key_check.py" "$OUT" OPENROUTER_API_KEY || rc=3
exit $rc
