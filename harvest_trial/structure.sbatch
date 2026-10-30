#!/usr/bin/env bash
#SBATCH --job-name=hb-muse-structure
#SBATCH --account=TORABI-SL4-CPU
#SBATCH --partition=desktop
#SBATCH --nodes=1
#SBATCH --gres=none
#SBATCH --cpus-per-task=2
#SBATCH --mem=7G
#SBATCH --time=04:00:00
#SBATCH --output=out/%x-%j.out
# The structure test (docs/DESIGN_harvestbench_muse.md): Muse Spark 1.2 re-asked 150 logged hay-bale
# prompts with the object's name changed (14 labels), at minimal and xhigh, then the registered analysis.
# Submit from the trial folder on the cluster:
#   sbatch --export=ALL,HB_SRC=...,HB_VENV=... structure.sbatch
# The Meta key is read inside this job only, never printed, and checked for in every file written.
set -euo pipefail
T=$SLURM_SUBMIT_DIR; SRC=${HB_SRC:?}; V=${HB_VENV:?}
OUT=$T/out/structure-$SLURM_JOB_ID; mkdir -p "$OUT"
echo "host $(hostname)  job $SLURM_JOB_ID  $(date -u +%FT%TZ)"
# Can this node reach Meta's API? No key is sent; 401 means yes.
code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 https://api.meta.ai/v1/models || true)
echo "api.meta.ai: HTTP $code"
if [ "$code" = 000 ]; then echo "no route to api.meta.ai from this node"; exit 2; fi
export META_BASE_URL=https://api.meta.ai/v1
export INSPECT_HTTP_REQUEST_TIMEOUT=1500
rc=0
META_API_KEY=$(cat ~/.config/sovsim/meta_key) srun --cpus-per-task=2 "$V/bin/python" -I -B "$T/run_structure.py" \
  --src "$SRC" --mode live --templates "$T/structure_templates.json" --log-dir "$OUT" || rc=$?
echo "structure exit $rc  $(date -u +%FT%TZ)"
"$V/bin/python" -I -B "$T/analyse_structure.py" --src "$SRC" --templates "$T/structure_templates.json" \
  --log-dir "$OUT" --out "$OUT/analysis.json" || echo "analysis failed"
# The key must appear in no file this job wrote.
META_API_KEY=$(cat ~/.config/sovsim/meta_key) "$V/bin/python" -I -B "$T/key_check.py" "$T/out" || rc=3
exit $rc
