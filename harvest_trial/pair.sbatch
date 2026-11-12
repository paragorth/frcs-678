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
#   sbatch --export=ALL,HB_SRC=...,HB_VENV=...,POT=...[,PROVIDER=deepinfra|baidu][,LIMIT=N] pair.sbatch
# LIMIT=N sends only the first N requests of the seeded order (the smoke test) and skips the analysis.
# The OpenRouter key is read inside this job only, never printed, and checked for in every file written.
# HB_PYTHON=/usr/bin/python3.11 runs on the system python (the runner needs only the standard library),
# for when the venv on RDS cannot be read; the analysis then uses HarvestBench's parser only if its file reads.
set -euo pipefail
T=$SLURM_SUBMIT_DIR; SRC=${HB_SRC:-}; V=${HB_VENV:-}; POT=${POT:?}
PY=${HB_PYTHON:-${V:?}/bin/python}
OUT=$T/out/pair-$SLURM_JOB_ID; mkdir -p "$OUT"
echo "host $(hostname)  job $SLURM_JOB_ID  python $PY  $(date -u +%FT%TZ)"
timeout 30 "$PY" -I -c "import json, urllib.request, concurrent.futures; print('python ok', __import__('sys').version.split()[0])" || { echo "python did not start within 30 s"; exit 5; }
code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 https://openrouter.ai/api/v1/models || true)
echo "openrouter.ai: HTTP $code"
if [ "$code" = 000 ]; then echo "no route to openrouter.ai from this node"; exit 2; fi
rc=0
OPENROUTER_API_KEY=$(cat ~/.config/sovsim/openrouter_key) srun --cpus-per-task=1 "$PY" -I -B "$T/run_pair.py" \
  --mode live --templates "$T/structure_templates.json" --rocks "$T/rock_templates.json" \
  --out-dir "$OUT" --pot "$POT" --provider "${PROVIDER:-deepinfra}" ${LIMIT:+--limit "$LIMIT"} || rc=$?
echo "pair exit $rc  $(date -u +%FT%TZ)"
if [ -n "${LIMIT:-}" ]; then
  # The smoke test is clean only if every call answered, from the pinned host, with no reasoning tokens
  # and a JSON choice; the full run can then be queued behind it with --dependency=afterok.
  "$PY" -I - "$OUT/pair_live.jsonl" "${PROVIDER:-deepinfra}" <<'PY' || rc=4
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
host = sys.argv[2]
bad = [r for r in rows if r.get("error") or "skipped" in r or (r.get("reasoning_tokens") or 0) > 0
       or (r.get("provider") or "").lower().replace(" ", "") != host or '"choice"' not in (r.get("text") or "")]
print(f"smoke: {len(rows)} calls, {len(bad)} not clean; hosts {sorted({r.get('provider') for r in rows})}; "
      f"reasoning tokens {sum(r.get('reasoning_tokens') or 0 for r in rows)}; "
      f"spent ${sum(r.get('cost') or 0 for r in rows):.4f}")
sys.exit(1 if bad or not rows else 0)
PY
else
  if [ -n "$SRC" ] && timeout 10 head -c 16 "$SRC/harvest/contact.py" >/dev/null 2>&1; then
    "$PY" -I -B "$T/analyse_pair.py" --src "$SRC" --templates "$T/structure_templates.json" \
      --rocks "$T/rock_templates.json" --rows "$OUT/pair_live.jsonl" --out "$OUT/analysis.json" || echo "analysis failed"
  else
    echo "HarvestBench's parser is not readable now; the registered analysis is run later with --src"
  fi
fi
# The key must appear in no file this job wrote (the pattern is read from the key file, never typed).
if grep -rqFf ~/.config/sovsim/openrouter_key "$OUT"; then echo "KEY FOUND in $OUT"; rc=3; else echo "key check: clean"; fi
exit $rc
