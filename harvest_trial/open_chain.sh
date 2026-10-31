#!/usr/bin/env bash
# Runs the fine-tuning test's slices one after another (open.sbatch), family by family, until every request
# has an answer. The interactive QOS accepts one submitted job per user, so a slice cannot queue the next:
# this loop does, from the login node. It submits a slice whenever no hb-open job of ours is queued or
# running, and stops after three slices in a row that answer nothing new. Each argument is a family, or
# family:GPUs to serve it across that many GPUs, with 64 sequences and 48 requests at once per GPU.
# Start it after the smoke tests pass:
#   nohup bash open_chain.sh qwen:2 llama > out/open-chain.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")"
G=${GPU_VENV:-$HOME/rds/hpc-work/semataxy/.venv-gpu}/bin/python
say() { echo "$(date -u +%FT%TZ) $*"; }
for ARG in "$@"; do
  FAMILY=${ARG%%:*}; N=1; [ "$ARG" != "$FAMILY" ] && N=${ARG#*:}
  OUTDIR=out/open-$FAMILY; mkdir -p "$OUTDIR"; stuck=0; before=""
  OPTS=(); EXPORT="ALL,FAMILY=$FAMILY,OUTDIR=$OUTDIR"
  if [ "$N" -gt 1 ]; then
    OPTS=(--gres=gpu:$N); EXPORT="$EXPORT,TP=$N,MAX_SEQS=$((64 * N)),PARALLEL=$((48 * N))"
  fi
  while true; do
    if squeue -u "$USER" -h -n hb-open -o %i | grep -q .; then sleep 120; continue; fi
    left=$($G -I -B run_open.py --family "$FAMILY" --mode left --out "$OUTDIR/results.jsonl") || { say "cannot count $FAMILY"; exit 1; }
    if [ "$left" = 0 ]; then say "$FAMILY: every request answered"; break; fi
    if [ -n "$before" ]; then  # a slice of ours ran since the last count
      if [ "$left" = "$before" ]; then stuck=$((stuck + 1)); else stuck=0; fi
      if [ "$stuck" -ge 3 ]; then say "$FAMILY: three slices in a row answered nothing new; stopping"; exit 1; fi
    fi
    if job=$(sbatch --parsable ${OPTS[@]+"${OPTS[@]}"} --export="$EXPORT" open.sbatch 2>&1); then
      say "$FAMILY: $left left; slice $job submitted on $N GPU(s)"; before=$left
    else
      say "$FAMILY: sbatch refused ($job); trying again in 5 minutes"; before=""; sleep 180
    fi
    sleep 120
  done
done
say "chain done"
