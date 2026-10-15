#!/usr/bin/env bash
# The interactive GPU QOS (intr) allows one submitted job per user and starts in seconds. This loop is
# the single submitter for it: it reads runs/_slurm/intr_queue.txt one line at a time (each line a full
# sbatch command, run from the repo root), submits the next line whenever no intr job of ours is queued,
# and appends what it did to runs/_slurm/intr_queue.log. Run detached on a login node:
#   nohup bash cambridge/intr_queue.sh > /dev/null 2>&1 &
# Add work with:  echo 'sbatch ... ' >> runs/_slurm/intr_queue.txt
set -uo pipefail
cd "$(dirname "$0")/.."
Q=runs/_slurm/intr_queue.txt; L=runs/_slurm/intr_queue.log; touch "$Q" "$L"
while true; do
  if ! squeue -u "$USER" -h -O qos,jobid | grep -q "^intr"; then
    line=$(grep -m1 -v '^\s*$\|^#' "$Q" || true)
    if [ -n "$line" ]; then
      out=$(bash -c "$line" 2>&1); rc=$?
      echo "$(date +%F_%T) rc=$rc :: $line :: $out" >> "$L"
      # remove the first non-empty, non-comment line
      awk -v done=0 '!done && !/^\s*$/ && !/^#/ {done=1; next} {print}' "$Q" > "$Q.tmp" && mv "$Q.tmp" "$Q"
    fi
  fi
  sleep 60
done
