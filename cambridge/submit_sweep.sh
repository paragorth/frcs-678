#!/usr/bin/env bash
# Submit one half-node job per (manifest, seed): the pattern that schedules on CSD3 for this account.
#   bash cambridge/submit_sweep.sh <jobname-prefix> <n_seeds> <seed_start> manifest.yaml [more manifests]
# e.g. bash cambridge/submit_sweep.sh q32b_pond100 10 1 manifests/expB/B_res-q14b_in-q32b_r4_neutral_t07_pond100.yaml
# Whole-node jobs never start for this account; 38 cores (half an icelake node) start within minutes
# when the queue is quiet. 10 h fits every 12-month game seen so far (14B ~3.5 h, 32B stranger ~3 h).
set -euo pipefail
prefix=$1; n=$2; start=$3; shift 3
for m in "$@"; do
  for s in $(seq "$start" $((start + n - 1))); do
    N_SEEDS=1 SEED_START=$s PARALLEL=1 sbatch --parsable -J "${prefix}-s${s}" \
      --cpus-per-task=38 --time=10:00:00 cambridge/govsim_cpu.sbatch "$m"
  done
done
