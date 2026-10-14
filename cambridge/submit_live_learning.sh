#!/usr/bin/env bash
# Live-learning cluster track (CLUSTER_AGENT_PROMPT section 6b), one half-node CPU job each.
#   bash cambridge/submit_live_learning.sh cycles     # 5 learn-and-consolidate cycles, three model sizes
#   bash cambridge/submit_live_learning.sh replay     # replay-fraction sweep: does a stronger model need less replay?
# Results land in results/live_learning/v8_<LABEL>_results.json (the sbatch copies them).
set -euo pipefail
what=${1:-cycles}
declare -A MODELS=( [pythia160m]=EleutherAI/pythia-160m [qwen05b]=Qwen/Qwen2.5-0.5B-Instruct [qwen15b]=Qwen/Qwen2.5-1.5B-Instruct )
sub() {  # label model extra_args
  sbatch --parsable -J "ll-$1" --cpus-per-task=38 --time=12:00:00 \
    --export=ALL,MODEL="$2",LABEL="$1",EXTRA_ARGS="$3" cambridge/live_learning.sbatch
}
case "$what" in
  cycles)
    for l in pythia160m qwen05b qwen15b; do sub "${l}_cyc5" "${MODELS[$l]}" "--cycles 5"; done ;;
  replay)
    # v8 used 0.75. Parag's prediction: the stronger model keeps held-out loss with less replay.
    for l in pythia160m qwen05b qwen15b; do for r in 0.25 0.5; do
      sub "${l}_rf${r/./}" "${MODELS[$l]}" "--replay-frac $r"; done; done ;;
  *) echo "unknown: $what"; exit 1 ;;
esac
