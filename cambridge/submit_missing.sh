#!/usr/bin/env bash
# Idempotent submission of the 15 Oct 2026 programme: submits only (job name, seed) pairs that are
# neither queued/running nor already finished with a run directory. Safe to re-run; run it detached
# (nohup) so a dropped login link cannot cut it off:
#   nohup bash cambridge/submit_missing.sh > runs/_slurm/submit_missing.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
queued=$(squeue -u "$USER" -h -o %j)
have() { grep -qx "$1" <<<"$queued"; }
done_run() { ls -d "runs/${1}_s${2}_"* >/dev/null 2>&1 && grep -q '"status": "completed"' "runs/${1}_s${2}_"*/run.json 2>/dev/null; }
game() {  # prefix manifest seed_from seed_to
  local prefix=$1 m=$2 a=$3 b=$4 name; name=$(grep -m1 "^name:" "$m" | awk '{print $2}')
  for s in $(seq "$a" "$b"); do
    if have "${prefix}-s${s}" || done_run "$name" "$s"; then continue; fi
    N_SEEDS=1 SEED_START=$s PARALLEL=1 sbatch --parsable -J "${prefix}-s${s}" --cpus-per-task=38 --time=10:00:00 \
      cambridge/govsim_cpu.sbatch "$m" && echo "submitted ${prefix}-s${s}"
  done
}
game q7b_r2_t07  manifests/expB/B_res-q14b_in-q7b_r2_neutral_t07.yaml 3 3
game q14b_r2_t07 manifests/expB/B_res-q14b_in-q14b_r2_neutral_t07.yaml 2 2
for m in q7b q14b q32b; do game ${m}_r4_hid manifests/expB/B_res-q14b_in-${m}_r4_neutral_t07_hidden.yaml 1 10; done
for s in q14b q32b; do for c in 20 100 300; do game ${s}_pond${c} manifests/expB/B_res-q14b_in-${s}_r4_neutral_t07_pond${c}.yaml 1 10; done; done
for w in sheep pollution; do game A_q14b_${w} manifests/expA/A_q14b_${w}_h12_selffw.yaml 1 10; done
for s in q14b q32b; do game ${s}_succ8 manifests/expB/B_res-q14b_in-${s}_r4_succ8_neutral_t07.yaml 1 10; done
# live learning and the base-model conversion
declare -A MODELS=( [pythia160m]=EleutherAI/pythia-160m [qwen05b]=Qwen/Qwen2.5-0.5B-Instruct [qwen15b]=Qwen/Qwen2.5-1.5B-Instruct )
ll() { local label=$1 extra=$2; have "ll-$label" && return; [ -f "results/live_learning/v8_${label}_results.json" ] && return
  sbatch --parsable -J "ll-$label" --cpus-per-task=38 --time=12:00:00 --export=ALL,MODEL="${MODELS[${label%%_*}]}",LABEL="$label",EXTRA_ARGS="$extra" \
    cambridge/live_learning.sbatch && echo "submitted ll-$label"; }
for l in pythia160m qwen05b qwen15b; do ll "${l}_cyc5" "--cycles 5"; ll "${l}_rf025" "--replay-frac 0.25"; ll "${l}_rf05" "--replay-frac 0.5"; done
for l in pythia160m qwen05b qwen15b; do ll "${l}_ext" "--gated --t13 --implications model"; ll "${l}_cyc5_decay" "--cycles 5 --decay-tau 2"; done
if ! have gguf-convert && [ ! -f "$HOME/rds/hpc-work/semataxy/gguf/qwen2.5-7b-q4_k_m.gguf" ]; then
  sbatch --parsable --export=ALL,HF_ID=Qwen/Qwen2.5-7B,OUT=qwen2.5-7b-q4_k_m.gguf cambridge/convert_gguf.sbatch && echo "submitted gguf-convert"
fi
echo "--- queue $(date +%H:%M)"; squeue -u "$USER" -h -o "%j %T" | sed -E "s/-s[0-9]+ / /" | sort | uniq -c; echo "total: $(squeue -u "$USER" -h | wc -l)"
