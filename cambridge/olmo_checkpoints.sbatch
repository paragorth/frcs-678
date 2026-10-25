#!/usr/bin/env bash
#SBATCH --job-name=olmo-ckpt
#SBATCH --account=TORABI-SL4-CPU
#SBATCH --partition=desktop
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=runs/_slurm/%x-%A_%a.out
# Paper 1 §5.2 in a second family (25 Oct; §7.7's pending "OLMo-7B's nine training checkpoints"): does
# the induction wiring form before copying in OLMo-7B-0724's own pretraining? The nine snapshots cached
# for Paper 1.5, heads found in the final one; weight distances streamed from the files (--stream), so
# one float32 copy (28 GB) fits a desktop node. Array 0-2 splits the snapshots over three nodes with one
# overlapping snapshot each (its movement per step comes from the part before); merge by dropping the
# first row of parts 1 and 2.
set -euo pipefail
if [ -z "${SEMATAXY_IN_SRUN:-}" ]; then
  export SEMATAXY_IN_SRUN=1
  exec srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" bash "$0" "$@"
fi
cd "$SLURM_SUBMIT_DIR"; mkdir -p runs/_slurm results/live_learning
source .venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_HOME=$HOME/rds/hpc-work/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTHONPATH="$SLURM_SUBMIT_DIR${PYTHONPATH:+:$PYTHONPATH}"
PARTS=(step0-tokens0B,step1000-tokens4B,step5000-tokens20B,step20000-tokens83B
       step20000-tokens83B,step50000-tokens209B,step100000-tokens419B,step200000-tokens838B
       step200000-tokens838B,step400000-tokens1677B,step650650-tokens2729B)
i=$SLURM_ARRAY_TASK_ID
echo "olmo checkpoints part $i on $(hostname) at $(date +%H:%M)"
python tools/pythia_checkpoints.py --model allenai/OLMo-7B-0724-hf --final step650650-tokens2729B \
    --revs ${PARTS[$i]} --stream --out results/live_learning/olmo7b_pretraining_induction_p$i.csv
echo "olmo checkpoints done at $(date +%H:%M)"
