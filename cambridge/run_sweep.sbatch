#!/usr/bin/env bash
#SBATCH --job-name=semataxy
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=runs/_slurm/%x-%j.out
#
# One node serves the model ladder; the simulations run as CPU processes on the
# same node against localhost. Partition, account and GPU type are deliberately
# not set: fill them in after `sinfo` / `sacctmgr show assoc user=$USER` on CSD3
# (or whichever Cambridge system this is). Nothing here has been submitted yet.
#
#   sbatch cambridge/run_sweep.sbatch manifests/expA/*.yaml
#   sbatch cambridge/run_sweep.sbatch manifests/expB/*.yaml
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p runs/_slurm
source .venv/bin/activate

export VLLM_HOST=http://127.0.0.1
export TOKENIZERS_PARALLELISM=false
N_SEEDS=${N_SEEDS:-10}
PARALLEL=${PARALLEL:-8}   # simultaneous simulations; vLLM batches their requests

bash cambridge/serve_models.sh

python -m semataxy sweep "$@" --n-seeds "$N_SEEDS" --out runs > runs/_slurm/cmds-$SLURM_JOB_ID.txt
echo "$(wc -l < runs/_slurm/cmds-$SLURM_JOB_ID.txt) runs queued"
# xargs keeps PARALLEL simulations in flight; each writes its own run directory.
xargs -P "$PARALLEL" -I{} bash -c "{}" < runs/_slurm/cmds-$SLURM_JOB_ID.txt

python -m semataxy analyze runs/*/ --table --csv runs/summary-$SLURM_JOB_ID.csv
