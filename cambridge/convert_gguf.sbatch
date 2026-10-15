#!/usr/bin/env bash
#SBATCH --job-name=gguf-convert
#SBATCH --account=TORABI-SL3-CPU
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=38
#SBATCH --time=04:00:00
#SBATCH --output=runs/_slurm/%x-%j.out
# Convert a Hugging Face checkpoint in the local hub to GGUF Q4_K_M for llama-server.
# Needed for models Qwen never published as GGUF (the 7B base) and for the OLMo training
# checkpoints (the "join": weight organisation, forecast error and preservation on one model).
#   sbatch --export=ALL,HF_ID=Qwen/Qwen2.5-7B,OUT=qwen2.5-7b-q4_k_m.gguf cambridge/convert_gguf.sbatch
#   sbatch --export=ALL,HF_ID=allenai/OLMo-7B-0724-hf,REVISION=step5000-tokens20B,OUT=olmo-7b-step5k-q4_k_m.gguf cambridge/convert_gguf.sbatch
# Too much CPU for the login node (its watchdog kills ~6 min of CPU), so it runs as a job.
set -euo pipefail
if [ -z "${SEMATAXY_IN_SRUN:-}" ]; then
  export SEMATAXY_IN_SRUN=1
  exec srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" bash "$0" "$@"
fi
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
export HF_HOME=${HF_HOME:-$HOME/rds/hpc-work/hf} HF_HUB_OFFLINE=1
LLAMA=${LLAMA:-$HOME/rds/hpc-work/semataxy/llama.cpp}
GGUF_DIR=${GGUF_DIR:-$HOME/rds/hpc-work/semataxy/gguf}
: "${HF_ID:?set HF_ID}"; : "${OUT:?set OUT (final q4_k_m file name)}"
REVISION=${REVISION:-main}

# the quantizer is not built by default; build just that target once
if [ ! -x "$LLAMA/build/bin/llama-quantize" ]; then
  cmake --build "$LLAMA/build" --target llama-quantize -j "$SLURM_CPUS_PER_TASK"
fi
python -c "import gguf" 2>/dev/null || pip install -q "$LLAMA/gguf-py"

# resolve the snapshot directory for this id/revision from the offline hub
SNAP=$(python - <<EOF
from huggingface_hub import snapshot_download
print(snapshot_download("$HF_ID", revision="$REVISION", local_files_only=True))
EOF
)
echo "converting $SNAP"
F16="$GGUF_DIR/${OUT%.gguf}.f16.gguf"
python "$LLAMA/convert_hf_to_gguf.py" "$SNAP" --outtype f16 --outfile "$F16"
"$LLAMA/build/bin/llama-quantize" "$F16" "$GGUF_DIR/$OUT" Q4_K_M "$SLURM_CPUS_PER_TASK"
rm -f "$F16"
ls -la "$GGUF_DIR/$OUT"
