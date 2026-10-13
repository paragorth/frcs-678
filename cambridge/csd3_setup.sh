#!/usr/bin/env bash
# One-time setup on a CSD3 LOGIN node (login nodes have internet; compute nodes do not).
# Account: TORABI-SL3-CPU. Run from your home or RDS directory:
#   git clone --recurse-submodules -b claude/semataxy-govsim https://github.com/paragorth/frcs-678
#   cd frcs-678 && bash cambridge/csd3_setup.sh
set -euo pipefail

# --- storage: keep models and the venv on RDS, not in the 50 GB home quota ------------
export RDS=${RDS:-$HOME/rds/hpc-work}
mkdir -p "$RDS/semataxy" "$RDS/hf"
export HF_HOME="$RDS/hf"
grep -q "HF_HOME=" ~/.bashrc || echo "export HF_HOME=$RDS/hf" >> ~/.bashrc

# --- python env (uv is a single static binary; no module needed) -----------------------
module purge; module load rhel8/default-icl 2>/dev/null || module load rhel8/default-amp 2>/dev/null || true
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv "$RDS/semataxy/.venv" --python 3.11
source "$RDS/semataxy/.venv/bin/activate"
uv pip install --index-url https://download.pytorch.org/whl/cpu torch
uv pip install -r requirements.txt fastparquet huggingface_hub peft safetensors
ln -sfn "$RDS/semataxy/.venv" .venv

# --- llama.cpp: OpenAI-compatible CPU inference server ---------------------------------
module load cmake gcc 2>/dev/null || true
if [ ! -x "$RDS/semataxy/llama.cpp/build/bin/llama-server" ]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp "$RDS/semataxy/llama.cpp"
  cmake -S "$RDS/semataxy/llama.cpp" -B "$RDS/semataxy/llama.cpp/build" -DGGML_NATIVE=ON -DLLAMA_CURL=OFF
  cmake --build "$RDS/semataxy/llama.cpp/build" --config Release -j 8 --target llama-server
fi
grep -q "LLAMA_SERVER=" ~/.bashrc || echo "export LLAMA_SERVER=$RDS/semataxy/llama.cpp/build/bin/llama-server" >> ~/.bashrc

# --- models: GGUF Q4_K_M for inference (llama.cpp), safetensors for Sigma_struct ---------
# Download on the login node; compute nodes read them from RDS.
python - <<'EOF'
from huggingface_hub import hf_hub_download, snapshot_download
import os
rds = os.environ["RDS"]
gguf = {  # official Qwen GGUF repos, 4-bit
    "q1.5b": ("Qwen/Qwen2.5-1.5B-Instruct-GGUF", "qwen2.5-1.5b-instruct-q4_k_m.gguf"),
    "q3b":   ("Qwen/Qwen2.5-3B-Instruct-GGUF",   "qwen2.5-3b-instruct-q4_k_m.gguf"),
    "q7b":   ("Qwen/Qwen2.5-7B-Instruct-GGUF",   "qwen2.5-7b-instruct-q4_k_m.gguf"),
    "q14b":  ("Qwen/Qwen2.5-14B-Instruct-GGUF",  "qwen2.5-14b-instruct-q4_k_m.gguf"),
}
os.makedirs(f"{rds}/semataxy/gguf", exist_ok=True)
for label, (repo, fn) in gguf.items():
    p = hf_hub_download(repo, fn, local_dir=f"{rds}/semataxy/gguf")
    print(label, p)
# safetensors for the structural ladder (streamed per tensor; 72B is ~145 GB on disk)
for m in ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct",
          "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-14B-Instruct"]:
    print(m, snapshot_download(m, allow_patterns=["*.safetensors", "*.json"]))
# 32B/72B: run the same two lines for them once the above works (~65 GB and ~145 GB).
EOF
echo "setup done. Next: sbatch cambridge/expC_structure.sbatch   (structural ladder, cheapest falsifier)"
