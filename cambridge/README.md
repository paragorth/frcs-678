# Running on Cambridge compute

Status: **not yet run on any cluster.** Everything below is prepared, not verified.
The pipeline has been verified end to end only against the scripted mock server
and a CPU-served 0.5B model in a container without GPUs (see the repo README).

## What I need to know before the first submission

1. **Which system** — CSD3 (Wilkes3 A100 nodes), Dawn (Intel PVC), or a departmental box?
   The sbatch header assumes Slurm + NVIDIA; Dawn would need vLLM's XPU build.
2. **Account and partition** names (`sacctmgr show assoc user=$USER`, `sinfo`).
3. **Outbound network from compute nodes.** If compute nodes cannot reach
   huggingface.co, pre-download weights on a login node:
   `huggingface-cli download Qwen/Qwen2.5-7B-Instruct` for each model,
   then `export HF_HUB_OFFLINE=1` in the job.
4. **Python env.** `uv venv .venv --python 3.11 && uv pip install -r requirements.txt vllm`
   (vLLM pins its own torch/CUDA; install it last and let it win).
5. **GPU memory.** 3B/7B/14B fit one 80 GB A100 each in bf16; 32B needs ~70 GB
   (one A100-80 at 0.9 utilisation is tight; use TP=2 to be safe); 72B needs TP=2
   on 80 GB or TP=4 on 40 GB. Adjust `serve_models.sh` to the node.

Never put credentials in manifests or logs. The runner reads `VLLM_HOST` only.

## Order of operations

1. `salloc` one GPU node, `bash cambridge/serve_models.sh`, then
   `VLLM_HOST=http://127.0.0.1 python -m semataxy run manifests/expA/A_q3b_h12_selffw.yaml --seed 1`
   — one stock-baseline run on one model. Read `runs/<id>/llm_calls.jsonl` by eye.
2. Three seeds of `A_q3b_h12_selffw`, `A_q7b_h12_selffw`, `A_q14b_h12_selffw`
   — the historical pilot ladder. Compare against the GovSim paper's qualitative
   pattern (small models collapse early; larger ones sometimes sustain).
3. The mixed-population pilot: `manifests/expB/B_res-q3b_in-q{3b,7b,14b}_r4_neutral.yaml`.
4. Only then the sweeps (`run_sweep.sbatch`), 10 seeds first, then scale.

## Cost sketch

One stock 12-round, 5-agent fishery run is roughly 1,000–1,500 LLM calls and
~1–2 M prompt tokens (prompts carry memories). On one A100 a 7B model serves
that in ~10–20 minutes when 8 runs share the server. The full ExpA/ExpB manifest
set at 10 seeds is ~400 runs; at 30 seeds ~1,200. Fine for a few node-days.
