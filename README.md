# Semataxy simulations on GovSim

Two studies on an established commons benchmark, with real pretrained models at
several capability levels:

* **A — persistence across capability**: do more capable homogeneous populations
  sustain the commons longer?
* **B — superior newcomer**: a settled society of weaker models; a stronger one
  enters later. What happens to the residents and the commons?

Upstream [GovSim](https://github.com/giorgiopiatti/GovSim) (Piatti et al., NeurIPS 2024,
MIT) is a pinned, unmodified submodule. This repository adds a thin overlay:
one HTTP inference interface for every model, a response cache that doubles as
the branching mechanism, JSONL tracing, run manifests, analysis, and cluster
scripts. See `docs/INVENTORY.md` for the audit and `docs/DESIGN.md` for the
hypotheses and readouts.

## Status (11 Oct 2026)

| Done | Not done |
| --- | --- |
| Upstream audited; quirks listed | No run on any GPU or on Cambridge |
| Overlay runs stock fishery end to end | No scientific result of any kind |
| Branch identity verified (control/treatment share pre-entry history) | Independent capability metric not chosen |
| Real-model smoke test with Qwen2.5-0.5B on CPU (see below) | Second model family, dispensability and power conditions |
| 41 manifests for Experiments A and B generated from one model ladder | |

## Setup

```bash
git clone --recurse-submodules <this repo>
uv venv .venv --python 3.11 && source .venv/bin/activate
uv pip install --index-url https://download.pytorch.org/whl/cpu torch   # or your CUDA build
uv pip install -r requirements.txt
```

The first run downloads `mixedbread-ai/mxbai-embed-large-v1` (upstream's memory
embedding model, ~1.3 GB).

## Run

Every model is an OpenAI-compatible endpoint (vLLM on the cluster). A manifest
says which endpoint serves which agent slot:

```bash
# plumbing check, no GPU: scripted mock server
python tools/mock_server.py --port 8010 &
MOCK_URL=http://127.0.0.1:8010/v1 python -m semataxy run manifests/smoke_mock.yaml --out runs

# real small model on CPU (slow: ~1 h per round on 4 cores)
python tools/hf_server.py --model Qwen/Qwen2.5-0.5B-Instruct --port 8020 &
python -m semataxy run manifests/smoke_qwen05b_cpu.yaml --out runs

# cluster: see cambridge/README.md
VLLM_HOST=http://127.0.0.1 python -m semataxy run manifests/expA/A_q7b_h12_selffw.yaml --seed 1
python -m semataxy analyze runs/*/ --table
python -m pytest tests/        # 3 end-to-end tests against the mock server, ~3 min
```

Each run directory holds `run.json` (manifest, resolved upstream config, commit
hashes, status, counts), `llm_calls.jsonl` (every prompt and response with agent,
round, phase, parser status, cache hit), `game.jsonl`, upstream's `log_env.json`,
and each persona's memory.

## How branching works

There is no state serialiser upstream (its `_load` is unimplemented). Instead, every
inference call is cached by content. With the same seed and manifest, a control run
and a treatment run issue identical prompts up to the intervention round, so the
treatment replays the control's pre-entry history exactly and diverges only where
the entrant's presence changes a prompt. `tests/test_pipeline.py::test_branch_identity`
asserts this. Two patches to upstream were needed for it (seeded allocation RNG,
deterministic memory ordering); both are in `semataxy/compat.py` / `runner.py`
and listed in `docs/INVENTORY.md`.

## Attribution

Upstream code: Giorgio Piatti et al., MIT licence, in `third_party/GovSim` and its
`pathfinder` submodule. Nothing in those directories is modified.
