# Stage 0 inventory: what exists upstream, what this overlay adds, what is still required

Audited 11 Oct 2026 against GovSim `1d11adf` (19 Jan 2025) and pathfinder `69b8d64` (2 Sep 2024).
The previously described `combat_govsim_pilot_v0_1.zip` was not available to this session;
nothing here is recovered from it. Everything in `semataxy/` is new code.

## Upstream functionality (unmodified, imported as a pinned submodule)

| Area | What upstream provides | Where |
| --- | --- | --- |
| Scenarios | Fishery, pasture, pollution; identical mechanics (stock 100, doubles to cap, collapse at <5) | `simulation/scenarios/*` |
| Round structure | lake (decide + harvest) → pool_after_harvesting (observe) → restaurant (group chat, ≤10 utterances) → home (reflect) | `scenarios/common/environment/concurrent_env.py` |
| Agents | Generative-agent persona: perceive → store (LLM importance 1–10) → retrieve (recency+importance+embedding) → act / converse / reflect | `simulation/persona/*`, `scenarios/fishing/agents/persona_v3/*` |
| Prompts | System prompt v3 (+ paraphrases p1–p3, nocom); "think step-by-step"; act answer parsed by regex `\d+` with default 0 | `persona_v3/cognition/*_prompts.py`, `utils.py` |
| Framework model | Conversation summary, agreed-limit extraction (`Answer: N` / N/A) | `converse_prompts.py`, `reflect_prompts.py` |
| Mixed models | `mix_llm` list, one model per persona slot, plus `framework_model` | `simulation/conf/multiple_llm.yaml`, `main.py` |
| Late entry | `insert_outsider` perturbation: 4 agents for rounds < R, 5th agent initialised at round R with its own (empty) memory | `perturbation_env.py`; config `fish_perturbation_outsider` (R=3, max 15 rounds) |
| Outsider persona | Greedy goal text appended to the system prompt | `persona/outsider_fisherman.yaml` |
| No-language ablation | `change_language_nature` perturbation | `fish_perturbation_no_language` |
| Universalization | Injected sustainability hint | `*_universalization` configs |
| Logging | wandb traces; `log_env.json` per run (harvest rows, utterances, agreed limits); per-persona `nodes.json`/`embeddings.json` | `simulation/utils/logger.py`, env `save_log`, memory `save` |
| Inference | pathfinder: transformers / vLLM in-process; OpenAI, Azure, Anthropic, Mistral, OpenRouter APIs routed by model-name substring | `pathfinder/pathfinder/loader.py`, `api.py` |
| Subskills | Separate benchmark of arithmetic/sustainability reasoning | `subskills/` (not used yet) |

### Upstream quirks found in the audit (left as-is for baseline fidelity, flagged for later)

1. **Population size is baked into prompts.** `persona_names = ["John","Kate","Jack","Emma","Luke"]` and "along with four other fishermen" are hard-coded (`utils.py:get_sytem_prompt_*`). In the outsider condition the four residents are therefore told about a fifth fisherman who has not arrived. Upstream accepted this; any population-size change needs a prompt change.
2. **Unseeded allocation RNG.** `ConcurrentEnv.reset(seed=None)` creates `RandomState(None)`; `PerturbationEnv.reset()` accepts no seed. Stochastic allocation when demand exceeds stock was therefore unseeded upstream. Fixed in the overlay (see below).
3. **Non-deterministic memory ordering.** `RetrieveComponent.retrieve` builds a `set` of nodes and sorts by timestamp only; ties fall in object-hash order, which changes prompt text between processes. Patched in the overlay.
4. **Memory load is unimplemented** (`AssociativeMemory._load` raises). Save/reload of agent state is not available; branching is done by replay instead.
5. **API prefill.** For OpenAI-style APIs upstream sends the assistant prefix as a trailing assistant message and then strips it from the reply if echoed. True continuation is only possible with vLLM's `continue_final_message`; the overlay uses it.
6. `experiment.name` interpolates `${group_name}` with a trailing slash when empty; harmless.
7. Hydra `defaults` list lacks `_self_` (warning only).
8. wandb is hard-wired (`wandb.init` in `WandbLogger.__init__`; `debug=true` disables it).

## Overlay additions (this repository, `semataxy/`)

| Need | Implementation | File |
| --- | --- | --- |
| One interface for every capability level | `OpenAICompatibleAPI` pathfinder backend: any OpenAI-compatible endpoint (vLLM), per-model `base_url`, prefill via `continue_final_message` | `backend.py` |
| Snapshot / branching | Content-addressed response cache keyed on (model, messages, temperature, top_p, max_tokens, prefill). Same seed + same manifest ⇒ identical pre-intervention history; branches diverge only where prompts diverge. Verified by `tests/test_pipeline.py::test_branch_identity` | `backend.py` |
| Run manifest | YAML: experiment id, model specs, model per slot, framework model, seed, Hydra overrides, intervention description | `manifest.py` |
| Minimum record | `run.json` (manifest, resolved config, upstream commits, patches, status, termination reason, call/token/cache counts), `llm_calls.jsonl` (agent, round, phase, prompt, response, parser_status, tokens, latency, cache), `game.jsonl`, upstream `log_env.json` and memories | `tracing.py`, `runner.py` |
| Invalid-action detection | Wrapper catches upstream's "Returning default value" warning and stamps `parser_status=default` | `tracing.py` |
| Seeded allocation | `env.random = RandomState(seed)` after reset | `runner.py` |
| Deterministic retrieval | `(created, id)` sort | `compat.py` |
| Capability-ladder dispatch | `agents: [label×5]`, `framework: label` | `manifest.py`, `runner.py` |
| Late entry with model swap | Upstream `insert_outsider` + persona override to `baseline_fisherman` + slot-4 model from the ladder; entry round via override | `manifests/expB/*` |
| Analysis | Collapse round, stock trajectory, group yield, per-agent catch, Gini, resident pre/post catch, resident share, newcomer gain, resident–newcomer gap trend, agreed limits, parse fallbacks per slot | `analysis.py` |
| Cluster | vLLM-per-model launcher, Slurm sweep template, order of operations | `cambridge/` |
| Test doubles | Scripted mock server; CPU HF server with prefill | `tools/` |

## Still required

- **Independent capability score** for each model (held-out reasoning/arithmetic set, or upstream `subskills`) — needed before any "capability" claim; parameter count is only the ladder's order.
- **Second model family** (e.g. Llama-3.1 8B/70B, Gemma-2) and base-vs-instruct at one size, to separate capability from family-specific tuning.
- **Population-aware prompts** if addition (4→5) is to be compared with replacement (5→5) cleanly; currently only addition via upstream's mechanism.
- **Dispensability manipulation** (newcomer with a private or dominant resource) so that extensions A and B can come apart; not in stock GovSim.
- **Power asymmetry** (exclusion / redistribution proposals, SelfGovern lineage) if "superior" is to mean anything beyond harvest + talk.
- **Resident prediction probes** (ask each agent to forecast next-month stock) as a direct uncertainty-reduction readout.
- **Statistical plan**: seed-level units, preregistered smallest effect, held-out seeds.
- **Pasture/pollution** scenarios: supported upstream, not wired in the runner yet (`runner.py` raises).
