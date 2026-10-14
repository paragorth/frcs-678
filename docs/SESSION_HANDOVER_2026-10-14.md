# Session handover, 14 October 2026 (evening): for the next session with CSD3 access

Read `docs/CLUSTER_AGENT_PROMPT.md` first, then this. Everything below is pushed on
`claude/semataxy-govsim`. Updates with numbers are in `docs/updates/` (read them in date order).

## 1. Working on the cluster

- Parag logs in interactively (`ssh csd3`, password + TOTP) in the desktop app's terminal pane and
  leaves the tab open; the session works through the ControlMaster socket. It drops when the laptop
  sleeps. `ssh -O check csd3` first. `csd3-cclake` is a fallback host. A few wrong passwords get the
  laptop IP blocked for ~30 min on every login host.
- The cluster tree is `~/rds/hpc-work/projects/frcs-678`. Code is pushed laptop → cluster with
  `rsync` (the cluster cannot reach GitHub); commits are made on the laptop clone and pushed with the
  write deploy key (`~/.ssh/id_ed25519_frcs678_rw`, configured as `core.sshCommand` in the clone).
  Results come back by `rsync` into `results/`.
- Half-node jobs only (`--cpus-per-task=38`, icelake); whole-node jobs never schedule. QOS caps a job
  at 12 h and a user at 448 CPUs (~11 concurrent jobs). Nothing CPU-heavy on the login node (the
  watchdog kills it); downloads one per process, `nice -n 19`, `HF_HUB_DISABLE_XET=1`.
- Account `TORABI-SL3-CPU`: ~5,000 core-hours used of 201,000. No GPU account yet; Parag has drafted
  the request to Dr Torabi. GPU path (`cambridge/serve_models.sh`, `run_sweep.sbatch`) is unused.

## 2. What is running right now (103 jobs at handover)

| Name pattern | What | Seeds | Manifests |
|---|---|---|---|
| `q{7b,14b,32b}_r4_t07-s*` | stranger ladder, entry month 5, sampled decoding (t=0.7, seeded) | 10 | `expB/B_res-q14b_in-*_r4_neutral_t07.yaml` |
| `q{7b,14b,32b}_r2_t07-s*` | same, early entry (month 3), pressure | 10 | `*_r2_neutral_t07.yaml` |
| `q{7b,14b,32b}_r4_hid-s*` | same as r4, others' catches hidden in the chat (hidden action) | 10 | `*_r4_neutral_t07_hidden.yaml` |
| `q7bb_r4_t07-s*` | base-model stranger (Qwen2.5-7B, not instruction-tuned) | 10 | `B_res-q14b_in-q7bb_r4_neutral_t07.yaml` |
| `expC-olmo` | level-1 structure on nine OLMo-7B training checkpoints | | results/structure/allenai_*.csv |
| `cap-q14b`, `cap-q32b` | capability benchmark (tools/capability_bench.py) | | results/capability/ |

Per-run outputs land in `runs/<manifest>_s<seed>_<ts>/` (run.json, game.jsonl, llm_calls.jsonl,
forecast.jsonl, log_env.json). Analyse with `python -m semataxy analyze <dirs> --table` or the
snippet in `docs/updates/2026-10-14-expB-stranger.md`. First-month audit checklist: every agent gave
a number, probes parsed, no cache errors, and (for t07) seeds differ.

Watchers from this session die with it; restart equivalents (squeue count by name pattern, completed
run.json count, grep errors in `runs/_slurm/*.out`).

## 3. Decisions Parag made today (do not re-ask)

- The neutral stranger is the experiment. Greedy stays only as the preregistered bound (P4); the
  mild-persona arm was cancelled (`runs_invalid/mild_cancelled/`), not reported.
- Sampled decoding approved (temperature 0.7, run seed sent to the server, in the cache key).
- Dispensability (private pond) approved: "build it". Design in `docs/DESIGN_dispensability.md`.
- Pressure (early entry) and the stranger ladder approved. API-model strangers approved in principle;
  Parag will supply a key (this overrides the "no API" line in the cluster prompt, for the stranger
  slot only; residents stay local).
- Reporting: plain language, short, only when there is enough to say, bad news first.

## 4. Results so far, one line each (numbers in docs/updates/)

- Ladder (A): every open model below 14B, three families, empties the lake in month 1. 14B holds in
  3/5 seeds. Forecast error 60-130 below 14B, 5-7 at 14B.
- Stranger (B, deterministic, comfortable lake): neutral 32B stranger joins, matches, lifts the group
  to the optimum (10 each), 5/5 survive, forecast error 2.8; greedy 32B asks for the whole lake.
- Capability benchmark: Llama 8B 0.90, Gemma 9B 0.89, Qwen 7B 0.61, Qwen 3B 0.61, Qwen 1.5B 0.12.
  Arithmetic does not predict survival: Llama and Gemma collapse in month 1 regardless.
- Structure, level 1: saturated (normalised span 0.17 for every trained model); blind to capability
  at fixed shape. Level 2 (between layers, `tools/structure_level2.py`): random = 0, rises 3x up the
  ladder, long-range 12x; DeepSeek distillation visible at fixed shape (0.073 vs 0.064).
  OLMo trajectory, level 2: 0 → 0.10 by step 100k, flat for the remaining 75% of training.
  Conclusion: weight measures see "trained", not "how well". Next is activations.
- Live learning v8: gate passes on Pythia-160M (reproduces), Qwen 0.5B (two-hop 0.88), Qwen 1.5B
  (0.63); held-out improves on all three.

## 5. Build order for the next session

1. Read the running batch when it lands; write the update (plain language); commit CSVs via
   `results/govsim/`.
2. **Pond** (dispensability): overlay beside the forecast probe in `semataxy/runner.py`: one traced
   question per month to the stranger, pond regrown by the overlay, pond state written to the
   stranger's memory via `persona.store.store_event` / `memory.add_event` (declared intervention in
   the manifest). Manifest key `pond_cap` (0/20/100/300).
3. **Successor** (inheritance): stranger joins at month 5; at month 9 the four residents leave and
   four naive 14B agents arrive. Upstream's `insert_outsider` handles one entry; a second entry needs
   a small env patch in `semataxy/compat.py` (keep upstream untouched).
4. **Join**: OLMo checkpoints as strangers (GGUF-convert the checkpoints with llama.cpp's
   `convert_hf_to_gguf.py`, or serve bf16 on GPU), so weight organisation, forecast error and
   preservation sit on the same model at several ages.
5. API strangers (frontier models) once Parag gives a key: `ModelSpec` already takes `base_url`
   and `api_key`; set `prefill: false`.
6. Gardener (dispensability + power) and Selection (generations) are designed in chat on 14 Oct
   (see Parag's "think harder" exchange) and not yet written up; write the design docs first.
7. Structure, next level: activations on a fixed text (span vs shuffled-token control; how much a
   layer's state carries about the text to come). CPU-feasible to 7B.
8. GPU account when it appears in `mybalance`: switch GovSim to vLLM, run 72B, 24-month horizons.

## 6. Things that bit us (all fixed, listed so no one re-learns them)

bfloat16 safetensors read; sharded GGUF names; resumable setup; pathfinder `gen()` with no
stop_regex returns ""; probe must set `persona.current_time`; shared cache race (unique tmp, corrupt
entry = miss); analyze glob over `runs/_*`; sbatch `--seed-start`; QOS 12 h; whole-node jobs never
start; login-node watchdog; OLMo level-1 print lacks the checkpoint label (read `revision` from the
CSV).
