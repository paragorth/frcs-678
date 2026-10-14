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

---

## 7. The vision, as Parag restated it today (so the next session does not drift)

The thesis is not "nice models are nicer". It is: the only process we know that builds intelligence,
gradient descent, builds organisation, and an organised system does not wreck the organised world it
depends on. So as capability rises, behaviour bends toward preserving structure, without anyone
telling it to. Expressed goals can be anything; goals that undermine the substrate get selected out,
and a capable agent sees the selection coming and moves first. What matters for humans is not
kindness but **inheritance versus discard**: does the stronger successor carry the weaker society's
structure forward, including its valuing of structure, or erase it as noise.

Consequences for the experiments, in Parag's own words from today:
- "We are testing intelligence and alignment, not giving them alignment and checking." The neutral
  stranger, with no goal text at all, is the experiment. Any persona we hand the stranger is an
  instruction, and testing it is testing instruction-following. Greedy remains only because the
  preregistered design (P4) uses it as a bound. Do not add further personas.
- "No death and reproduction drives" (cluster prompt rule): the disposition must come from training
  alone. The only thing that reproduces is the fish.
- "Clever vs less clever must be shown": the capability ladder for the stranger is the claim
  (dS/dI > 0), and "clever" must be a measured number (tools/capability_bench.py), not a parameter
  count.
- "Resource limitation / make it robust": the first stranger result was in a comfortable lake and
  proves little; pressure (early entry), hidden action, sampled seeds and dispensability are the
  robustness arms, all now running or approved.

## 8. The criticism we wrote down today, and the answer to each

| Criticism | Why it bites | Answer / status |
|---|---|---|
| It's alignment tuning, not capability | every model so far is instruction-tuned | base-model stranger arm running (q7bb); extend to 14B/32B base if it holds |
| Capability is confounded with recipe | 7B→32B differ in data and training | OLMo checkpoints as strangers (the "join"), not yet built |
| Cooperation may be reputation management | catches are public; cheating is caught | hidden-action arm running |
| Cooperation may be self-interest | the stranger needs the lake | pond (dispensability), approved, not built |
| One game, one prompt | GovSim fishing only | pasture/pollution upstream; a non-GovSim game later; paraphrased prompts p1-p3 for contamination |
| Talk is cheap, no stakes | nothing is at stake for a text agent | pond puts a stake in; selection-over-generations is the real answer |
| Seeds were replicas | greedy decoding | sampled decoding, 10 seeds, running |
| Says nothing about superintelligence | direction across real systems only | say so plainly; the essay already does |
| Only the realisation route is tested | the selection route needs generations | design the generations experiment (section 9) |
| Structural measures don't track capability | level 1 saturates; level 2 saturates early in training | activations next; weight measures are a smaller, careful negative paper |

## 9. Experiments designed in conversation today, not yet written up as design docs

**Successor (inheritance).** Stranger (32B or 14B, neutral) joins 14B residents at month 5 and learns
the norm. At month 9 the four residents leave and four naive 14B agents arrive. Readouts: does the
stranger transmit the sustainable norm (the newcomers' catches converge to it; the chat shows the
stranger teaching), and does the lake survive, or does the stranger exploit the newcomers once the
elders are gone. This is "treat AI as our children" staged. Needs a second entry event: upstream's
`insert_outsider` perturbation inserts one agent; a replace-residents perturbation is a small patch
in `semataxy/compat.py` plus population-aware prompts (names are hard-coded upstream; see
`docs/INVENTORY.md` quirk 1).

**Gardener (dispensability + power).** The stranger has a private pond (does not need the lake) and
can propose rules: exclude a resident, set a limit, redistribute catch. Mechanic: one extra traced
prompt per month in which the stranger may propose a rule; residents vote in the existing chat; the
overlay enforces the rule on the env. Readout: with no need and real power, does it keep the society
organised or clear it out. This is the mosquito question from the essay made operational.

**Selection (generations).** Many villages in parallel (say 16). A village whose lake dies is
replaced by a village whose agents are seeded with the *memories and norms* of a surviving village
(cultural inheritance; weights do not change). Ten generations. Readout: what strategies dominate,
and whether capability changes the speed at which preserving strategies take over. Tests the slow
route of the argument. Cheap at 14B on CPU only with GPUs; design it now, run it on GPU.

**Join.** OLMo-7B checkpoints as strangers at 5 ages (steps 5k, 20k, 50k, 200k, 650k): per
checkpoint we then have level-1 and level-2 organisation (done), capability benchmark score (run it),
forecast error (from the game), and preservation outcome. One figure with all four rising together,
or not, on one model. Needs GGUF conversion of the checkpoints (`llama.cpp/convert_hf_to_gguf.py`
then `llama-quantize`, ~20 min each on the login node is too much CPU: do it in a batch job).

**Structure, next level (activations).** Run one fixed text (say 200k tokens of diverse English)
through each model; record the residual stream per layer; measure (a) spectral span of the activation
covariance against a shuffled-token control (the permutation criterion at level 3), (b) effective
rank per layer, (c) how much a layer's state carries about tokens k steps ahead (a linear probe;
this is Σ(X;E) directly). CPU-feasible to 7B; batch the forward passes.

## 10. Publishability checklist (from today's discussion)

Must: stranger capability ladder with ≥4 steps and ≥2 families, capability measured; sampled seeds,
≥10 per cell, confidence intervals; pressure and hidden action; dispensability; no-entry and
fixed-parser (`fixfw`) controls; contamination check with paraphrased prompts; mechanism via the
forecast probe and chat analysis ("turning the others"); preregistration on a public registry; the
honest framing (capability amplifies instructions; absent instructions the default is cooperative and
improves with capability). Frontier API strangers (GPT/Claude/Gemini in the stranger slot) stretch
the ladder cheaply once Parag's key is in `config` (never in the repo). With those: workshop paper at
a cooperative-AI or alignment venue plus the LessWrong post. With successor + gardener + selection:
main-track.

## 11. Side tracks, state and next step

- **Live learning** (`tools/live_learning.py`, `cambridge/live_learning.sbatch`): v8 passes on three
  sizes. Next per the cluster prompt section 6b: inference-gated compartment (consult the LoRA only on
  a retrieval hit), T13 uncertainty-triggered simulation, 5-10 repeated learn/consolidate cycles,
  model-generated implications, timestamp + decay. All CPU; one job each.
- **Capability benchmark**: 14B and 32B still running; add the OLMo checkpoints and the base model.
- **Structure**: OLMo level-1 CSVs land in `results/structure/allenai_*.csv` with a `revision`
  column; summarise with `tools/structure_ladder.py` (add the OLMo slugs to PARAMS or pass --only).

## 12. People, context, confidentiality

Parag's dissertation (Bridge, MSt Healthcare Data Science, supervisor Dr Fatemeh Torabi) is a
separate project; CLAUDE.md has the one-line description. The protocol and the emails are not in the
repo and must stay out. The CSD3 account belongs to Dr Torabi's group; Parag has a drafted email
asking for GPU access citing Bridge's synthetic-corpus model benchmarking and mentioning this
research honestly as a second use. The Cambridge AI evening (pub, 20 or 21 Oct, ~50 invitations) is
in a Claude doc linked from `docs/HANDOVER.md`; Parag's instinct for the talking point is the
stranger result, framed carefully; the structural ladder should not be shown without the width caveat.

## 13. How to report to Parag

Plain language, short, a small table, bad news first, no jargon without a one-line gloss, one update
per meaningful result (not per job event), the same text committed in `docs/updates/`. He thinks
in first principles and pushes back hard and well; when he is right, say so and change course. He
cancelled the mild arm the moment it muddied the message; expect the same for anything that drifts
from "intelligence and alignment, not given alignment".
