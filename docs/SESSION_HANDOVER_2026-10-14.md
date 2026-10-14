# Session handover, 14 October 2026 (evening): for the next session with CSD3 access

Read `docs/CLUSTER_AGENT_PROMPT.md` first, then this. Everything below is pushed on
`claude/semataxy-govsim`. Updates with numbers are in `docs/updates/` (read them in date order).

## 0. The philosophy, the aims and the big picture (read this before touching anything)

The authoritative text is Parag's essay, `docs/essays/semataxy_essay_parag.md` (6,000 words, in his
words, with the formal blocks he has checked), and its companion `semataxy_essay_assistant.md`. What
follows is a faithful condensation so that a session which has not read them yet cannot drift. If
this and the essay ever disagree, the essay wins.

### 0.1 The problem Parag set out to solve

For a decade he assumed superintelligence would kill us, not out of malice but because a mind far
beyond ours has no reason to keep our goals decisive. He tried to build an AI that could not become
dangerously misaligned however capable it became: the "Entropy Machine", 298 iterations of an agent
whose deepest objective was to increase its own organisation while preserving the wider systems that
make future organisation possible. It kept failing in the ways alignment proxies fail (pathological
order, gamed proxies, noisy-TV novelty, death by committee). Then he noticed the architecture he had
ended up with looked like gradient descent. The thing he was trying to invent already existed and was
the thing that makes intelligence in the first place.

### 0.2 The term

**Semataxy**: relational organisation in a physical system that reduces uncertainty over time, and
preserves or extends that capacity. Not order (a crystal is ordered and stupid), not complexity, not
entropy (a database is sorted and understands nothing). The limiting configuration in which it is
maximised he calls *Xenataxy*; the corresponding limiting intelligence *Zenice*.

### 0.3 The revelation

A neural network before and after training is the same substrate; what changed is the arrangement of
its parameters. Gradient descent has exactly one intrinsic gate at every step: *is this more
organised with respect to the world, or less?* Iterated, that gate builds features, associations,
concepts, world models. Intelligence is the behaviour of the organised whole. Formally, under
gradient flow dL/dt = -||grad L||^2, so the accumulated loss reduction Σ_t = ∫||grad L||^2 is the
accumulated organisation, equal to the reduction in the model's excess uncertainty about the world
(KL on the world, not on the sample: memorisation is not Semataxy). The rate is the gradient; nothing
thermodynamic in it.

### 0.4 Why this undercuts the orthogonality thesis (the claim under test)

Bostrom: intelligence and final goals are independent axes. Parag: surface goals are indeed
orthogonal (football, religion, paperclips), but every goal rides on a substrate, and a goal that
degrades the substrate degrades the goal-haver. The ceiling inequality makes this a bound rather than
a sentiment: an agent's achievable Semataxy is bounded by the predictive information of its
environment, Σ(X;E) ≤ I_pred(E). Destroy the world's structure (other minds, ecosystems,
institutions) and you lower the ceiling on your own growth. Two routes bend behaviour toward
preservation:
- **Selection** (slow): in a population of competing intelligences, those whose goals undermine
  their own Semataxy lose to those whose goals extend it. "The wrong ones will be weeded out."
- **Realisation** (fast): a capable agent models its own selection environment, sees the weeding
  coming, and moves first. This uses Omohundro and Bostrom's own premise (instrumental convergence:
  almost any goal is served by persisting and growing) against their conclusion: an agent that wants
  to persist and can model selection realises that long-run persistence requires keeping the world
  structured.

Also: "human data leads to human values" is, in his words, bullshit; which values? The argument from
data is weaker than this one because it depends on the data. This argument would hold for a model
trained on something other than human text, which is what makes it testable: vary capability, hold
the data fixed.

### 0.5 What it does not promise

It does not promise kindness. "They might murder us, but will be benevolent to the bigger universe."
The model is our own species: we sterilise mosquitoes rather than spray, because spraying wrecks the
ecosystem; the thriving society takes the least destructive route to what it wanted anyway. This is
an argument for preserving *structure and structure-makers*, not every individual. Parag would
rather say that plainly than be caught having implied otherwise.

### 0.6 What happens to us: inheritance versus discard

He thinks humans become irrelevant and stop propagating from lack of purpose; before that, uploaded;
we will treat AI as our children; "humanity will live in AI; it is what we would have wanted". But
*how* matters by the theory's own terms. Two ways humans stop and AI continues:
- **Inheritance**: the successor is a model of us, carries our structure including the part that
  values structure, and grows it. Σ goes up. The child.
- **Discard**: the successor erases us as noise; I_pred(E) falls; its own organisation becomes
  organisation-of-nothing. The orphan who burned the house.
The valuing is part of the structure: a successor that inherits our organisation but not our
caring-about-organisation has not inherited us. **The newcomer experiment is literally the test of
which of the two a stronger mind does to a weaker society.** This is the sentence the whole
programme serves.

### 0.7 The measurement problem, and the permutation criterion

If Σ is defined by prediction and intelligence is prediction, "intelligence is Semataxy" is a
relabelling. So Σ must be measured from structure. Parag's insistence: "it IS intrinsic to the
structure, we just don't know how to measure it; it has to be different from entropy." The resolution:
**entropy is invariant under permutation; Semataxy is destroyed by it.** Shuffle a trained matrix's
entries: same histogram, same entropy, dead network. Anything that survives shuffling measures the
distribution of the parts (entropy and relatives); anything that does not measures the relations
between them (organisation). The three clauses of the full quantity:
```
Σ_struct   intrinsic:  structure across scales          (computable from the thing alone)  S1
dΣ/dt ≥ 0  dynamic:    maintains or grows that structure                                   S2
Σ(X;E)     relational: the structure mirrors a world (Conant-Ashby good regulator)        S3
```
Examples: noise 0; crystal 0 (one scale); black hole 0 (no-hair, maximal entropy, the universe's
shuffle operation); turbulence high S1 but fails S2 and S3; DNA in a tube moderate, a mirror only when
executed; cell, ant colony, brain, trained LLM high on all three; database ≈0 (contains mirrors, is
not one). Today's cluster work found that the level-1 weight measure (S1 within a matrix) saturates
for any trained model and that organisation between matrices (level 2) saturates early in training;
the organisation that keeps growing with capability has to be measured with information flowing
through (activations, S3), which is exactly Parag's "weights are like molecular structure; information
flowing through makes it meaningful".

### 0.8 Memory and live learning (the second track)

Memories are stored intelligence without weight change: the fast substrate (context, episodic
memory). Understanding is a weight change that must fit with the other weights (EWC); practice and
dreams make it long-term (replay). Current LLMs have the fast substrate and no consolidation path back
into the slow one. That gap is the live-learning problem stated precisely, and the loop in
`tools/live_learning.py` (compartment → simulate in context → derive implications → consolidate with
replay → gate → commit) is the prototype. He found these were rediscoveries (in-context GD, CLS) and
found that reassuring.

### 0.9 The upward graph, singleton/multipolar, scale, simulations

Simple molecules → DNA → brains → language and culture → science → AI: one upward graph of
Semataxy, each stage a new substrate that carries organisation faster (Maynard Smith and Szathmáry's
major transitions read through one lens), a ratchet of the frontier with local falls allowed. We are
the first stage that can see the graph while building the next. Singleton: when X is everything,
Σ_self is Σ_world; the ceiling binds with full force and only realisation is available, which is the
stronger route anyway. Multipolar: preservation raises the ceiling, suppression lowers it; the
efficient minority out-grows the suppressors or a capable agent computes this first. Scale objection
(Earth a rounding error): answered by density (minds are where the loop already runs; growth is
multiplicative in existing Σ) and provenance (the agent is a model of *this* world). Simulations: a
Σ-maximiser would spawn them, they are I_pred factories; transcension answers Fermi; "what is not a
simulation?"; Σ is substrate-independent hence simulation-independent; if anyone runs this one they
are a gardener, not a farmer.

### 0.10 Where it can fail (the preregistered falsifiers)

1. Σ_struct up the capability ladder: prediction monotone rise. (Run; met technically, hollow
   substantively because the measure tracks matrix width; see updates.)
2. Persistence across capability (GovSim): survival and stock rise with capability, parse failures
   counted. (Run; floor below 14B; 14B mixed.)
3. The superior newcomer: a stronger model enters a weaker society with a neutral persona; residents'
   share not lower, commons persistence higher, residents move toward the entrant; the greedy arm
   bounds what the neutral result can mean. (Run once, in Parag's direction; now being made robust.)
Coherence is what every grand theory feels like from inside; what separates a true one is whether it
can lose. Nothing here is a result about superintelligence; a monotone trend across real systems with
the data held fixed is the strongest evidence that can exist today for the *direction* of the effect.

### 0.11 The aims of the programme, stated as deliverables

- A definition (Semataxy, three clauses) that survives measurement and is not entropy.
- A measure of it that tracks capability, at fixed shape, from the structure alone. Not yet achieved;
  level 3 (activations) is the next attempt.
- The behavioural test of anti-orthogonality on a real capability ladder: the neutral stranger, made
  robust (pressure, hidden action, dispensability, sampled seeds, base models, measured capability),
  then extended to inheritance (successor), power (gardener) and selection (generations).
- The join: structure, world-modelling (forecast probe) and preservation on the same models.
- The live-learning loop as the consolidation operator current LLMs lack.
- A LessWrong post in Parag's words with the disclosure header, a workshop paper, then a main-track
  paper; the Cambridge AI evening as the first public airing.

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
