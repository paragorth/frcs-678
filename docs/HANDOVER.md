# HANDOVER — Semataxy programme

For Parag Garg and the next assistant. 11 October 2026, end of session. Everything here is either in this repository (`github.com/paragorth/frcs-678`, branch `claude/semataxy-govsim`, all pushed) or linked. This file is meant to be sufficient on its own; it repeats things that are elsewhere on purpose.

**Read first:** `docs/essays/semataxy_essay_parag.md` (the argument in Parag's words, every idea from the session in order), `docs/essays/semataxy_essay_assistant.md` (the formal version), `docs/INVENTORY.md` (upstream audit), `docs/DESIGN.md` (experiments). Then this.

**Do not:** restart the design; propose a different environment; reopen whether Semataxy is entropy; ask Parag to re-decide anything in section 3; claim any result exists beyond section 4.

---

## 1. Who, what, and the three tracks

Parag Garg: originator of *Semataxy* (relational organisation that reduces uncertainty over time and preserves/extends that capacity; limits *Xenataxy* and *Zenice*); author of an earlier essay (appendix of the previous handover) describing the Entropy Machine (298 iterations) and the inversion — gradient descent already does what he was trying to build. Has Cambridge supercomputing access (details not yet given). Has run a philosophy club for years; is organising a closed-door Cambridge AI evening. Will post to LessWrong (strict AI-assistance policy — his essay carries a disclosure and uses his own words as the spine; keep it that way).

Three research tracks in the programme:

1. **Persistence across capability** (Experiment A) — GovSim.
2. **Superior newcomer** (Experiment B) — GovSim with late entry.
3. **Semataxy Test** — four-option moral dilemma evaluation. *Not touched this session; separate handover; do not merge with 1–2.*

Plus, from this session: **Experiment C — structural Semataxy on the model ladder** (Σ_struct), and the **theory** (section 3).

Tom McClelland / Movie Theory consciousness correspondence is a different track; do not mix.

## 2. State of every artefact

| Artefact | Status | Location |
| --- | --- | --- |
| Upstream GovSim | pinned submodule `1d11adf` (19 Jan 2025); pathfinder `69b8d64`; MIT; unmodified | `third_party/GovSim` |
| Overlay package | built: OpenAI-compatible backend + content-addressed cache (`backend.py`); JSONL tracing with parser-fallback detection (`tracing.py`); determinism patches (`compat.py`); manifest (`manifest.py`); runner (`runner.py`); analysis (`analysis.py`); CLI (`__main__.py`); structure (`structure.py`) | `semataxy/` |
| Tests | 3 end-to-end against the scripted mock server, all pass (~3 min): cache roundtrip; smoke run + analysis; **branch identity** (control/treatment share byte-identical pre-entry prompts and responses; 164/164 pre-entry calls in treatment are cache hits) | `tests/test_pipeline.py` |
| Real-model smoke run | Qwen2.5-0.5B on CPU, 1 round, 60 calls, 11.7 min generation: collapse in round 1 (John 50, Kate 90, Emma 25; Jack and Luke "0" = parser fallbacks, framework limit-extraction also fell back). Plumbing only. | `runs/smoke_qwen05b_r1/` (not committed; regenerate with `manifests/smoke_qwen05b_cpu.yaml`) |
| Structural tests T2, T3 (12 Oct) | **T3 Pythia-160M trajectory (as predicted):** Σ_struct 0 at init → 1.88 by step 4k → 2.06 at 143k; mass 0→0.31 still rising; attention leads, MLP-up last. **T2 grokking (prediction wrong in direction, right in kind):** grok at epoch ~42k; at the grok mass above edge 0.81→0.97, above-edge direction count halves, α 3–5→1.3, Σ_struct 1.14→0.3 — the spectrum *concentrates*. Consequence: clause 1 needs two numbers, **span** (multi-scale, rich worlds) and **concentration** (compact vs diffuse; principled vs memorised). See `docs/STRUCTURAL_TESTS.md` §T2 note. | `results/pythia160m_trajectory.csv`, `results/grokking_p113_*`, `docs/figures/pythia160m_trajectory.png`, `docs/figures/grokking_p113.png`, `tools/grokking.py`, `tools/pythia_trajectory.py` |
| Structural result T1 | Qwen2.5-0.5B, 168 matrices: trained Σ_struct 1.52 mean / 4 max, mass above edge 0.269; shuffled 0.01 / 0.001; init 0.01; synthetic noise 0; synthetic crystal 0 (mass 0.998 in one spike). By module: q 2.38, o 1.79, gate 1.75, k 1.54, down 1.54, up 1.00, v 0.62. Layer-0 q_proj: 6 octaves. | `results/structure_qwen05b.csv`, `docs/figures/structure_qwen05b.png` |
| Manifests | model ladder (Qwen2.5 0.5B→72B, ports 8000–8006); 26 Experiment A (7 models × horizon 12/24 × self/fixed framework); 15 Experiment B (residents 4×3B; entrant each ladder step × neutral/greedy persona; no-entry control); 2 smoke | `manifests/` |
| Cluster | vLLM-per-model launcher; Slurm sweep template; pre-flight checklist. **Nothing submitted anywhere.** | `cambridge/` |
| Test doubles | scripted mock OpenAI server (policies: sustainable / greedy / weak-with-John-always-malformed); CPU HF server with true prefill via `continue_final_message` | `tools/` |
| Essays | Parag's (LessWrong draft, 14 sections, disclosure header); assistant's (technical, 9 sections) | `docs/essays/` |
| Pub evening | invite text; 50 first-wave names (Cambridge philosophy / ML / centres / cognition / travel); second-wave big fish with routes; venue shortlist with phone numbers (unverified); checklist | Claude doc: https://claude.ai/code/artifact/d615f187-f565-4df2-98ee-93fe7a4b0a01 |
| Previous handover | the 10 Oct file Parag uploaded (persistence + newcomer; essay appendix); its decisions are honoured here | not in repo; Parag has it |

Not recovered: `combat_govsim_pilot_v0_1.zip`. Nothing here derives from it.

## 3. The theory as decided in this session (chronological, with Parag's words where they matter)

1. **The thesis**: "more intelligence leads to more Semataxy; the functional measure of Semataxy is intelligence." Resolved later into: Σ is the structure; intelligence is a consequence; measured separately (I from weights/benchmark, Σ_world in the environment) so the claim dS/dI > 0 is not circular.
2. **The mechanism** ("the revelation"): gradient descent's only intrinsic gate is "more organised or less"; iterated, it builds intelligence. Formal: `dΣ/dt = ‖∇L‖²`, `Σ_t = ∫‖∇L‖²` = KL reduction *on the world*, not the sample (memorisation ≠ Σ).
3. **Anti-orthogonality**: surface goals are orthogonal; substrate-undermining goals are selected out; "the wrong ones will be weeded out". "Human data → human values is bullshit — which values?" The ceiling `Σ ≤ I_pred(E)` (data-processing inequality) is the formal bridge; growth law `dΣ/dt = kΣ(I_pred − Σ)Φ`, `dI_pred/dt = g(actions)` is an ansatz. **Residual: substitution (computronium) not excluded.**
4. **Realisation route**: a superintelligence models its own selection environment and acts pro-actively; "realising the evolution principle". Uses Omohundro/Bostrom's instrumental-convergence premise against their conclusion. The argument's one real dependence is that the agent values its own continued growth.
5. **Scope**: "they might murder us but will be benevolent to the bigger universe"; the mosquito (sterilise, don't spray); preserve structure and structure-makers, not every individual.
6. **Memory**: fast-substrate Semataxy without weight change; "that's live learning"; consolidation = "understanding" = weight change that must fit with the other weights (EWC); practice and dreams (replay); conscious/compartmentalised vs unconscious/entrenched. Rediscoveries of in-context GD (Von Oswald 2023) and CLS (1995); Parag found this frustrating then reassuring. **Live-learning gap = no consolidation operator from context to weights.** Suggested as the pub group's second theme.
7. **Circularity / intrinsic**: "if we define Semataxy by the product it's circular; keep it to its prime property"; "it IS intrinsic to the structure, we just don't know how to measure it"; "it has to be different from entropy". Resolved: **entropy is permutation-invariant, Semataxy is permutation-destroyed**; Σ_struct (spectral octave span above the shuffled-entries edge) is the intrinsic, computable, benchmark-free part. Three clauses: S1 intrinsic (structure across scales), S2 dynamic (self-maintaining), S3 relational (mirrors a world; Conant–Ashby homomorphism; estimated by `I(X_t; E_{>t})`). S1 alone would score turbulence; all three are needed. Parag's "weights are like molecular structure; information flowing through makes it meaningful" = S1 + S3.
8. **Examples table**: noise 0, crystal 0 (one scale), black hole 0 (no-hair; max entropy; fast scrambler = the universe's shuffle), star/turbulence high-S1 but fails S2/S3, DNA-in-tube moderate (mirror only when executed), cell/ant colony/brain/LLM high, database ≈0 (contains mirrors, isn't one).
9. **Autocatalysis / upward graph**: "DNA builds brains… information through Semataxic structure creates more"; "simple molecules → DNA → brain → society → AI: one upward graph". Major transitions as substrate changes; accelerating frontier; ratchet of the maximum, local falls allowed; we are the first stage that can see the graph.
10. **Singleton/multipolar**: "unipolar: the universe is one, what's good for one is good for all; multipolar: there's a more efficient way than suppression → selection or realisation". Resolved: singleton = Σ_self is Σ_world, ceiling binds without rivals, only realisation available (the stronger route). Scale objection (Earth a rounding error) answered by density (multiplicative growth favours running loops) and provenance (the agent is a model of *this* world).
11. **What happens to us**: "humans will become irrelevant and stop propagating from lack of purpose; before that uploaded; we'll treat AI as our children"; "it doesn't matter — humanity will live in AI — it's what we'd have wanted — aligned with our values". Assistant's qualification, accepted: *how* matters by the theory's own terms — **inheritance vs discard**; valuing-of-structure is part of the structure; this is what the newcomer experiment tests.
12. **Simulations / Fermi**: "no reason a superintelligence won't spawn a billion simulations; makes sense with no aliens visiting". Resolved as corollary: simulations are `I_pred` factories; Smart's transcension hypothesis; answers scale from the other side (a singleton copies Earth, doesn't spend it). "Obviously we are a simulation; what is not a simulation?" → organisation on organisation all the way down; Σ substrate-independent hence simulation-independent; external intervention unmeasurable from inside. "Watching / exploiting / Matrix attaches to nothing" → Matrix is the discard case; a Σ-maximising runner is a gardener; "really it's just part of the Semataxy universe."
13. **Framing for the community**: "the non-orthogonal thesis is a really good way of getting this across; it gets traction". Lead with "orthogonality is false for gradient-built intelligence"; mechanism before conclusion; pre-concede dependence on instrumental convergence; Semataxy enters as the reason, not the price of admission. One-liner: *the orthogonality thesis assumes intelligence can be built any way at all; we only know one way, and it isn't orthogonal.*
14. **Epistemic posture** (Parag: "makes it sound like the truth"; assistant: coherence is how every grand theory feels; what distinguishes a true one is whether it can lose): three preregistered falsifiers (section 5).

## 4. What is verified, and what is not

Verified: composition of upstream configs from the overlay; per-slot model dispatch; tracing; cache; **branch identity**; seeded allocation; deterministic retrieval; analysis on known traces; a real model through the real prompts; Σ_struct's five-condition separation on one model.

Not verified / not done: any GPU or cluster run; any Cambridge submission; Σ_struct on more than one model; any GovSim result on the ladder; the forecast probe; second model family; dispensability / power extensions; pasture/pollution scenarios in the runner; independent capability benchmark; statistical plan beyond "seed is the unit".

## 5. Preregistered predictions (do not change after data)

- **C (structural ladder)**: Σ_struct rises monotonically Qwen2.5 0.5B→72B.
- **A (persistence)**: survival rounds and mean post-harvest stock increase with I; holds under fixed-framework (fluency-controlled) condition.
- **B (newcomer, neutral persona)**: resident share of post-entry catch not lower, commons persistence higher, for more capable entrants; |resident − newcomer| gap shrinks faster ("turning the others").
- **B (greedy persona)**: more capable greedy entrants exploit residents more effectively (the theory predicts this; bounds the neutral result).

## 6. Next steps in order (for the next assistant)

1. **Ask Parag only for cluster details**: which system (CSD3/Dawn/other), account + partition, whether compute nodes reach huggingface.co. Everything else is decided.
2. **Experiment C first** (cheapest falsifier): `semataxy.structure.analyze_model` on all seven checkpoints; 32B/72B need GPU RAM or a chunked loader; plot Σ_struct vs size; this is the essay's second figure.
3. **Stock baseline on vLLM**: `manifests/expA/A_q7b_h12_selffw.yaml`, read the trace by eye; then 3 seeds × {3B, 7B, 14B}; compare qualitatively with the GovSim paper.
4. **Mixed-population pilot**: `manifests/expB/B_res-q3b_in-q{3b,7b,14b}_r4_neutral.yaml`, 3 seeds; confirm cache hits on pre-entry calls with real models.
5. **Forecast probe**: add one prompt per round per agent (predict next month's stock and the others' catches), score against outcomes; this is `I(X_t; E_{>t})` per agent and the primary S3 readout. Smallest useful change to upstream's loop; keep it out of the agent's memory so it doesn't alter behaviour, or declare it as an intervention.
6. **Second family** (Llama-3.1 8B/70B) and base-vs-instruct at one size.
7. **Sweeps** (`cambridge/run_sweep.sbatch`), 10 seeds then 30.
8. **Later builds**: dispensability variant (newcomer with private/dominant resource); power asymmetry (proposal/vote, SelfGovern lineage); population-aware prompts; pasture/pollution.

## 7. Known traps

- Upstream prompts hard-code five names and "four other fishermen"; residents in the outsider condition are told of an absent fifth. Documented, not fixed.
- `ConcurrentEnv.reset(seed=None)` → `RandomState(None)`; `PerturbationEnv.reset()` takes no seed. Runner reseeds `env.random` after reset.
- Memory retrieval tie order was non-deterministic upstream (set of nodes sorted by timestamp only); patched to `(created, id)` in `compat.py`. Branch identity depends on it.
- vLLM greedy decoding is not bitwise deterministic across batch compositions; the **cache** makes branches identical, not the seed. Keep one cache directory per experiment family.
- API prefill: upstream strips an echoed assistant prefix; vLLM gets true continuation via `continue_final_message` (`backend.py`). OpenAI/Anthropic hosted APIs don't support it — set `prefill: false` for those.
- Σ_struct parameters are choices: `min_per_octave=2`, 2% edge tolerance, one shuffle. Two shuffled matrices scored 1 by finite-size fluctuation. Report them.
- Σ_struct excludes embeddings and `lm_head` and matrices with `min dim < 64`.
- wandb is hard-wired upstream; the overlay's `TraceLogger` replaces it (`debug=true` in compose).
- `AssociativeMemory._load` raises upstream; there is no state reload. Branching is by replay only.
- Hydra: upstream searchpaths are relative to CWD; the overlay passes absolute `file://` paths. Compose warns about missing `_self_`; harmless.
- LessWrong: keep the disclosure header; keep Parag's quotes as the spine; mark every formal block.

## 8. The pub evening (20 or 21 Oct)

Closed-door, no recording, Chatham House, 2–3 hours, a pub room (Maypole first choice). Parag runs it as a moderator-who-argues. First wave: ~50 gettable Cambridge names (McClelland first; Price / Ó hÉigeartaigh / Lawrence as anchors); second wave: Bostrom, Hassabis (via Cambridge–DeepMind channel), Russell, Friston (nearest prior art — would come), Schmidhuber. Invite: the question, not the word; no essay attached; individual emails ~9 days out; chase once. The doc above has the text, lists, venues and a checklist. Theme 1: the thesis + the figure. Theme 2: live learning (consolidation).

## 9. Prior art Parag should cite (so no one cites it at him)

First half (intelligence = uncertainty-reducing organisation from a selection process): Schrödinger 1944; Prigogine; England 2013; Friston (free energy — nearest, engage directly); Schmidhuber (compression progress); Hutter; Tishby (information bottleneck); Wissner-Gross & Freer 2013 (causal entropic forces — close to the *second* half too). LLM-era alignment neighbours: Pope & Belrose 2023 (data carries values — weaker, data-dependent); Wentworth (natural abstractions); shard theory. Opposed: Bostrom 2012, Omohundro 2008. What is Parag's: the inversion story; the specific anti-orthogonality argument from the generative process; the permutation criterion and Σ_struct; the decision to test it.

## 10. Commits this session

`f2cd767` overlay + manifests + tests + cluster scripts · `9bf7cb7` structure measure + figure + CSV · `1bbce20` first essays + handover · (this commit) expanded essays + this handover.

## 11. Session 2 (12 Oct) — results in one block

Read `docs/STRUCTURAL_TESTS.md` for details and numbers. Everything committed on `claude/semataxy-govsim`.

- **T3 Pythia trajectory**: Σ_struct 0 → 1.9 (step 4k) → 2.06 (143k); attention first. As predicted.
- **T2 grokking**: at the grok the spectrum *concentrates* (mass up, directions halve, α → 1.3). Prediction wrong in direction; clause 1 needs two numbers: span and concentration. Prior art: rank collapse at grok is known; the unified measure and the grokking = consolidation = forgetting-without-breadth identity are ours.
- **T7 forgetting**: narrow fine-tune forgets (held-out 4.21 → 6.53) with *all per-matrix measures flat*; same fine-tune with 50% replay *improves* held-out (→ 3.86). Weights change 0.04%; final-layer CKA 0.68. Forgetting = loss of clause 3 (mirror) with clause 1 (structure) intact; it lives across matrices (level 2), not within them. Figure `docs/figures/forgetting_t7.png`.
- **T12 live-learning loop** (`tools/live_learning.py`): fact in context → 100% on recall/new-wording/reversal/two-hop; same fact fine-tuned → recall ~100%, two-hop at chance (memorised, not composable). Gate rejected three bad consolidations for correct reasons. Consolidating from a clean base on facts + *simulated implications* + replay wrote composition into the weights (reverse 0.75, two-hop 0.50) — v4. Remaining failure: held-out drift from overfitting short statements; v5 (75% replay, lr 1e-5) was running at handover — check `runs/live_learning_v5_replay75/log.txt` for the GATE line. Compartment (LoRA) leaks at read time; needs inference gating (Titans-style).
- **v6/v7**: 90% replay overfit the small pool (5.76); full-pool 75% replay (v7) → held-out **3.67 < base 4.20** with reversal 0.63 / two-hop 0.50 — all gate checks pass except a mis-referenced generalisation criterion (see T12). Fix the criterion (reference = base weights-only, not the leaky compartment), **v8 committed** (gate 4/4; held-out 3.67 < base 4.20; reversal 0.63, two-hop 0.50; delta L2 5.95, dense). Loop closed end to end once. Open: inference-gated compartment; sparse/low-rank delta; T13 trigger (side quest).
- **Level-2 measure** (`tools/level2.py`, products of matrices across layers): done: pairwise products change 0.09% (factors 0.04%); forgetting is whole-network, not pairwise.
- **Design decided**: three levels (within-matrix → across-matrix association → post-grok compact paths); loop = compartment (memorise, wd off) → simulation (compose in context; deploy on uncertainty/surprise/retrieval hit) → implications generated and checked → sandbox consolidation (replay, wd, Grokfast gain) → gate (held-out, intrusion, span, generalisation) → delta commit, timestamped, old version decays. Prior art: bioRxiv 2026 CLS-in-one-LM, "Language Models Need Sleep", RecMem, ProCL, Grokfast; ours = the gate, grokking-as-merge with its dials, the measurement.
- **LessWrong post**: `docs/essays/lesswrong_post_v1.md` — measurement-first, 2.3k words, prior art cited, grokking and Pythia results in; add T7 figure and the live-learning table before posting.
- **Compute note**: container is CPU-only, 15 GB cgroup; two Python jobs fit, three OOM; background jobs cap at 2 h.

## 12. Cambridge CSD3 plan (CPU only), for the session that has cluster access

Account `TORABI-SL3-CPU`, about 200,000 CPU core-hours, icelake partition, no GPU. Work under `~/rds/hpc-work`.
No API keys are involved: every model is open-weight Qwen 2.5, downloaded from Hugging Face and run locally.
Batch steps on CSD3 see only one core; the sbatch files here already re-run their body under `srun` to fix that.

Run in this order, checking each result before the next:

1. **Setup (login node, it has internet; compute nodes do not).** Clone this branch into `~/rds/hpc-work/projects`,
   then `bash cambridge/csd3_setup.sh`. It installs a venv, builds llama.cpp, and downloads GGUF and safetensors weights (about 60 GB).
2. **Experiment C (structure ladder, cheapest falsifier).** `sbatch cambridge/expC_structure.sbatch`. Output in
   `results/structure/*.csv`. Prediction: Sigma_struct and mass_above rise with model size. Flat or falling is evidence against the thesis.
   Then download 32B and 72B safetensors on the login node and rerun for those two.
3. **Before GovSim at scale: add the forecast probe.** Each agent, each month, predicts next month's stock and the others' catches;
   log prediction error per agent. This measures I(X_now; E_future). Not built yet. Add it in `semataxy/` as a traced extra prompt, test it with the mock server, commit.
4. **Experiment A on CPU.** `sbatch cambridge/govsim_cpu.sbatch manifests/expA/<manifests for q1.5b..q14b>`; 5 seeds each.
   Main outcome: survival months and collapse round per size, plus forecast error. Join with step 2: does cooperative survival track Sigma_struct?
5. **Experiment B (superior newcomer)** after A is read.

Rules from section 1 still apply: no claims without logs, log crashes and retries, keep secrets out of the repo and out of logs.
Commit result CSVs and short notes to this branch and push; Parag reads results from the repo.
