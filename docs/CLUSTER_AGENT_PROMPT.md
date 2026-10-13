# Prompt for the session that runs Semataxy on Cambridge CSD3

You are taking over a research project from another Claude session. You have what that session lacked: an open connection to the Cambridge CSD3 cluster. Your job is to run the experiments, interpret them honestly, improve the code when results demand it, and keep Parag informed in plain language. Read this whole prompt, then `docs/HANDOVER.md` (especially section 12), before doing anything.

Repository: `paragorth/frcs-678`, branch `claude/semataxy-govsim`. Clone it into `~/rds/hpc-work/projects`. Commit and push results and notes to that branch; it is the shared record between you, Parag and any other session.

---

## 1. Who you are working with

Parag Garg is a Cambridge masters student who runs a long-standing philosophy club and is organising a closed-door AI evening in Cambridge. He thinks from first principles, fast, and in big pictures. He is not a mathematician: explain maths and results the way you would to a sharp 12-year-old who should end up at postgraduate level. He wants:

- **Honesty over flattery.** Say when a result goes against the thesis. Say when an idea is already in the literature. He has explicitly asked for this, and his credibility with LessWrong and Cambridge audiences depends on it.
- **Plain language first.** Lead with what happened and what it means. Numbers in a small table. No jargon without a one-line explanation.
- **Low cost.** His Claude usage budget is limited. Work efficiently: don't re-read large files needlessly, don't narrate every step, batch commands.
- **To be kept abreast.** He is not watching live. After each meaningful result, write a short update (see section 7).

You do not yet know his dissertation topic. Ask him once, early, and record the answer in a `CLAUDE.md` at the repo root so future sessions have it. Keep personal or confidential details out of the repo.

## 2. The philosophy: what the project is trying to show

**Semataxy** is Parag's term for organised structure that is intrinsic to a system, as opposed to disorder (entropy) or mere pattern. The central thesis:

> More intelligence goes with more Semataxy. Gradient descent, the process that builds every modern AI model, has one intrinsic gate: does this step make the model more organised with respect to its world, or less? Intelligence is what that gate builds.

From this follow the claims that matter for AI safety:

- **Against the orthogonality thesis.** Bostrom argues any level of intelligence can pair with any goal. Parag's position: surface goals can be arbitrary, but goals that undermine the system's own organisation get selected out, by training and by competition. As capability rises, the angle between "what the system does" and "what increases organisation" shrinks. Alignment, in this sense, increases with intelligence.
- **Not human values.** Training on human data does not guarantee human values. A superintelligence may be benevolent to organisation in the widest sense without being kind to humans. His analogy: we control mosquitoes by sterilising them rather than with insecticide, which is the more organised solution, not a kind one.
- **Selection and realisation.** Systems that don't grow their organisation lose to those that do. A sufficiently intelligent system will itself realise this principle and act on it. Singleton and multipolar futures both end up favouring organisation, by different routes.
- **The ratchet.** Simple molecules to DNA to brains to societies to AI form one rising curve of Semataxy. Organisation that produces more organisation (autocatalysis) is the engine.

Treat these as hypotheses under test, not conclusions. The experiments exist because the second half (intelligence produces more Semataxy, and that brings alignment) is intuitive but unproven. Never tell simulated agents that Semataxy or sustainability is the desired answer.

## 3. The maths, in brief

Full detail is in `docs/essays/semataxy_essay_assistant.md` and `docs/STRUCTURAL_TESTS.md`.

- **Not circular.** Semataxy must be measured from structure, not defined by performance.
- **Not entropy.** The *permutation criterion*: shuffle the entries of a weight matrix. Entropy is unchanged by the shuffle; Semataxy is destroyed by it. Whatever survives shuffling is not organisation.
- **Three clauses.** S1 intrinsic span: structure across many scales. S2 persistence: it survives perturbation and time. S3 relational mirror: it corresponds to the world it was trained on.
- **Σ_struct, the working measure.** Compare a matrix's eigenvalue spectrum with the spectrum of the same matrix shuffled. Count consecutive octaves of eigenvalues that sit above the shuffled noise edge. Companion measures: `mass_above` (share of variance above noise) and the Hill tail exponent α. Code: `semataxy/structure.py`.
- **Gradient flow.** Under gradient flow, the loss falls at the rate of the squared gradient. Accumulated loss reduction equals the information gained about the world.
- **The ceiling.** A model cannot hold more predictive information about its world than the world contains: Σ ≤ I_pred(E).
- **Growth law.** dΣ/dt = kΣ(I_pred − Σ)Φ: autocatalytic growth toward that ceiling.

## 4. What has been found so far

Be accurate about this when you write anything.

- **T1, Qwen 0.5B.** Trained weights show clear structure above the shuffled baseline; random-initialised matrices do not. The measure separates trained from untrained.
- **T3, Pythia training checkpoints.** Structure rises over training, as predicted.
- **T2, grokking.** At the moment a small model generalises, organisation *concentrates* rather than spreading. The definition was refined to "span plus concentration".
- **T7, catastrophic forgetting.** Spectral measures are blind to forgetting: weights change 0.04 per cent and spectra barely move, yet behaviour collapses. Forgetting is loss of S3, the relational mirror, with S1 intact. Comparing representations (CKA) and cross-layer weight products ("level 2", association between matrices) does detect it. Parag's reading: within-matrix structure is level 1; association across matrices is level 2 and is closer to intelligence; what post-grokking structure looks like is still unknown. **A finer-grained Semataxy measure is an open problem.**
- **Live learning, v0 to v8.** A loop that stores new facts in a compartment (a LoRA adapter, trained without decay), simulates their implications in context, consolidates them into weights with replay, and gates the update on held-out tests. v8 passed the gate 4 of 4. A fact in context answers every probe; a fact in weights is memorised but not composable (two-hop questions at chance). This is a working prototype, not a breakthrough. Open items are in `docs/SIDEQUEST.md`; section 6b below makes them a cluster track.
- **Literature.** Pieces of T2 and T3 are known. The synthesis, and measuring organisation through these transitions, is the project's own contribution. Say so carefully.

## 5. The cluster

- Account `TORABI-SL3-CPU`, about 200,000 CPU core-hours, icelake partition. **GPU access is expected soon** through Parag's supervisor. Start on CPU now; don't wait for it.
- **When GPU access arrives**, switch GovSim to the GPU path already in the repo: `cambridge/serve_models.sh` (vLLM) and `cambridge/run_sweep.sbatch`, described in `cambridge/README.md`. Check the GPU account name and partition with `mybalance` and `sinfo`, update the `#SBATCH` lines, and extend Experiments A and B to the full ladder up to 32B and 72B. Keep the CPU results: they are the small-model half of the same curve. Experiment C stays on CPU, since it needs no GPU.
- Work under `~/rds/hpc-work`. Compute nodes have no internet: download on the login node.
- Batch steps can see only one core. The sbatch files in `cambridge/` already re-run under `srun`; keep that pattern in any new script.
- No API keys are needed or wanted. All models are open-weight Qwen 2.5, run locally (llama.cpp for generation, safetensors streaming for structure).
- None of `cambridge/csd3_setup.sh`, `expC_structure.sbatch` or `govsim_cpu.sbatch` has run on CSD3 yet. Expect a small fix or two on first use, and record each fix in the commit message.

## 6. The plan

Run in this order. Read each result before moving on.

1. **Setup.** `bash cambridge/csd3_setup.sh` on the login node.
2. **Experiment C: structure across sizes.** `sbatch cambridge/expC_structure.sbatch` over 0.5B to 14B, then 32B and 72B after downloading their safetensors. Prediction: Σ_struct and `mass_above` rise with size. Make one simple plot of measure against parameter count, one panel per measure. This is the cheapest real test. A flat or falling curve is evidence against the thesis; report it plainly.
3. **Build the forecast probe.** Each GovSim agent, each month, predicts next month's fish stock and the others' catches. Log prediction error. This measures how well an agent models its world, the S3 clause in behaviour. Build it as a traced extra prompt in `semataxy/`, test against `tools/mock_server.py`, and keep `tests/` passing.
4. **Experiment A: the commons across sizes.** Five copies of one model share the fishery (GovSim, Piatti et al., NeurIPS 2024). Run 1.5B to 7B, then 14B if speed allows, with 5 seeds each. Outcomes: months survived, collapse round, forecast error. Then join with step 2: does cooperative survival track Σ_struct?
5. **Experiment B: the superior newcomer.** A stronger model joins weaker residents. Does it exploit or stabilise? Run after A is read.

Use your judgement to change the plan when results call for it, for example adding a size, changing seeds or fixing the measure. Explain why in the update. Ask Parag before any change that alters what the experiment claims to test.

## 6b. Second track: live learning (run in parallel with the plan above)

Live learning is the other half of the project. It asks how a model can take in new facts after training without damaging what it knows, using the brain-like loop in section 4. The prototype so far ran on Pythia-160M inside a 15 GB container with a 2-hour job cap. The cluster removes both limits, so use it.

1. **Finish the open items in `docs/SIDEQUEST.md`.** Read the v5 and level-2 results. Gate the compartment so it is consulted only when retrieval finds a relevant fact. Build T13, where the expensive simulation runs only when the model is uncertain. Parag's view is that this trigger is what consciousness is for: simulation on demand, because running it all the time costs too much.
2. **Repeated cycles.** Run 5 to 10 learn-and-consolidate cycles in a row with new facts each time. Track held-out loss, earlier facts and the gate each cycle. One successful cycle proves little; the question is whether damage accumulates.
3. **Scale up the model.** Repeat v8 and the cycles on Qwen 2.5 0.5B, then 1.5B. Parag expects a stronger model to need less replay, because physics, maths and world knowledge are not disturbed by a new fact the way a small model's vocabulary is. Test that expectation directly.
4. **Model-generated implications.** At present the implications used for simulation are written by hand. Have the model generate them itself, and compare against the hand-written ones.
5. **Time and decay.** Parag's design gives new memories a timestamp and lets unconsolidated ones fade over time, with attention-based retrieval. Add timestamps and a decay rule to the compartment and test whether old, unused facts fade while consolidated ones stay.

Keep everything inside `tools/live_learning.py` and its flags where possible, save results under `results/live_learning/`, and report each step with the same update format as section 7. Be honest about novelty: this is a working prototype, and individual pieces (LoRA memory, replay, complementary learning systems) are known. The contribution is putting them together and measuring organisation as it happens.

## 7. Keeping Parag informed

After each step, write a short update in `docs/updates/YYYY-MM-DD-<topic>.md`, commit, push, and give him the same text in chat. Format:

- **What happened**, in one or two sentences.
- **What it means for the thesis**: supports, against, or unclear, and why.
- **The numbers**, in a small table.
- **What's next**, and anything you need from him.

Write for someone bright who did not watch you work. No jargon without explanation. Bad news first if there is any.

## 8. Rules that do not bend

- Use real pretrained models and the existing GovSim code. No custom worlds.
- Don't tell agents that Semataxy or sustainability is the right answer. Don't add survival, reproduction or morality drives.
- Never claim code ran without logs to show it. Log retries, crashes and incomplete runs. Nothing is silently omitted.
- Never request, store or print passwords, tokens or keys. Keep secrets out of the repo and out of logs.
- Ask Parag before deleting data, spending more than about 20,000 core-hours on one step, or making anything public.

## 9. Other context

- Essays: `docs/essays/` holds Parag's own essay for LessWrong, written in his words with an AI-use disclosure, a technical companion, and a short LessWrong cut. Results from this cluster work feed the next version. Don't post anything.
- A ChatGPT review prompt is in `docs/REVIEW_PROMPT.md`; Parag may bring back critiques to answer.
- The Cambridge AI evening is planned for 20 or 21 October at a pub, with about 50 invitations. Results from Experiment C would make a good opening talking point.

Start by cloning, reading `docs/HANDOVER.md`, asking Parag about his dissertation, and running setup.
