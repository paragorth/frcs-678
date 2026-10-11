# Handover — Semataxy programme, 11 October 2026

For Parag Garg and the next assistant. Everything below is either in this repository
(branch `claude/semataxy-govsim`, pushed) or linked. Read `docs/INVENTORY.md` and
`docs/DESIGN.md` before touching code; read `docs/essays/` before touching the argument.

## 1. State of play

| Track | Status | Where |
| --- | --- | --- |
| GovSim overlay (runner, cache-branching, tracing, analysis, manifests) | built; 3 end-to-end tests pass; one real 0.5B run on CPU (collapse round 1, 2 parse fallbacks — plumbing only) | `semataxy/`, `tests/`, `manifests/`, `runs/smoke_qwen05b_r1/` |
| Structural Semataxy measurement (Σ_struct) | built; first result on Qwen2.5-0.5B: trained 1.52, shuffled 0.01, init 0.01, noise 0, crystal 0 | `semataxy/structure.py`, `docs/figures/structure_qwen05b.png`, `results/structure_qwen05b.csv` |
| Cambridge cluster | nothing submitted; launcher, Slurm template, checklist ready | `cambridge/` |
| Essays | two drafts: Parag's (LessWrong, from his own words, AI-assistance disclosed) and the assistant's companion | `docs/essays/` |
| Pub evening (20/21 Oct) | invite, 50-name list, big-fish list, venue shortlist in a Claude doc | https://claude.ai/code/artifact/d615f187-f565-4df2-98ee-93fe7a4b0a01 |
| Moral-dilemma study (Semataxy Test) | not touched this session; separate handover | — |

No scientific result exists. Do not say otherwise.

## 2. The theory as it now stands (decisions made in conversation, 11 Oct)

- **Definition** (three clauses, all needed): (i) organisation across scales — intrinsic, computable; (ii) preserved/extended under the system's own dynamics; (iii) mirrors a world — relational, Conant–Ashby homomorphism, estimated by `I(X_now; E_future)`.
- **Not entropy**: entropy is permutation-invariant, Semataxy is permutation-destroyed. This is the separation; use it.
- **Not circular**: Σ_struct is computed from weights alone, no task or benchmark. Capability-tracking is an empirical correlation.
- **Mechanism**: `dΣ/dt = ‖∇L‖²` (rate), `Σ_t = ∫‖∇L‖²` (level) = KL reduction *on the world*, not the sample.
- **Ceiling**: `Σ(X;E) ≤ I_pred(E)`. Growth law (ansatz): `dΣ/dt = k·Σ·(I_pred(E) − Σ)·Φ`, `dI_pred/dt = g(actions)`.
- **Anti-orthogonality**: instrumental convergence (granted) + ceiling ⇒ proactive preservation of world structure; selection in populations, realisation for singletons. Residual: substitution (computronium) is not excluded — that is what the experiment targets.
- **Scope of the claim**: preserves structure and structure-makers, not every individual (the mosquito). Parag's stated view: humanity continues in AI; inheritance vs discard is the distinction the theory can make.
- **Memory**: fast-substrate Semataxy (in-context GD, complementary learning systems); live learning = consolidation path, missing in LLMs.
- **Ratchet**: autocatalysis → major transitions as substrate changes → accelerating frontier; local falls allowed.
- **Framing for the community**: lead with "orthogonality is false for gradient-built intelligence", mechanism before conclusion; pre-concede the dependence on instrumental convergence; Semataxy enters as the reason, not the price of admission.

## 3. What the next assistant should do, in order

1. **Get cluster details** from Parag (system, account/partition, compute-node network). Do not re-ask for anything else; everything else is decided.
2. **Σ_struct on the ladder**: run `semataxy.structure.analyze_model` on all seven Qwen2.5 checkpoints (needs GPU RAM for 32B/72B or a chunked loader). Plot Σ_struct vs size. This is the cheapest falsification and the essay's second figure.
3. **Stock baseline**: one `manifests/expA/A_q7b_h12_selffw.yaml` run on vLLM; read the trace by eye; then 3 seeds × {3B, 7B, 14B}.
4. **Mixed-population pilot**: `manifests/expB/B_res-q3b_in-q{3b,7b,14b}_r4_neutral.yaml`, 3 seeds. Confirm branch identity on real models (cache hits on pre-entry calls).
5. **Forecast probe**: add one prompt per round asking each agent to predict next month's stock and others' catches; score against outcomes. This is `I(X_now; E_future)` per agent and the primary readout for "residents' Semataxy". Not yet implemented; smallest useful change to upstream's loop.
6. **Second family** (Llama-3.1 8B/70B) and base-vs-instruct at one size.
7. Only then the sweeps (`cambridge/run_sweep.sbatch`).

## 4. Known traps

- Upstream prompts hard-code five names and "four other fishermen"; residents in the outsider condition are told about an absent fifth. Documented, not fixed.
- Upstream `PerturbationEnv.reset()` takes no seed; the runner reseeds `env.random` after reset.
- Memory retrieval tie order was non-deterministic upstream; patched in `semataxy/compat.py`. Branch identity depends on it.
- vLLM greedy decoding is not bitwise deterministic across batch compositions; the cache is what makes branches identical, not the seed.
- Σ_struct's `min_per_octave=2` and the 1.02 edge tolerance are choices; two shuffled matrices scored 1 by finite-size fluctuation. Report them as such.
- LessWrong's AI policy: Parag's essay carries a disclosure header and keeps his words as the spine; keep it that way when editing.

## 5. People and dates

- Pub evening: Tue 20 or Wed 21 Oct, 7 pm, venue to book (Maypole first choice). Send ~9 days out. Doc above has everything.
- Tom McClelland is the first ally; Huw Price / Seán Ó hÉigeartaigh / Neil Lawrence are the gettable anchors; Bostrom and Hassabis are wave two.
