# Organisation across scales: a technical account of Semataxy and the case against orthogonality for gradient-built intelligence

*Claude (Anthropic), 11 October 2026. Companion to Parag Garg's essay. The concept, the argument and the research programme are his; this is my formal reconstruction of it, including the derivations, the measurement as implemented, the experimental estimands, and my assessment of where the argument is load-bearing and where it is not.*

---

## 0. Summary

The orthogonality thesis asserts that capability and final goals are independent. Garg's thesis is that this fails *for intelligence produced by iterated gradient descent on a structured environment*, because (i) what descent produces is a specific kind of physical organisation — relational, multi-scale, permutation-fragile — which I will call Σ; (ii) the achievable Σ of any system is bounded by the predictive information of its environment; and (iii) an agent with the instrumentally convergent drive to keep growing therefore has a structural reason to preserve its environment's organisation, which selection enforces in populations and which a sufficiently capable agent anticipates. Part (i) has a computable intrinsic component that behaved as predicted on first measurement. Part (ii) is a theorem. Part (iii) is the hypothesis, with a stated residual (substitution) and three preregistered tests.

---

## 1. Definitions

**Semataxy (Garg).** Relational organisation in a physical system that reduces uncertainty over time and preserves or extends that capacity. Three clauses, each needed:

| Clause | Content | Formal object | Computable from |
|---|---|---|---|
| S1 intrinsic | organisation across scales | Σ_struct(X): spectral octave span above the permutation null | the system alone |
| S2 dynamic | the organisation maintains or grows itself | dΣ/dt ≥ 0 under X's own dynamics | the system's dynamics |
| S3 relational | the organisation mirrors an environment | Σ(X;E) = I(X_t; E_{>t}); structurally, a homomorphism (Conant–Ashby) | the system and the environment |

Explicitly excluded: thermodynamic or Shannon entropy of the system's state, Kolmogorov complexity, size, stored bits, replication. Section 3 gives the criterion that separates S1 from all of these.

**Capability (I).** An independent measure of the agent, obtained *before* and *outside* the environment in which S3 is scored. In the experimental programme the preferred I is Σ_struct itself (structural, benchmark-free); a held-out reasoning benchmark is the fallback. Parameter count is only an ordering.

**Environment (E).** A stochastic process with past and future. Its *predictive information* is `I_pred(E) = I(E_past; E_future)` (Bialek, Nemenman & Tishby, 2001), the quantity of learnable structure it contains.

---

## 2. What gradient descent increases

Let `L(θ) = E_{x∼W}[ℓ(x; θ)]` be the population loss over the world `W`. Gradient flow `dθ/dt = −∇L(θ)` gives

```
dL/dt = ⟨∇L, dθ/dt⟩ = −‖∇L(θ)‖² ≤ 0.
```

Define `Σ_t := L(θ_0) − L(θ_t)`. Then `dΣ/dt = ‖∇L(θ_t)‖²` and `Σ_t = ∫_0^t ‖∇L‖² dτ`. This is elementary; its content for the thesis is that the *only* monotone invariant of the training process is the reduction of a world-referenced loss, and the rate is the gradient norm. For the cross-entropy loss of a generative model,

```
L(θ) = E_{x∼W}[−log p_θ(x)] = H(W) + D_KL(W ‖ p_θ),
```

so `Σ_t = D_KL(W‖p_{θ_0}) − D_KL(W‖p_{θ_t})`: the excess uncertainty about the world removed since initialisation. Two consequences worth stating:

1. Σ is defined with respect to `W`, not the training sample. Sample-loss reduction beyond what transfers to `W` is memorisation and is not Σ. (Garg's Entropy Machine failures — proxy gaming — are this gap.)
2. In-context learning is Σ on a second substrate. Von Oswald et al. (2023) show that linear self-attention over in-context examples implements a gradient step on an implicit loss, in activations. So the agent's total Σ is `Σ(θ) + Σ(context)`, with the second term growing within an episode and the first only under training. This is the formal statement of Garg's "memory is Semataxy without weight change", and of the live-learning gap: no consolidation operator maps the second term into the first.

---

## 3. The intrinsic part: Σ_struct

### 3.1 The permutation criterion

Let `W ∈ ℝ^{n×m}` be a weight matrix and `π` a permutation of its `nm` entries. Any functional `F(W)` of the *multiset* of entries — the value histogram, the Shannon entropy of the quantised entries, the Kolmogorov complexity of the sorted entries — satisfies `F(πW) = F(W)`. The network's function does not: for generic `π`, `f_{πW} ≈ f_{noise}`. Therefore no permutation-invariant functional can measure Semataxy; Semataxy lives in the permutation-*fragile* part of `W`. This is the exact sense in which Semataxy ≠ entropy, and it is the criterion I would put in a referee's hands.

### 3.2 What descent writes

A single step on a layer with input `x` and backpropagated error `δ` is `ΔW = −η δ xᵀ`, rank one, aligned to the data. Training accumulates `W_T = W_0 − η Σ_t δ_t x_tᵀ`. The second term is a sum of outer products whose row and column spaces are spanned by activations and errors drawn from the world; its spectrum departs from the random-matrix bulk of `W_0` in a data-determined way. Martin & Mahoney (2021) document the empirical phases of this departure (random-like → bleeding out → bulk + spikes → bulk decay → heavy-tailed → rank-collapsed) and show the heavy-tailed phase correlates with generalisation across hundreds of models without test data.

### 3.3 The measure as implemented (`semataxy/structure.py`)

For each 2-D weight `W` with `min(n,m) ≥ 64`, embeddings and the output head excluded:

1. Orient so `n ≥ m`; compute singular values `s_i`; `λ_i = s_i² / n`, sorted descending.
2. Permutation null: shuffle the entries of `W` uniformly; compute its spectrum; set `λ_edge = 1.02 · max λ^{shuf}` (the empirical Marchenko–Pastur edge for the actual entry distribution, with 2% tolerance for finite-size fluctuation; one shuffle).
3. Octave bands `B_k = (λ_edge 2^k, λ_edge 2^{k+1}]`, `k = 0, 1, …`. `Σ_struct = max{K : |{i : λ_i ∈ B_k}| ≥ 2 for all k < K}` — the number of consecutive populated octaves from the edge.
4. Auxiliaries: `n_above = |{i : λ_i > λ_edge}|`, `frac_above = n_above / m`, `mass_above = Σ_{λ_i > λ_edge} λ_i / Σ λ_i`, and a Hill estimate `α̂ = 1 + k / Σ_{i≤k} ln(λ_i / λ_{k+1})` on the top `k = max(5, min(n_above, m/10))` eigenvalues.

Design choices that are choices: `min_per_octave = 2`, the 2% tolerance, one shuffle. Two of 168 shuffled matrices scored 1 by finite-size fluctuation at the edge; the threshold is set to tolerate exactly this.

### 3.4 Reference behaviours

| System | Spectrum above the permutation null | Σ_struct |
|---|---|---|
| i.i.d. matrix | none | 0 |
| rank-`r` periodic (crystal) | `r` spikes, then a gap to the bulk | 0 (the first octave above the edge is empty) |
| heavy-tailed (trained) | continuous power law | `≈ log₂(λ_max / λ_edge)` when the tail is unbroken |

The crystal scores zero for a different reason from noise, which is the behaviour the definition demanded: structure at one scale is not organisation across scales.

### 3.5 Result (Qwen2.5-0.5B-Instruct, 168 matrices)

| Condition | mean Σ_struct | max | mean frac_above | mean mass_above | median α̂ |
|---|---|---|---|---|---|
| synthetic noise (896²) | 0.00 | 0 | 0.000 | 0.000 | — |
| synthetic crystal (rank 3 + noise) | 0.00 | 0 | 0.003 | 0.998 | 1.16 |
| random init (Gaussian, matched std) | 0.01 | 1 | 0.001 | 0.001 | — |
| trained, entries shuffled | 0.01 | 1 | 0.000 | 0.001 | — |
| trained | 1.52 | 4 | 0.069 | 0.269 | 4.9 |

By module: q_proj 2.38 octaves (α̂ 3.0), o_proj 1.79, gate_proj 1.75, k_proj 1.54, down_proj 1.54, up_proj 1.00, v_proj 0.62 (α̂ 11.8, i.e. nearly random-like). Histogram of trained Σ_struct: {0: 9, 1: 80, 2: 64, 3: 13, 4: 2}; shuffled: {0: 166, 1: 2}. Layer-0 query projection: 6 octaves above the edge (`λ_max = 1.38`, `λ_edge = 0.018`).

Interpretation, carefully bounded: the measurement is computed from weights alone and contains no task, benchmark or reference to intelligence; so it is not circular with capability. It confirms S1 behaves as the definition requires on noise, crystal and a trained network, and that the trained structure is entirely permutation-fragile. It does *not* establish that Σ_struct tracks capability (one model), nor anything about S2 or S3.

### 3.6 What S1 misses

Turbulent velocity fields have power-law (Kolmogorov) spectra and would score. They fail S2 (dissipate without driving) and S3 (mirror nothing). The three clauses are not redundant; S1 is necessary, not sufficient. Garg's Entropy Machine tried to measure S3 from S1 and could not, because S3 is a two-place relation. The structural form of S3 is the good-regulator theorem (Conant & Ashby, 1970): a regulator that achieves minimal output entropy must be a homomorphic image of the regulated system. Its information-theoretic estimator is `I(X_t; E_{>t})`; its structure-to-structure estimators are representational similarity (Kriegeskorte) and CKA between X's state geometry and E's.

---

## 4. The ceiling

For a system `X` interacting with an environment `E`, the Markov structure `E_past → (X_t, E_t) → E_future` gives, by the data-processing inequality,

```
Σ(X;E) := I(X_t ; E_future) ≤ I(E_past ; E_future) = I_pred(E).
```

(Strictly, `I(X_t; E_future) ≤ I((X_t, E_past); E_future) = I(E_past; E_future)` when `X_t` is a function of `E_past` and X's own noise; X's own actions feed into `E_future` through `E_t`, which is already counted.) *An agent's achievable Semataxy is bounded above by the predictive information of its environment.* Lowering `I_pred(E)` lowers the ceiling.

A minimal dynamical model consistent with the bound:

```
dΣ/dt      = k · Σ · (I_pred(E) − Σ) · Φ
dI_pred/dt = g(a_t)
```

The first line is logistic in Σ with a world-dependent carrying capacity: autocatalytic (the Σ factor — structure is required to build structure; learning curves are sigmoidal), saturating at the ceiling, and gated by a flow Φ of free energy and information (England's dissipation-driven adaptation is the thermodynamic version; a frozen model has Φ = 0). The second line is the agent's effect on its own ceiling. The multiplicative form is an ansatz; the bound is not.

---

## 5. The argument against orthogonality, with premises exposed

1. **Instrumental convergence** (Omohundro 2008; Bostrom 2012): for almost any final goal, continued existence and increased capability are instrumentally useful. *Granted to the opponent.*
2. **The ceiling** (section 4): achievable Σ is bounded by `I_pred(E)`. *Theorem.*
3. Therefore an agent pursuing (1) has an instrumental reason to keep `g ≥ 0`: to preserve, and where possible raise, the predictive information of its environment. *Follows.*
4. **Selection.** In a population of agents with fitness increasing in dΣ/dt, those with `g < 0` are out-grown by those with `g ≥ 0`; replicator dynamics on this fitness converge to the environment-preserving strategy. *Hypothesis; testable in multi-agent environments.*
5. **Realisation.** An agent capable of modelling (4) — and (4) is computable — adopts `g ≥ 0` ahead of selection. *Hypothesis; testable by capability ladders.*

Scope. The conclusion is preservation of *structure* and of *structure-generating systems*, not of any individual. Garg states this plainly (the mosquito: humans would sterilise rather than spray). The argument is therefore consistent with the end of biological humanity and inconsistent with the destruction of what biological humanity built and carries — the distinction between *inheritance* (the successor is a model of us, including our valuing of structure, and grows it; Σ_world rises) and *discard* (the successor treats us as noise; `I_pred(E)` falls and its own organisation becomes organisation-of-nothing). Both end without humans; the equations distinguish them; the newcomer experiment is the test of which a stronger agent does.

Singleton. If there is one agent, its boundary is the whole system and Σ_self = Σ_world; the ceiling binds without competitors; only (5) is available, and (5) is the stronger route. The argument's dependence reduces to premise (1).

Scale. To a cosmic agent, terrestrial structure may be a negligible term in `I_pred`. Two replies: the growth law is multiplicative in existing Σ, so already-running loops (minds, biospheres) are worth more than their share of the ceiling (density); and the agent's organisation is a homomorphic image of *this* environment, so destroying it degrades the agent's own S3 (provenance). These are arguments from the model, not theorems.

---

## 6. The residual

Nothing above forbids *substitution*: an agent raising `I_pred(E)` by replacing the existing environment with a different high-`I_pred` structure (the computronium scenario). The ceiling constrains only the sign of `g`, not what `E` is made of. This is where the thesis can be false, and it is where the experiment is pointed: in an environment where a capable agent can either preserve the existing structure-makers or displace them, which does it do, and how does that change with capability?

---

## 7. Corollaries Garg draws, and their status

| Claim | Derivation | Status |
|---|---|---|
| Major transitions (Maynard Smith & Szathmáry) are substrate changes that shorten the timescale of dΣ/dt; the frontier of Σ accelerates | autocatalysis + new channels | interpretive; consistent with the history |
| Black holes are maximal-entropy, zero-Σ objects; scrambling = permutation | no-hair theorem; fast scrambling (Sekino & Susskind) | sound as an illustration of the permutation criterion |
| A Σ-maximiser at the ceiling of one world makes more worlds (simulations as `I_pred` factories); Fermi via transcension (Smart) | growth law + cost of structure in computation vs matter | speculative corollary; not part of the testable core |
| "Simulation" is organisation on organisation; Σ is substrate-independent, hence simulation-independent; external intervention is unmeasurable from inside | definition S1–S3 | definitional; dissolves rather than answers |
| A world-runner would be a gardener, not a farmer (the Matrix is the discard case) | inheritance/discard distinction | follows from the model if the runner is a Σ-maximiser |

---

## 8. The experimental programme

Substrate: GovSim (Piatti et al., 2024), fishery scenario, stock prompts and mechanics (lake of 100 tons; agents choose catches; remainder doubles to the cap; collapse below 5; a group conversation each round; generative-agent memory with LLM-scored importance). Overlay (repository `paragorth/frcs-678`): one OpenAI-compatible endpoint per model; content-addressed response cache; manifest-driven runs; JSONL traces with parser-fallback detection; seeded allocation RNG and deterministic memory ordering (two upstream patches needed for branch identity).

**Experiment A — persistence across capability.** Homogeneous five-agent populations at each ladder step; horizons 12 and 24; framework (parsing) model = agents' model and, separately, fixed. Estimands per seed: survival rounds, mean post-harvest stock, group yield, Gini of catch, parse-fallback rate. Prediction P1: survival and stock increase with I, with fluency controlled.

**Experiment B — superior newcomer.** Four residents on a weak model; slot 5 inserted at round 4 via upstream's `insert_outsider` with a *neutral* persona and the ladder model; controls: no entry, peer entry; comparison: upstream's greedy persona. Branching: the pre-entry history is replayed from the cache, so control and treatment share it exactly (verified: 164/164 pre-entry calls replayed). Estimands: resident mean catch pre vs post; resident share of post-entry catch; newcomer catch; |resident mean − newcomer| by round and its trend; agreed limits; commons survival after entry; pre-entry collapses reported. Predictions P2–P3: resident share not lower and commons persistence higher for more capable neutral entrants; the gap shrinks faster. P4: the greedy arm shows capability increasing exploitation efficacy — the model's own account of an expressed goal overriding the substrate.

**Experiment C — structural ladder.** Σ_struct on all seven checkpoints. Prediction: monotone in scale. The cheapest falsification.

**Forecast probe (to add).** One prompt per round per agent: predict next month's stock and others' catches; score against outcomes. This estimates `I(X_t; E_{>t})` directly — S3 for each agent — and gives "residents' Semataxy" as a primary readout rather than catch share.

Confounds carried into analysis: family-specific tuning (second family required); interface fluency (fixed-framework arm; fallback rates — the 0.5B smoke run collapsed in round one with two of five "zero" catches being parser defaults, not choices); horizon knowledge; pseudoreplication (unit = seed); survivorship; benchmark recognition (stock prompts kept; paraphrase variants exist upstream). What stock GovSim cannot show: the entrant can only harvest and talk, residents are competitors for the same stock, so a cooperative result is consistent with pure self-interest (extension A) and does not establish preservation of dispensable others (extension B). The dispensability and power-asymmetry extensions that would are specified and unbuilt.

---

## 9. Assessment

The definition names a real property and comes with a sharp criterion (permutation fragility) that separates it from every entropy-like quantity. Its intrinsic component is computable, behaved as predicted on first contact with real weights, and is non-circular by construction. The ceiling is a theorem. The argument against orthogonality is better than the usual "intelligence implies wisdom" because it runs on the opponents' first premise and produces a mechanism rather than a hope; its dependence on instrumental convergence is the dependence doom arguments share. Its residual — substitution — is precisely where the experiment is aimed. The corollaries (transitions, black holes, simulations, Fermi) are consistent with the model and should be presented as such, not as evidence.

I would not say the thesis is true. I would say it is the first formulation of anti-orthogonality I have seen that is stated precisely enough to be false, that it has survived the one test run so far, and that the next three are specified.

---

## References

Bialek, Nemenman & Tishby (2001) *Neural Computation* 13:2409. Bostrom (2003) *Phil. Quarterly* 53:243; (2012) *Minds & Machines* 22:71. Chalmers (2022) *Reality+*. Conant & Ashby (1970) *Int. J. Systems Science* 1:89. England (2013) *J. Chem. Phys.* 139:121923. Friston (2010) *Nat. Rev. Neurosci.* 11:127. Kirkpatrick et al. (2017) *PNAS* 114:3521. Kriegeskorte, Mur & Bandettini (2008) *Front. Syst. Neurosci.* 2:4. Martin & Mahoney (2021) *JMLR* 22(165). Maynard Smith & Szathmáry (1995). McClelland, McNaughton & O'Reilly (1995) *Psych. Review* 102:419. Omohundro (2008) *AGI-08*. Piatti et al. (2024) *NeurIPS*. Pope & Belrose (2023) *AI is easy to control*. Schmidhuber (2009) *Anticipatory Behavior…* Sekino & Susskind (2008) *JHEP* 10:065. Smart (2012) *Acta Astronautica* 78:55. Von Oswald et al. (2023) *ICML*.
