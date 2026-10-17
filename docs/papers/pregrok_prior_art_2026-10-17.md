# Prior art for the pre-grokking claim (research agent, 17 Oct 2026)

I have what I need; no further fetches are required. Below is the deliverable.

## (a) Prior-work table

| Paper (year) | What is measured | Computed from | Lead before test-acc jump | Code |
|---|---|---|---|---|
| Nanda et al. 2023, [2301.05217](https://arxiv.org/abs/2301.05217) | Restricted loss, excluded loss (Fourier-space logit ablations); sum of squared weights; Gini of Fourier-basis norms of W_E, W_L; trig-component norms | Ablation + train data (restricted/excluded); **weights only** (L2 norm, Gini, Fourier norms — but "key frequencies" must first be found by mech-interp) | Phases: memorisation 0–1.4k, circuit formation 1.4k–9.4k, cleanup 9.4k–14k epochs. Excluded loss rises and L2 norm "decreases smoothly" from ~1.4k while test loss is flat until ~9.4k, i.e. ~8k epochs lead. Gini rises "sharply during cleanup" (coincident). Authors state measures are "specific to … one particular algorithmic task" | [neelnanda.io/grokking-paper](https://neelnanda.io/grokking-paper) → [github](https://github.com/mechanistic-interpretability-grokking/progress-measures-paper) |
| Varma et al. 2023, [2309.02390](https://arxiv.org/abs/2309.02390) | Circuit efficiency (logits per parameter norm); ungrokking, semi-grokking (~83% acc near D_crit) | Logits on data + parameter norm | No timing prediction; explicitly a limitation | none |
| Liu et al. 2022, [2205.10343](https://arxiv.org/abs/2205.10343) | Representation Quality Index (RQI): fraction of parallelogram relations satisfied by embeddings; phase diagram (comprehension/grokking/memorisation/confusion) | Embeddings, but uses the group's addition structure (label-informed) | RQI>0.95 used *as* the grokking time; theory predicts t∼1/λ3; no weight-only lead stated | [github.com/ejmichaud/grokking-squared](https://github.com/ejmichaud/grokking-squared) |
| Liu, Michaud, Tegmark 2023 Omnigrok, [2210.01117](https://arxiv.org/abs/2210.01117) | Weight norm ‖w‖₂ vs reduced loss ("LU mechanism", Goldilocks zone w≈w_c) | Weights only (norm); losses need data | Grokking happens when decay brings ‖w‖ to w_c; init scale controls delay; no quoted lead | [github.com/KindXiaoming/Omnigrok](https://github.com/KindXiaoming/Omnigrok) |
| Barak et al. 2022, [2207.08799](https://arxiv.org/abs/2207.08799) | Fourier gap of population gradient; weight movement ρ=‖w_t−w_0‖∞ | ρ: weights only; Fourier gap: gradients | Argues for progress measures "functions of … the model weights w_t … predictive of the time to successful convergence"; ρ grows throughout plateau (parities, not grokking) | none |
| Thilak et al. 2022 Slingshot, [2206.04817](https://arxiv.org/abs/2206.04817) | Last-layer weight norm: cyclic growth/plateau with loss spikes | Weights only (norm) + train loss | Norm growth starts after 100% train acc; features change at phase switches; no timing law | none |
| Kumar et al. 2024, [2310.06110](https://arxiv.org/abs/2310.06110) | NTK–task alignment, laziness, weight norm | Kernel on data + labels | Mechanistic, no lead quoted | — |
| Lee et al. 2024 Grokfast, [2405.20233](https://arxiv.org/abs/2405.20233) | Slow/fast gradient components (EMA filter) | Gradients | Accelerates ×50, doesn't predict | [github.com/ironjr/grokfast](https://github.com/ironjr/grokfast) |
| Humayun et al. 2024, [2402.15555](https://arxiv.org/abs/2402.15555) | Local complexity (density of linear regions) | Activations on input samples (no labels) | Descent-ascent-descent; LC drop accompanies grokking | bit.ly/grok-adversarial |
| Prieto et al. 2025, [2501.04697](https://arxiv.org/abs/2501.04697) | Softmax collapse; NLM direction; weight norm | Gradients/logits | Shows decreasing weight norm "is not necessary for grokking" | [github.com/LucasPrietoAl](https://github.com/LucasPrietoAl/) |
| Notsawo et al. 2023, [2306.13253](https://arxiv.org/abs/2306.13253) | Low-frequency Fourier amplitude of early train-loss curve | Loss curve | Predicts *whether* grokking will occur, not when | — |
| Yunis et al. 2024, [2408.11804](https://arxiv.org/abs/2408.11804) | Singular values, effective rank, singular-vector alignment | **Weights only** | "sudden drop in validation loss coincides precisely with the onset of low-rank behavior" — coincident, not leading | [github.com/dyunis/spectral_dynamics](https://github.com/dyunis/spectral_dynamics) |
| Prakash & Martin 2025/26 (HTSR), [2506.04434](https://arxiv.org/abs/2506.04434), [2602.02859](https://arxiv.org/abs/2602.02859), [2605.12394](https://arxiv.org/abs/2605.12394) | WeightWatcher α (power-law tail of ESD); Correlation Traps = eigenvalues of entry-shuffled W beyond MP edge + Tracy-Widom | **Weights only** | α: 4.0±0.6 pre-grok → 2.9±0.2 at grok → 1.1±0.3 anti-grok; drop "coincides with" the accuracy rise. Traps appear only at anti-grokking ("effectively zero" before). Mostly seed 0 | [WeightWatcher](https://github.com/CalculatedContent/WeightWatcher) |
| Xu 2026, [2602.16746](https://arxiv.org/abs/2602.16746); [2602.16967](https://arxiv.org/abs/2602.16967); [2603.15678](https://arxiv.org/abs/2603.15678) | Commutator defect ‖θ_AB−θ_BA‖/(‖ηg_A‖‖ηg_B‖); PCA of weight trajectory; "spectral edge" of update SVD | Gradients on two minibatches (needs data); PCA of parameter updates needs no labels | Onset 600–1,600 steps before grok (12 runs, p=2⁻¹²); lead ∝ grok_step^1.27 (R²=0.97, n=43); necessary not sufficient | [github.com/skydancerosel/grokking-integrability](https://github.com/skydancerosel/grokking-integrability) |
| Truong et al. 2026, [2604.13123](https://arxiv.org/abs/2604.13123) | Normalised spectral entropy of penultimate-activation covariance; threshold H̃*≈0.609 | Activations on data (no labels) | Collapse leads by mean 1,020 steps (CI 890–1,140; 10 seeds); ΔT=C₁(H̃−H̃*)^1.65+C₂, R²=0.54, LOO median error 3.2% | [github.com/clevix/grokking-entropy](https://github.com/clevix/grokking-entropy) |
| Truong et al. 2026, [2606.13753](https://arxiv.org/abs/2606.13753) | Critical total weight norm W_c (CV 1–2% across seeds); T_grok ∝ exp(7.5ρ), R²=0.996 | **Weights only** (norm), via clamping intervention | Grokking "on the descent as the norm approaches W_c"; no free-running lead quoted | [github.com/ClevixLab/critical-norm-grokking](https://github.com/ClevixLab/critical-norm-grokking) |
| [2606.12966](https://arxiv.org/abs/2606.12966) Circuit synchronisation (FSD) | Fraction of MLP neurons sharing a dominant Fourier frequency | Activations grouped by input sum (uses inputs, not labels) | 500–3,000 steps lead, mean +1,722, 9/9 configs; beats restricted loss | "upon publication" |
| [2604.20923](https://arxiv.org/abs/2604.20923) ILDR | Inter/intra-class distance ratio of penultimate reps | Activations + **labels** | 950±250 steps lead, 8 seeds | — |
| [2607.06639](https://arxiv.org/abs/2607.06639) | Audit: embedding effective rank | Weights | Rank compression *lags* accuracy by ≥10⁴ steps | — |
| [2605.20441](https://arxiv.org/abs/2605.20441) | Attention-head cosine similarity, entropy spread; α tracked too | Attention patterns on data | AUC 0.80 predicting regime, below their 0.85 bar | [github.com/lucky-verma/grokking-diagnostics](https://github.com/lucky-verma/grokking-diagnostics) |

## (b) Novelty verdict

No published work shows a **task-agnostic, weight-only, null-calibrated** spectral measure that both leads the jump and forecasts its time across seeds. But the pieces exist:

- Weight-only measures that lead: Nanda's total L2 norm already falls from ~1.4k epochs while test loss is flat to ~9.4k (same setup as yours). Weight norm reaching a near-constant critical value W_c (2606.13753, CV 1–2%) is effectively a weight-only timing predictor, by intervention rather than forecast.
- Weight-only spectral measures: HTSR α and shuffled-matrix MP outliers (Prakash & Martin) — your closest methodological kin (permutation null, bulk edge) — but they report α as coincident and traps as a *post*-grokking (anti-grokking) signal, and mostly one seed. Yunis et al.: rank collapse coincident; 2607.06639: lags.
- Leading indicators with timing laws exist only from activations/gradients: spectral entropy (1,020-step lead, LOO 3.2%), FSD (1,722), ILDR (950), commutator defect (600–1,600, power-law lead).

Your difference: edge-relative *mass*, *fraction*, concentration, Hill tail and octave span of each matrix, no data, no ablation, logged across seeds with a timing fit. That gap is real but narrow; state it as "first weight-only spectral measure with a leading, cross-seed timing fit," not "first early indicator."

## (c) Measures to compute on your checkpoints

1. Total L2 norm Σw² (Nanda Fig. 7) and norm-to-W_c descent (2606.13753: ‖W‖=(‖E‖²_F+Σ‖W_i‖²_F)^½, W_c = value at test-acc 0.9).
2. Nanda Gini of Fourier-basis norms of W_E and W_L (DFT over the p token rows; Gini of the per-frequency norms), and norms of key-frequency components.
3. Effective rank exp(−Σp_i log p_i), p_i=σ_i/Σσ (Yunis eq. 1–2), per matrix.
4. WeightWatcher α (power-law fit to ESD tail) and Correlation-Trap count (entry-shuffle W, eigenvalues of WᵀW/N beyond λ₊^MP+Δ_TW) — run `weightwatcher` v0.7.5.5 directly.
5. Spectral entropy H̃ of the penultimate covariance (2604.13123) as the activation-side comparator, since it has the best published timing law.
6. Restricted/excluded loss via Nanda's repo as the ablation baseline (FSD paper shows restricted loss can *lag* at p=113).

## (d) Strongest reviewer criticism

"Weight norm — a weight-only scalar — already leads the jump by ~8k epochs in exactly this setup (Nanda 2023), and a near-constant critical norm already predicts timing (2606.13753). Your edge-relative spectral statistics must be shown to carry information *beyond* ‖W‖ (partial out norm; test under norm-clamping and under no-weight-decay grokking per Prieto 2025, where norm doesn't fall). Also: the spectral-entropy, FSD and commutator-defect papers already give leads of ~1–2k steps with quantified prediction error on 8–15 seeds; HTSR α on the same shuffled-null idea is reported as coincident, so you must explain why your edge statistics lead when α does not; and 2607.06639 shows weight-space compression can lag by ≥10⁴ steps, so the direction of your claim is contested unless you report lead with CIs across seeds and primes."

Sources: all arXiv links above; Nanda code https://github.com/mechanistic-interpretability-grokking/progress-measures-paper; WeightWatcher https://github.com/CalculatedContent/WeightWatcher.
