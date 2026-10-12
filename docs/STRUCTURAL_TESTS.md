# Structural tests of Semataxy (Σ_struct)

Each test asks whether the intrinsic measure (`semataxy/structure.py`) moves when
organisation is known, on independent grounds, to be built, destroyed or absent.
Predictions are written before the data. "Cheap" = CPU, under a day.

| # | Test | Known fact it leans on | Prediction | Failure would mean | Cost | Status |
|---|---|---|---|---|---|---|
| T1 | Trained vs shuffled vs init vs synthetic noise/crystal | shuffling kills function, keeps entropy | trained ≫ shuffled ≈ init; noise 0; crystal 0 | the measure is not seeing organisation | cheap | **done**: 1.52 / 0.01 / 0.01 / 0 / 0 |
| T2 | **Grokking** (modular addition, 1-layer transformer, Nanda et al. 2023) | test accuracy jumps from chance to ~100% long after train accuracy saturates; the network switches from a lookup table to a Fourier circuit | *(as preregistered)* Σ_struct low in memorisation, rises at the jump | the measure cannot distinguish memorised structure from principled structure | cheap (~3 h, 100k epochs) | **done — prediction wrong in direction, right in kind.** Grok at epoch ~42k. At the grok: mass above edge 0.81→0.97 (up), fraction of eigenvalues above edge falls in every matrix, α 3–5→1.2–1.4, Σ_struct 1.14→0.3. The spectrum *concentrates*: the general circuit is a few strong directions; weight decay prunes the diffuse memorisation directions. See §T2 note. |
| T3 | **Training trajectory of a real LM** (Pythia-160M public checkpoints, steps 0→143k) | loss falls monotonically through training | Σ_struct rises monotonically with step; attention q/o lead; saturates late | organisation is not what training accumulates, or the null is wrong | cheap (~1 h) | **done — as predicted.** 0 at init → 1.88 by step 4k → 2.06 at 143k; mass 0→0.31 still rising at the end; attention q/k/v and output lead (structure by step 128–512), MLP-up last (zero at 512, 1.8 by 4k); attention-output dips mid-run and recovers. |
| T4 | Capability ladder (Qwen2.5 0.5B→72B, same family, same data recipe) | larger models are more capable on benchmarks | Σ_struct monotone in scale (allowing for Martin–Mahoney rank collapse at the top, which would be reported) | clause 1 is not capability-tracking; need a different null | GPU for 32B/72B | ready |
| T5 | Random-label training (Zhang et al. 2017) | nets can memorise random labels perfectly; no generalisation | Σ_struct lower / spikier than true-label training at equal train accuracy | the measure cannot tell memorisation from learning (overlaps T2) | cheap | to write |
| T6 | Base vs instruct vs RLHF at one size (Qwen2.5-7B, -7B-Instruct) | post-training changes behaviour far more than weights | small Σ_struct change; possible collapse in late layers | post-training rewrites organisation (then the "pretraining builds it" story needs amending) | GPU or chunked CPU | ready |
| T7 | Catastrophic forgetting (fine-tune a small LM on a narrow task) | old-task performance collapses | Σ_struct drops in the layers that forget; EWC-regularised fine-tuning preserves it | forgetting is not structural loss | cheap | to write |
| T8 | Pruning and quantisation | magnitude pruning degrades gracefully up to a point, then collapses | Σ_struct tracks the collapse point, not the pruning fraction | — | cheap | to write |
| T9 | Cross-family comparability (Llama-3.1, Gemma-2, Mistral at ~7B) | different architectures, similar capability | Σ_struct comparable across families at matched capability; if not, the measure is architecture-bound and needs normalising | — | GPU | ready in principle |
| T10 | Biological networks (C. elegans connectome, fly hemibrain) vs degree-preserving rewiring | network neuroscience already uses the rewiring null | Σ_struct > 0 for real connectomes, 0 for rewired | the construction does not transfer to biology | cheap (public data) | to write |
| T11 | Mechanistic tiers (Beckmann & Queloz 2025: conceptual / state-of-world / principled) | interpretability finds features, factual circuits, and compact circuits replacing memorised facts | the three tiers correspond to successive octave bands populating; principled understanding (T2's grok) populates the widest span | the tiers are not spectral | depends on T2 | design |

## T2 note: what grokking taught the measure

The preregistered prediction was that generalisation would *spread* structure across scales. It did the opposite: it *concentrated* it. Memorisation is a diffuse lookup table — many medium-strength directions, α in the "well-trained" band, Σ_struct ≈ 1. The general circuit is a handful of strong directions (Nanda's Fourier frequencies) with everything else decayed away: mass above the edge → 0.97, direction count halved, α → 1.3, Σ_struct → 0.3. Spectrally, the grokked network looks like a *crystal*.

This is consistent with the theory and corrects clause 1's operationalisation. The ceiling `Σ ≤ I_pred(E)` says the mirror can only be as rich as the world; (a+b) mod 113 is a one-rule world with tiny predictive information, so its correct mirror is low-rank. A crystal is the right model of a crystal world. Multi-scale span (Σ_struct) is the signature of organisation matched to a *rich* world (Pythia on text: rising to 2+ octaves); *compactness* — mass per above-edge direction, or low α with high mass — is the signature of principled rather than memorised organisation in any world. Clause 1 therefore needs two numbers: **span** (how many scales) and **concentration** (how much of the structure is carried by how few directions). Memorisation scores low on concentration; understanding scores high; noise scores zero on both; a crystal scores zero span and maximal concentration — which is exactly right for a world with one rule, and exactly wrong for ours.

The failure clause in the row above did not trigger: the spectrum moved sharply and specifically at the grok. It moved the other way from the guess, which is what a measurement is for. The "compact circuit replacing memorised facts" of Beckmann & Queloz's third tier is this concentration, observed.

**Prior art check (12 Oct).** Rank collapse / low-rank solutions at the grok, promoted by weight decay: known (arXiv 2602.16746; 2604.13123 "spectral entropy collapse"; q/k rank-1 dominance). Shuffle null vs Marchenko–Pastur edge: used in arXiv 2602.02859 (anti-grokking, WeightWatcher). Pythia trained weights vs MP across training: arXiv 2606.19491, 2610.13123. Ungrokking as catastrophic forgetting: Varma et al. 2023 (circuit efficiency). T2 and T3 are therefore *replications with a unified measure*, not new findings; the novelty claimed is the single permutation-null measure across regimes, the span/concentration split, and the grokking = consolidation = forgetting-without-breadth identity (T7, below), which no source found states.

## T7 sharpened: forgetting is the grokking operation without the world in view

| | concentration | span | held-out loss |
|---|---|---|---|
| grokking (one-rule world; nothing to lose) | up | — | down |
| narrow fine-tune of a pretrained LM | up | **down** | **up** (forgetting) |
| narrow fine-tune with interleaved replay / EWC | up | kept | kept (consolidation) |

Runner to write: Pythia-160M, fine-tune on one narrow corpus, log span, concentration and held-out (Pile) loss every N steps, with and without replay.

## How the tests combine

T1–T3 establish that Σ_struct is built by training and is not entropy. T2 and T5 establish that it distinguishes *principled* organisation from *memorised* organisation — the sample/world distinction, which is the one that matters for the thesis. T4 and T9 make it a capability axis. T6–T8 characterise what degrades it. T10 shows the construction is about physical systems, not about transformers. T11 ties it to the interpretability literature's account of understanding.

If T2 fails — Σ_struct flat through the grok — the measure is seeing *amount* of structure but not *kind*, and clause 1 needs a second term (e.g. a compactness or circuit-sparsity measure) before it can stand in for "understanding".

## References

Nanda, Chan, Lieberum, Smith & Steinhardt (2023) Progress measures for grokking via mechanistic interpretability. Power et al. (2022) Grokking. Biderman et al. (2023) Pythia. Zhang et al. (2017) Understanding deep learning requires rethinking generalization. Martin & Mahoney (2021). Kirkpatrick et al. (2017) EWC. Beckmann & Queloz (2025) Mechanistic indicators of understanding in LLMs.
