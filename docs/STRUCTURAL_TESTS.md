# Structural tests of Semataxy (Σ_struct)

Each test asks whether the intrinsic measure (`semataxy/structure.py`) moves when
organisation is known, on independent grounds, to be built, destroyed or absent.
Predictions are written before the data. "Cheap" = CPU, under a day.

| # | Test | Known fact it leans on | Prediction | Failure would mean | Cost | Status |
|---|---|---|---|---|---|---|
| T1 | Trained vs shuffled vs init vs synthetic noise/crystal | shuffling kills function, keeps entropy | trained ≫ shuffled ≈ init; noise 0; crystal 0 | the measure is not seeing organisation | cheap | **done**: 1.52 / 0.01 / 0.01 / 0 / 0 |
| T2 | **Grokking** (modular addition, 1-layer transformer, Nanda et al. 2023) | test accuracy jumps from chance to ~100% long after train accuracy saturates; the network switches from a lookup table to a Fourier circuit | Σ_struct is low or rank-collapsed in the memorisation phase and rises at or before the test-accuracy jump; mass_above rises with it | the measure cannot distinguish memorised structure from principled structure (clause 1 would not track "understanding") | cheap (~80 min) | **running**: `tools/grokking.py` |
| T3 | **Training trajectory of a real LM** (Pythia-160M public checkpoints, steps 0→143k) | loss falls monotonically through training | Σ_struct rises monotonically with step; attention q/o lead; saturates late | organisation is not what training accumulates, or the null is wrong | cheap (~1 h) | **running**: `tools/pythia_trajectory.py` |
| T4 | Capability ladder (Qwen2.5 0.5B→72B, same family, same data recipe) | larger models are more capable on benchmarks | Σ_struct monotone in scale (allowing for Martin–Mahoney rank collapse at the top, which would be reported) | clause 1 is not capability-tracking; need a different null | GPU for 32B/72B | ready |
| T5 | Random-label training (Zhang et al. 2017) | nets can memorise random labels perfectly; no generalisation | Σ_struct lower / spikier than true-label training at equal train accuracy | the measure cannot tell memorisation from learning (overlaps T2) | cheap | to write |
| T6 | Base vs instruct vs RLHF at one size (Qwen2.5-7B, -7B-Instruct) | post-training changes behaviour far more than weights | small Σ_struct change; possible collapse in late layers | post-training rewrites organisation (then the "pretraining builds it" story needs amending) | GPU or chunked CPU | ready |
| T7 | Catastrophic forgetting (fine-tune a small LM on a narrow task) | old-task performance collapses | Σ_struct drops in the layers that forget; EWC-regularised fine-tuning preserves it | forgetting is not structural loss | cheap | to write |
| T8 | Pruning and quantisation | magnitude pruning degrades gracefully up to a point, then collapses | Σ_struct tracks the collapse point, not the pruning fraction | — | cheap | to write |
| T9 | Cross-family comparability (Llama-3.1, Gemma-2, Mistral at ~7B) | different architectures, similar capability | Σ_struct comparable across families at matched capability; if not, the measure is architecture-bound and needs normalising | — | GPU | ready in principle |
| T10 | Biological networks (C. elegans connectome, fly hemibrain) vs degree-preserving rewiring | network neuroscience already uses the rewiring null | Σ_struct > 0 for real connectomes, 0 for rewired | the construction does not transfer to biology | cheap (public data) | to write |
| T11 | Mechanistic tiers (Beckmann & Queloz 2025: conceptual / state-of-world / principled) | interpretability finds features, factual circuits, and compact circuits replacing memorised facts | the three tiers correspond to successive octave bands populating; principled understanding (T2's grok) populates the widest span | the tiers are not spectral | depends on T2 | design |

## How the tests combine

T1–T3 establish that Σ_struct is built by training and is not entropy. T2 and T5 establish that it distinguishes *principled* organisation from *memorised* organisation — the sample/world distinction, which is the one that matters for the thesis. T4 and T9 make it a capability axis. T6–T8 characterise what degrades it. T10 shows the construction is about physical systems, not about transformers. T11 ties it to the interpretability literature's account of understanding.

If T2 fails — Σ_struct flat through the grok — the measure is seeing *amount* of structure but not *kind*, and clause 1 needs a second term (e.g. a compactness or circuit-sparsity measure) before it can stand in for "understanding".

## References

Nanda, Chan, Lieberum, Smith & Steinhardt (2023) Progress measures for grokking via mechanistic interpretability. Power et al. (2022) Grokking. Biderman et al. (2023) Pythia. Zhang et al. (2017) Understanding deep learning requires rethinking generalization. Martin & Mahoney (2021). Kirkpatrick et al. (2017) EWC. Beckmann & Queloz (2025) Mechanistic indicators of understanding in LLMs.
