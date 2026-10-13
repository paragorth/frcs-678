# Update, 13 October 2026: Experiment C, the structure ladder (first six rungs)

**What happened.** Σ_struct was measured on the Qwen2.5 instruct models from 0.5B to 32B on CSD3,
streaming one weight matrix at a time (job 38230611 for 0.5B–14B, 38239602 for 32B). The 0.5B rung
reproduces the laptop result from T1 exactly (1.52, mass 0.268). A size control was added at Parag's
suggestion: the same matrix shapes filled with untrained random weights (job 38242616, 0.5B–7B so far).

**What it means for the thesis.** Supports, on the main measure. The organisation score rises at every
step of the ladder, and the random-weight control sits at zero at every size tested, so the rise is not
an artefact of bigger matrices. The secondary measure (share of variance above the noise edge) was
preregistered to rise too; it is flat-to-falling from 0.5B to 14B and then jumps at 32B. Call the
secondary prediction unclear until 72B is in. The ladder still confounds size with capability; the
same-size comparison (three Qwen generations at 7B, base vs instruct, Mistral, Llama, Gemma) is running
and is the test that separates them.

**The numbers** (means over all 2-D weight matrices except embeddings and the output head).

| Model | Σ_struct mean | Σ_struct max | mass above noise | random-init Σ_struct |
|---|---|---|---|---|
| Qwen2.5 0.5B | 1.52 | 4 | 0.268 | 0.01 |
| Qwen2.5 1.5B | 1.72 | 4 | 0.247 | 0.00 |
| Qwen2.5 3B | 1.85 | 4 | 0.257 | 0.00 |
| Qwen2.5 7B | 1.96 | 5 | 0.237 | 0.00 |
| Qwen2.5 14B | 2.16 | 6 | 0.231 | running |
| Qwen2.5 32B | 2.18 | 6 | 0.320 | running |
| Qwen2.5 72B | running | | | running |

Figure: `docs/figures/structure_ladder.png`. Per-matrix CSVs: `results/structure/`. Summary:
`results/structure/ladder_summary.csv`. Parameters as in T1: `min_per_octave=2`, 2% edge tolerance,
one shuffle; matrices with a side under 64 or over 20,000 skipped.

**Caveats, stated plainly.**
- Σ_struct mean is a mean over matrices, and larger models have more matrices per layer type in the
  same proportions, so the mean is comparable across sizes; the max rises from 4 to 6 octaves.
- 14B to 32B is a small step on the main measure (2.16 to 2.18). The 72B rung decides whether the
  curve is saturating.
- The 32B jump in the secondary measure comes with a lower tail exponent (α 3.98 vs 4.5–4.9), i.e. a
  heavier spectral tail. Whether that is "more organised" or "differently organised" is open.

**What's next.** 72B (running, 12 h cap), the control at 14B–72B, then the same-size comparison.
Experiments A (four sizes, five seeds) and B (newcomer pilot) are queued on the cluster with the new
forecast probe. Nothing needed from Parag.

First-use fixes this session, each in a commit message: bfloat16 safetensors read; sharded GGUF
download; resumable setup; pathfinder gen() with no stop regex returns ""; probe sets
`persona.current_time`; SL3 QOS caps jobs at 12 h.
