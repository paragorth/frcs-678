# Update, 13 October 2026 (evening): Experiment C complete, with the same-size test and a shape check

**What happened.** The full Qwen2.5 ladder (0.5B to 72B) is measured; the random-init size control is
measured at 0.5B to 32B (72B still running); and the same-size comparison is in: three Qwen generations
at 7B, Qwen2.5 7B base vs instruct, and Mistral 7B, Llama 3.1 8B, Gemma 2 9B. A shape check
(`tools/structure_shape_check.py`) asks how much of Σ_struct is explained by matrix dimensions alone.

**What it means for the thesis.** Mixed, and the bad news comes first.

- *Against, for Σ_struct as a measure of capability.* At fixed size, Σ_struct does not track
  intelligence. Qwen 1.5 (the oldest, weakest 7B) scores highest of the seven; the three Qwen
  generations go 2.26, 1.93, 1.96; base and instruct are identical (1.95 vs 1.96). Qwen2 and Qwen2.5
  at 7B have identical architecture and a clear capability gap, and 91% of their 196 matrices get
  exactly the same octave count. Across every trained model, Σ_struct correlates 0.47 with the log of
  the smaller matrix dimension, about +0.3 octaves per doubling. After adjusting for that, the ladder
  is flat from 0.5B to 14B; only 32B and 72B sit above what their shapes predict (+0.11, +0.26).
  So most of the ladder's rise was matrix size, not organisation. The random-init control could not
  have caught this: it shows trained ≠ random at every size, which is true, but says nothing about
  whether the *slope* is meaningful.
- *Supports, weakly, on the secondary measure.* Mass above the noise edge is not shape-driven (flat
  against dimension) and does move with capability where we can see it: Qwen 1.5 → 2 → 2.5 at 7B goes
  0.201 → 0.228 → 0.237, and the ladder's two largest models jump to 0.32 and 0.34. Three points in one
  family is not much. Base vs instruct does not move it at all (0.237 → 0.237).
- *Still true.* Trained wiring is organised and random wiring is not, at every size and family: the
  permutation criterion separates them cleanly. That was T1; it now holds on thirteen models.

The honest summary: the preregistered prediction "Σ_struct rises monotonically 0.5B → 72B" is
technically met and substantively hollow, because the measure's octave count grows with matrix width.
The HANDOVER already listed "a finer-grained Semataxy measure" as the open problem; this is the
cluster-scale confirmation that it is the *blocking* problem for the structural half of the programme.

**The numbers.**

| Model | params | Σ_struct | shape-adjusted Σ | mass above noise |
|---|---|---|---|---|
| Qwen2.5 0.5B | 0.5B | 1.52 | +0.10 | 0.268 |
| Qwen2.5 1.5B | 1.5B | 1.72 | +0.05 | 0.247 |
| Qwen2.5 3B | 3B | 1.85 | +0.08 | 0.257 |
| Qwen2.5 7B | 7B | 1.96 | −0.06 | 0.237 |
| Qwen2.5 14B | 14B | 2.16 | −0.05 | 0.231 |
| Qwen2.5 32B | 32B | 2.18 | +0.11 | 0.320 |
| Qwen2.5 72B | 72B | 2.42 | +0.26 | 0.344 |
| Qwen 1.5 7B chat | 7B | 2.26 | −0.06 | 0.201 |
| Qwen 2 7B instruct | 7B | 1.93 | −0.09 | 0.228 |
| Qwen2.5 7B base | 7B | 1.95 | −0.07 | 0.237 |
| Mistral 7B instruct | 7B | 2.04 | −0.10 | 0.243 |
| Llama 3.1 8B instruct | 8B | 2.11 | −0.04 | 0.234 |
| Gemma 2 9B it | 9B | 2.09 | −0.11 | 0.216 |
| random init, any size | — | 0.00 | — | 0.000 |

Shape-adjusted Σ = Σ_struct minus a linear fit on log2(smaller dimension) pooled over all trained
models; 0 means "what its matrix sizes alone predict". Figure `docs/figures/structure_ladder.png`;
per-matrix CSVs and `ladder_summary.csv` in `results/structure/`.

**What's next, and what I'd ask Parag.**
1. The measurement needs fixing before more structural ladders are worth running. Two candidates,
   both cheap on CPU: (a) normalise octaves by the octaves *available* to a matrix of that shape (the
   `octaves_total` column is already recorded); (b) weight by mass, i.e. treat mass-above-edge as
   primary and Σ_struct as a span diagnostic. I will run (a) on the existing CSVs tonight; it needs no
   new compute. If neither rescues a capability signal at fixed size, that is the finding.
2. The behavioural half is unaffected and is the stronger test anyway: Experiments A and B are queued
   on the cclake partition (moved from the busy icelake queue) with the forecast probe. Their outcome
   does not depend on Σ_struct.
3. Decision for Parag: the pub-evening talking point should be the T1 separation (trained vs random,
   thirteen models) and the live-learning loop, not "organisation rises with scale". I would not show
   the ladder curve without the shape caveat beside it.
