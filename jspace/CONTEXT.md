# Project context

Orientation for anyone (human or model) picking this up. Written to the repo
because shared state lives here, not in a context window.

## The thesis
Fine-tuning operates at the wrong level of abstraction. Catastrophic forgetting
is unsolved because every mitigation acts on parameters and is blind to meaning.
The missing piece is not acquisition — it is **persistence**. The inference
already happens in the forward pass and is then discarded. What is needed is a
**write path from activation to weight**.

Everything in this repo is instrumentation for that.

## Why the J-space lens
Anthropic's global-workspace paper (Transformer Circuits, July 2026) supplies a
coordinate system: a corpus-averaged Jacobian from a middle-layer activation to
vocabulary logits, isolating a low-dimensional subspace that drives what the
model reports (~10% of activation variance; a concept's component in it is 6-7%
of that concept's variance but drives report on 59% of trials vs 5% for the
rest).

It is used here as a **target space** — not because workspace theory is the
point, but because it is the only measurable, manipulable handle on "the meaning
the model is currently holding".

## Phase 3 — the load-bearing experiment
Capture the workspace state with a skill in context. Then reach that same state
**through weights**, with the file removed, while penalising or projecting away
change in orthogonal directions.

    L = L_target + lambda_orth * L_orth + lambda_size * ||delta_theta||^2

Three mechanisms to compare:
1. soft penalty
2. hard gradient projection (GEM/OGD lineage)
3. verify-before-commit gate

Against ordinary fine-tuning at **matched weight drift**. The finding is a
**slope across magnitudes**, not a point. **No LoRA** — low-rank adaptation is
itself a forgetting mitigation and confounds the measurement.

**If this works it is a paper. Nothing else in the repo is.**

## Ignition — a side quest, parked
Dehaene & Naccache's commentary flagged that the workspace evidence lacks an
ignition signature. Parag's own deployment model of consciousness independently
predicts a prediction-error threshold. Same structural claim, arrived at
separately — so the lens offers a cheap test of his prediction inside someone
else's model. Interesting, cheap, but advances the write path not at all.

**Parked.** Do not resume unless the write path is blocked.

## Established
- Lens validated on 7B and 14B: 4/4 baseline, swapped, moved; cross-check vs
  full J at 0.9999.
- 1.5B: **WEAK** pass — 3/4 swapped vs 4/4 on both larger models, and 3/4 became
  the threshold after the fact.
- Skill corpus: 10/14 kept, 5 with >=95% gap.
- Injection: 1.7% -> 86.7% over alpha 0-8 on the 7B, inside the coherent window,
  random control never fires.

## Retracted — and this matters more than what is established
- Job 2 headline ("100%, 60/60") — scored degenerate repetition.
- Job 4 ("confinement loses to fine-tuning") — asymmetric optimisation effort,
  unmatched install success, n=1. **NOT TESTED**, not falsified.
- Three ignition findings — each a monotonic signal plus a threshold, which
  produces a step by construction.

## The recurring failure mode
Every retraction is the same error in different clothes: **measuring something
that had to come out that way, or measuring outside the regime where the model
is still coherent.** Three findings died to an un-gated alpha window alone.

Standing consequences:
- **Every sweep reports its coherent alpha ceiling** — by repeat fraction,
  derived from that sweep's own prompt pool — before any analysis runs.
- **Confirming a null is not testing an alternative.** No power analysis means
  the question is open, not answered.
- **Lens validity and window validity are independent defects.** Fixing one does
  not strengthen a result that suffers from the other.

## Working practice
Before reporting any headline: open a fresh context, hand it the results with no
framing, and ask it to find the reason the headline is wrong. Every significant
correction so far came from data already in hand that had not been interrogated.
The gap is stance, not capability.

## Phase numbering — a correction
**Phase E is NOT part of this protocol's numbering.** It belongs to a separate
paper: live learning via a certified external skill library with a frozen LLM.
The `verify_gate` arm borrows its certify-before-commit logic — a real lineage,
but not a position in this sequence. **Do not infer a Phase D.**

`protocol.md` is stale and is being rewritten. **Do not reconstruct the phase map
from it.** Phases 3 and 5 as described above are current.
