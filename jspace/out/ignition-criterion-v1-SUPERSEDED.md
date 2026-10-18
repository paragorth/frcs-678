# Ignition criterion — what would falsify what

**Status of the ignition question: OPEN, NOT TESTED.**

Written before further ignition work, because the last three "results" were each
withdrawn for the same reason: a measurement was reported as evidence about the
model when it was actually evidence about the instrument. This document fixes,
in advance, what the null predicts, what the alternative predicts that the null
cannot, and the effect size required — so the next run can fail.

---

## 1. The null (H0) — and why it is not yet rejected

**H0:** report is a deterministic threshold on a continuous quantity (the
concept margin) that varies smoothly with α, plus prompt-level variation in that
quantity. No bistability, no self-amplification, no threshold in the underlying
process.

H0 predicts: report onset spread across prompts equals the spread predicted by
applying one global margin criterion to each prompt's own margin curve.

Measured (7B, coherent window α ≤ 8, n=60): **spread ratio 1.12**, global
criterion predicts report at 83.3% vs 69.6% baseline.

**This confirms the null. It does not test the alternative.** See §3.

## 2. The alternative (H1) — ignition

**H1:** entry into the workspace is a bistable, self-amplifying transition.
Once a concept crosses a critical level, recurrent amplification drives it to a
high-occupancy state; below it, the concept decays.

Predictions H0 **cannot** produce:

| # | H1 prediction | H0 prediction | testable here? |
|---|---|---|---|
| A | **Hysteresis** in α across a feedback path: onset differs on a rising vs falling ramp | exactly zero — a memoryless threshold has no history | **yes, but only through generation** (§4) |
| B | **Bimodal margin** across prompts at fixed α near the transition | unimodal, sliding | yes, weak power |
| C | **Superlinear** dm/dα peaking at the transition | dm/dα ≈ constant | yes |
| D | Onset **residual clustering** after conditioning on each prompt's own margin | residual is pure readout noise | yes |

## 3. Power — what the completed test could and could not have seen

Computed from the actual data (n=60, grid step 1.0, window [0,8]):

- **Onset-spread ratio**: SE of log-ratio = 0.130 → 2σ band **[0.77, 1.30]**.
  The test could only have rejected onsets **≥23% tighter** than thresholding.
  Observed 1.12. **It had no power to detect anything subtler than 23%.**
  Whether real ignition would tighten onsets by ≥23% is *unknown* — that number
  was never derived from a model of ignition, it is just what n=60 buys.
- **Grid quantisation floor**: step 1.0 contributes ~0.29 to onset SD. A
  genuinely stepped population with true onset SD below 0.29 would be
  **indistinguishable from the grid**. Observed SD 1.48, so this is not
  currently binding — but it caps any future sharpness claim at step/√12.
- **Dynamic range**: inside the coherent window the margin moves −19.9 → −3.3
  and the whole report transition occupies **~25% of the usable α range**.
  This is the binding constraint on every test below.

**Conclusion: "consistent with pure thresholding" is a statement about a test
with 23% resolution, not a falsification of ignition.**

## 4. The discriminating test, and an honest problem with it

Hysteresis (A) is the only *qualitative* discriminator: H0 predicts exactly
zero, H1 predicts a loop. But **a stateless forward pass cannot show
hysteresis** — α is re-applied fresh each pass, no history carries over, so up-
and down-sweeps are identical by construction. Testing A on independent forward
passes is guaranteed to return zero and would prove nothing.

The only genuine feedback path in this setup is **autoregressive generation**:
injected content enters the context and conditions later tokens. So:

> **Ramp test.** During generation with injection active, ramp α upward across
> token positions, and separately downward, over the same range. Compare the
> token position at which report begins. H0: identical. H1: the down-ramp
> sustains report below the up-ramp's onset — a loop.

Power: loop width detectable at 2σ is **≥0.54 α units at n=60**, **≥0.30 at
n=200** — 7% and 4% of the coherent window respectively.

**Falsification condition, fixed now:** if the up/down onset difference is
< 0.30 α units at n ≥ 200, with the coherence gate passing at every α on both
ramps, then self-amplification of the size the workspace account requires is
excluded in this model. If the difference exceeds that and survives the
random-direction control, that is positive evidence for ignition.

## 5. Whether a falsifiable version exists at all

Being honest about the constraint stack: the coherent window is ~8 α units wide,
the transition occupies ~25% of it, and the only feedback path is generation,
which also degrades fastest under injection. The ramp test may not fit inside
the window — degeneracy could arrive before the loop is resolvable.

**If the coherence gate cannot be held across a ramp wide enough to resolve
0.30 α units, then ignition is not falsifiable in this setup, and that is the
result** — reported as such, not as a null. The next run must record the
coherence gate at every α of both ramps, and abort rather than report if it
fails.

## 6. Standing corrections

- **Stage 4's null has one of two defects resolved, not the null strengthened.**
  The lens is now validated. The *window* is not: Stage 4 swept [−6, 24] on the
  1.5B with **no repeat-fraction gate**. The 7B degenerates by α≈8; the 1.5B
  ceiling is likely lower. That sweep may sit largely over degeneracy — the same
  failure as the withdrawn Job 2 headline. **Untested until the 1.5B coherence
  ceiling is measured.**
- **The 1.5B gate pass is WEAK**: swapped 3/4, versus 4/4 on both larger models
  — and 3/4 became the passing threshold *after* the criterion was revised.
- Every ignition claim in this repository is withdrawn. Status: **OPEN**.
