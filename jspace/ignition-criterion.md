# IGNITION-CRITERION — pre-registered

Written before any further ignition measurement. **Committed before running
anything. If the result contradicts it, the result wins.**

Supersedes `out/ignition-criterion.md` (v1) and `out/ignition-stephold-prereg.md`
(the step-and-hold design). Both retained as records.

## Why this document exists
Three ignition findings have died here, all the same way: a monotonic signal plus
any threshold produces a step, and every convenient readout — argmax, rank,
binarised report — is itself a threshold. The measurement manufactured the shape
being hunted; null and alternative made identical predictions.

**Nothing further gets measured until there is a statistic on which they do not.**

## H0 — thresholded monotonic readout
Injecting alpha along a concept direction raises that concept's internal
representation smoothly and monotonically. Any apparent all-or-none behaviour is
produced entirely by the readout.

H0 predicts:
- the continuous internal quantity is smooth and monotone in alpha, no region of
  accelerating gain
- per-prompt report onsets are exactly predicted by each prompt's own continuous
  curve crossing a single fixed criterion
- the distribution of the continuous quantity at fixed alpha is unimodal
- every layer tracks alpha with the same smooth form; only the final argmax steps

## H1 — ignition
Past a threshold the network adds energy: recurrent or cross-layer amplification
makes the state self-reinforcing rather than input-driven.

H1 predicts, and H0 cannot produce:
- a region of gain above unity — downstream amplitude rising faster than drive
- bimodality of the continuous amplitude at intermediate alpha, before thresholding
- onsets sharper than each prompt's own criterion-crossing predicts
- layer dissociation: early layers linear, workspace layers nonlinear

## The measurement fix
1. **Inject at layer L, measure at layer L+k**, k large enough that the network's
   computation intervenes.
2. **Continuous quantity only.** No argmax, rank or report indicator anywhere in
   the primary statistic.
3. **Readout direction derived independently of the injection direction.** If
   readout and injection are the same vector the test is void.
4. **Normalise by injected drive.** The primary statistic is a gain.

## Primary discriminator — gain
`G(alpha) = d(downstream concept amplitude) / d(injected drive)`, both in matched
units, alpha=0 recorded.

H0: G <= 1 everywhere, non-increasing once saturation begins.
H1: a region where G > 1.

**Criterion.** Ignition is supported if the per-prompt max of G exceeds the
**95th percentile of the empirical null**, estimated from the random-matched
direction control through the identical pipeline — not assumed.

**Falsification.** If max G does not exceed the control's 95th percentile,
ignition is not supported at this scale. Report that as the result.

### One correction to the framing, recorded now
The absolute value "1" is **not** the operative threshold. Layer L and layer L+k
have different natural coordinate scales, so a raw ratio is dominated by that
scale difference, and a passive linear map can exceed 1 in normalised units. The
operative criterion is the **control-relative** one above, which is unit-robust
and is what this document already specifies. "Gain above unity" is retained as
the intuition, not the test.

## Secondary — bimodality
Hartigan's dip on the raw per-prompt continuous amplitudes at the alpha nearest
median report onset. No pooling before analysis. Criterion: dip p < 0.05 with the
control failing to reject. **Low power at achievable n; a null here is
uninformative unless the power analysis says otherwise.**

## Tertiary — onsets sharper than self-predicted
Per prompt, take its own continuous curve, find where it crosses a fixed
criterion, use that as H0's predicted onset. Criterion: variance of
(observed - predicted) significantly below the control's.

## Power analysis — MANDATORY, results written in before the run
See `## ACHIEVED POWER` below. If primary power < 0.8, either raise n or record
in advance that the result is inconclusive rather than negative.

## Standing gates
- **Coherent alpha ceiling first**, by repeat fraction from this sweep's own
  prompt pool -- **not entropy, not KL**, which flag distribution *shift* and so
  penalise the intervention for working. Nothing above it is analysed. Three
  findings died to this, all of them correctly (log 30).
  **The 0.5 repeat threshold is not calibrated and is too permissive on short
  samples.** Every ceiling must be supported by inspected text at the boundary,
  with samples recorded. Three instruments gave alpha 4, 9 and 13 on the same
  7B data; only the inspected number (9) is defensible.

- **The usable window is narrow, and report never saturates inside it.**
  Measured on the 7B (log 30), strict-scored, generations inspected:

  ```
  strict report 50% crossing   alpha 5.31
  coherent ceiling (inspected) alpha ~9      strict report there: 83.3%
  degeneration begins          alpha 10      'spider spider spider spider'
  strict report reaches 100%   alpha 12      -- inside wreckage, unusable
  ```

  **Report and coherence overlap, but only just: width ~3.7 alpha units, and
  the concept never becomes fully reportable while the model is still writing
  English.** This is a hard constraint on any ignition test. A discriminator
  that needs a saturated report, or that needs both halves of a symmetric ramp
  populated, does not fit in this window -- which is the same conclusion the
  ramp tests reached independently on all three models (log 20i, 20k). Any
  ignition design must either work inside [5.31, 9] with report topping out
  around 83%, or justify a different injection paradigm.
- **Model: 7B** (lens validation 4/4 on all three conditions). Not the 1.5B,
  whose pass is 3/4 with a threshold set after the fact.
- Per-prompt trajectories. Variance from prompts, not seeds. Record alpha=0.
- **Probe measure, not margin** (margin reached report on 35/250 in Stage 4; the
  probe fires at 86.7% inside the coherent window).
- Controls: random direction and orthogonal direction, both at matched norm,
  both through the identical pipeline.
- Alignment: import the shared token-suffix function. Four recurrences.

## What would make me abandon this
Primary test inside the control distribution with adequate power => ignition is
not present in this model at this scale by this measure. Record it, stop, and do
not re-run with a different statistic until a new prediction motivates one.

Dehaene and Naccache reported no ignition signature in the original paper with
considerably more infrastructure. **A negative here is the expected outcome, not
a surprise, and is worth exactly one clean measurement rather than a campaign.**

## Scope limits
- Injecting into chosen directions and reading out is a **projection onto those
  directions**, not the paper's matching-pursuit occupancy measure. Do not call
  it occupancy.
- A transformer has no recurrence within a forward pass. **Cross-layer
  amplification is the only mechanism available**, so a negative bears on
  ignition-in-feedforward-inference and says nothing about recurrent systems.
- One model, one concept family, one lens. Generalisation untested.

## ACHIEVED POWER — written before the run

Measured inputs, from the existing dense sweep (n=60, alpha <= 8, coherent window):
- per-prompt residual SD around a smooth fit: **0.8156**
- median rise across the window: **16.5352**
- rise / noise: **20.3**

### The primary statistic changed, and the power analysis is why
The gain-ratio statistic as first written was **broken**: normalising each
curve's max local gain by its own mean gain explodes on the near-flat control
(mean gain ~ 0), so the control's 95th percentile is unbounded and nothing can
exceed it. Simulated power came out **0.00 for H1 AND for H0** — the signature of
a broken statistic, not a hard problem. This is the same failure as the ramp
test's step-index over-detecting on flat noisy curves.

**Replacement primary statistic: CONVEXITY.** H0 says the internal quantity
saturates or is linear — concave everywhere, or flat. A region of accelerating
gain is a region of **positive second derivative**, which neither a saturating
nor a linear curve has. Second differences are scaled by the measurement noise,
so concept and control are comparable despite ~20x different amplitudes, and
nothing is divided by a quantity that can vanish.

### Simulated power, H1 with gain peaking at 1.3

| n prompts | power | false-positive rate under H0 |
|---|---|---|
| 60 | 0.58 | 0.09 |
| 120 | 0.67 | 0.04 |
| **250** | **0.81** | **0.01** |
| 500 | 0.93 | 0.02 |

**Planned n = 250**, the smallest count clearing the 0.8 requirement, at a 1%
false-positive rate. n=60 is explicitly insufficient (0.58) — the existing dense
sweep could not have detected this effect, which is worth stating when the
earlier null is described.

### Secondary (Hartigan dip) — declared underpowered in advance
At any achievable n the dip detects bimodal separation only above ~12% of the
rise. **A null on the dip is uninformative** and will be reported as such, per
the caveat already in this document.

### Noise caveat
The variance is estimated from Stage-5 per-prompt **margin** curves, the only
per-prompt continuous trajectories measured so far. The new primary statistic is
a concept **amplitude at a downstream layer** — similar in kind, not identical.
These power numbers are indicative and are labelled so.
