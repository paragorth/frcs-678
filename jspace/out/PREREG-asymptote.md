# CANCELLED — see log 22c

The question this pre-registered ('can SGD drive cosine toward the direction
tolerance') is void: a cosine of exactly 1.0 — a perfect copy of the whole
activation tensor at every layer — installs 0%. There is no tolerance to reach.
Retained unedited below as a record of what was frozen and why it lapsed.

---

# Pre-registration — the asymptote run

Written **before** the curve is seen, and before the three cheap measurements
report. Frozen. If any of it needs changing afterwards, the change is recorded as
a deviation, not an edit.

## Question
Can gradient descent drive activation-trajectory cosine (band L9-18, token-aligned,
at corrected norm) from the 0.928 reached in 80 steps up to the direction
tolerance measured by measurement 1?

## Design
One arm (`soft_penalty`, lambda_orth = 0 so it is pure `L_target` descent), **no
drift cap**, 2000 steps, LR chosen by short search for fastest cosine ascent.
Report cosine every 50 steps, plus norm ratio and `||delta_theta||`.

## Pre-committed decision rule
Let **T** be the direction tolerance from measurement 1 — the cosine at which
install falls to half its corrected-norm peak.

- **ABANDON state-matching as a practical write path** if cosine plateaus below
  **T** while `||delta_theta||` exceeds **2x** the drift at which plain
  fine-tuning reaches 75% install (measured: 0.083). Plateau = < 0.002 cosine
  gain over 300 consecutive steps.
- **CONTINUE** if cosine reaches T at drift <= 2x, whether or not install
  follows. Install then becomes the next question, not the blocker.
- **INCONCLUSIVE** if it neither plateaus nor reaches T within 2000 steps; report
  the curve and the extrapolated step count, do not guess.

## Recorded in advance
- The optimiser may reach high cosine by exploiting the 33-position, 12-instance
  target rather than by acquiring the skill. **Held-out cosine is therefore
  reported alongside train cosine at every checkpoint**, and a train/held-out gap
  above 0.02 counts as memorisation, not matching.
- Norm ratio is reported throughout. A run that reaches cosine T while norm ratio
  drifts outside 0.95-1.05 has not met the criterion, since install needs both.
- This run tests **reachability of the target**, not install. Install is measured
  but is not the decision variable — measurement 1 already fixes what install
  should be at a given cosine.
