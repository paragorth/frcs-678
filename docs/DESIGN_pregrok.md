# Design: pre-grokking. Does organisation in the weights move before behaviour does?

Status: written 17 October 2026 before the sweep ran. This is the programme's own claim (T2 note in
`docs/STRUCTURAL_TESTS.md`): at the memorise-to-generalise transition the weight spectrum
*concentrates*. The question here is sharper and is the one that matters for both goals: is the
structural change visible **before** any behavioural sign of generalisation, and does its timing
**predict** when generalisation arrives? If yes, organisation measured from the weights is a leading
indicator of understanding, and the live-learning gate can be built on it. If no, structure and
behaviour move together and the measure is a description, not a predictor.

## What the one existing run says (seed 0, weight decay 1, logged every 200 epochs)

Test accuracy leaves chance at about epoch 27,600 (2 per cent), reaches half at 41,000. Averaged
over matrices, concentration and mass leave their memorisation plateau at 30,000 to 33,000: after the
first behavioural movement, well before the main jump. Per matrix, the value matrix's span and
concentration and the MLP-up matrix's tail exponent leave their plateau at 12,800 to 14,200, long
before anything behavioural. That could be a real lead or slow drift under weight decay tripping an
ad hoc threshold. One seed, one shuffle draw, a 3-standard-deviation rule chosen after looking: not
evidence. Hence this design.

## Runs

| arm | seeds | weight decay | train fraction | purpose |
|---|---|---|---|---|
| grok | 0-4 | 0.3, 1.0, 3.0 | 0.3 | the transition at three speeds; 15 runs |
| no-grok control | 0-2 | 0.0 | 0.3 | memorises and (within 80k epochs) never generalises: structural onsets here are false positives |
| fast control | 0-2 | 1.0 | 0.6 | generalises almost at once: no separated plateau, no lead to find |

Task and model as before: (a + b) mod 113, one-layer transformer (Nanda et al. 2023), 80,000
epochs. Logged every 100 epochs: train and test loss and accuracy; per matrix (embed, Q, K, V, O,
up, down, unembed): span, mass above the shuffled edge, fraction of directions above, concentration
(mass per above-edge direction), tail exponent, top 16 singular values, Frobenius norm; the shuffled
null averaged over 4 draws. Checkpoints every 500 epochs, so circuit-level measures (the OV and
MLP products, the embedding-to-unembedding path) can be computed offline.

## Onset rules, fixed now

- **Behavioural onset** (two definitions, both reported): B1 = first epoch at which the 5-log rolling
  median of test loss falls below 0.8 times its plateau median and stays there; B2 = first epoch at
  which test accuracy exceeds 5 per cent and stays there. Plateau = epochs 2,000 to 10,000 (train
  accuracy is 1.0 throughout this window in every arm).
- **Structural onset** per measure and matrix: fit a straight line to the measure over the plateau
  (this absorbs slow drift under weight decay); residuals thereafter are standardised by the
  plateau's residual spread; onset = first epoch at which the CUSUM of standardised residuals
  exceeds 8 and the measure stays beyond 3 spreads for 10 consecutive logs. The same rule is applied
  to the control arms, where any onset is a false positive.
- **Lead** = behavioural onset minus structural onset, in epochs, per run, per measure.

## Predictions, written before the sweep

Thesis (structure leads): for concentration and tail exponent in at least one matrix, lead is
positive in at least 12 of 15 grok runs under B1 (sign test p < 0.05), and the structural onset
epoch correlates with the behavioural onset epoch across the 15 runs (Spearman > 0.7), i.e. it is a
predictor, not a fixed offset. The no-grok controls produce onsets in fewer than 1 in 5
measure-matrix pairs (false-positive rate), and the fast controls produce no separable lead.

Against: leads scattered around zero or negative; or positive leads with no correlation across runs
(the measure moves at a fixed time under weight decay regardless of when the grok comes); or
control false-positive rates as high as the grok arms' hit rates.

What would be ambiguous: a lead only on the mean over matrices with no single matrix leading, or a
lead that exists at weight decay 3 and not at 0.3 (then it is a weight-decay signature, not a
generalisation one).

## What this design cannot not produce

Weight decay shrinks everything monotonically, so any threshold on a shrinking quantity fires
eventually; the plateau detrend and the no-grok control exist to catch exactly that. Test loss is
noisy during memorisation (it oscillates by a factor of two); the rolling median and the "stays
there" clause exist for that. Five seeds is the minimum for a sign test to mean anything.

## Cost and tools

21 half-node CPU jobs of 2 to 4 hours. `tools/grokking.py` (logging, checkpoints),
`cambridge/grokking_sweep.sbatch`, analysis `tools/pregrok_analysis.py` (the rules above, nothing
tuned after seeing the sweep), output `results/grokking/`.
