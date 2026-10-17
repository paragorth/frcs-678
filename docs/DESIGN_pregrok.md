# Design: pre-grokking. Does organisation in the weights move before behaviour does?

Status: written 17 October 2026 before the sweep ran. **Result (same day, `docs/updates/2026-10-17-pregrok-sweep.md`): the preregistered tests are not met; the onset rule fires in 0.77 of pairs in the no-decay controls, so the measures are not specific to generalisation at level 1. Level 2, the representation (amendment 2) and the control-calibrated rule (amendment 3) all ran the same day: no rule family separates the controls; the claim is not supported and not refuted by this sweep. See the update note.** This is the programme's own claim (T2 note in
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

## The paper this answers to (added 17 Oct 2026)

Beckmann and Queloz, *Mechanistic Indicators of Understanding in Large Language Models*
(arXiv:2507.08017v5; Philosophical Studies 2026; copy in `docs/papers/`). A synthesis paper with no
code or measure of its own. Its third tier, principled understanding, is "when a model ceases to
rely on memorized facts and discovers a compact circuit connecting these facts", and its evidence is
Nanda et al.'s grokking study, which it summarises as three phases: memorisation, **circuit
formation**, cleanup, with the jump in test accuracy coming "in the cleanup phase, after the
generalizing mechanism was learned". That ordering, established by ablation with knowledge of the
task, is the premise of this design. What the sweep adds: whether the formation of the circuit can
be read from the weights alone, with no ablation and no knowledge of what the task is, through the
permutation-null measures, before the behavioural jump. If so, it is a task-agnostic indicator for
their tier-three transition, which they do not have, and the live-learning gate's signal.

## Prior art, and what it changes (added 17 Oct 2026, from `docs/papers/pregrok_prior_art_2026-10-17.md`)

Nobody has a task-agnostic, weight-only, null-calibrated spectral measure that leads the jump and
predicts its timing across seeds. But the pieces exist and set the bar:
- **Weight norm already leads.** In this exact setup Nanda et al. report the total L2 norm falling
  from about epoch 1,400 while test loss stays flat to about 9,400; a near-constant critical norm
  predicts the grok time by intervention (Truong et al. 2026). So any weight-only measure must be
  shown to carry information *beyond* the norm.
- **Spectral measures on weights have been reported as coincident or lagging**: HTSR tail exponent
  and shuffled-matrix outliers (Prakash and Martin; our closest methodological kin) coincide with
  the jump; effective-rank collapse coincides (Yunis et al. 2024) or lags (2607.06639).
- **Leading indicators with timing laws exist only from activations or gradients**: spectral entropy
  of penultimate activations (lead about 1,000 steps, 10 seeds), circuit synchronisation (about
  1,700), commutator defect of gradients (600 to 1,600, power-law lead over 43 runs).

Amendments to this design, before the sweep is read:
1. **Norm is a comparator, logged and analysed like the rest.** Per-matrix Frobenius norm and the
   total norm get the same onset rule; every spectral lead is reported alongside the norm's lead,
   and the spectral onset is also measured *relative to the norm onset*. The claim is only made if
   concentration or the tail exponent moves before, or independently of, the norm.
2. **Effective rank** (entropy of the normalised singular values, Yunis et al.) is computed from the
   checkpoints as the published weight-only comparator.
3. **The no-weight-decay control is the decisive one**: there the norm does not fall; if our
   measures still fire there we are measuring norm; if they fire only where grokking occurs and
   lead it, we are not.
4. The honest claim, if it holds: *the first weight-only spectral measure, calibrated against a
   permutation null, that leads the generalisation jump with a cross-seed timing fit*, not "the
   first early indicator".
5. The strongest reviewer objection is quoted in the prior-art file and will be answered with data
   or conceded.

## Amendment 2, preregistered 17 Oct 2026 (afternoon), before running: organisation of the representation

Why: the level-1 result shows a weight-only permutation null cannot tell "organised about the task"
from "organised at all" (it fires in the no-decay controls). Semataxy is organisation *relative to the
environment*, so the environment must do the pointing. `tools/pregrok_acts.py` passes a fixed random
sample of 2,048 task inputs (seed 12345, the same set for every checkpoint) through each saved
checkpoint and applies the same shuffle-null measures to the activation matrices at the prediction
position: attention output, residual after attention, MLP hidden (post-ReLU), MLP output, residual
before unembedding, logits. Comparators: effective rank and Frobenius norm of each activation
matrix, and a task-aware upper bound (Gini of the DFT, over answer classes, of the mean logit
vector per class; this one knows the task and should lead, as Nanda's does).

Rules: identical to the level-1 rules above (plateau 2,000 to 10,000, detrend, CUSUM > 8, ten logs
beyond three spreads; the logs are every 500 epochs here, so "ten logs" is 5,000 epochs; B1 and B2
unchanged). Runs: the same 21 (the three fast-decay runs that generalise inside the plateau window
remain unanalysable and are reported as such).

Predictions. Thesis: mass above the edge or concentration in the MLP hidden or residual-post
activations leads B1 in at least 7 of the 8 clean-plateau grok runs and in at least 10 of the 13
runs with a B1, with onsets in the no-decay controls in fewer than 1 in 5 (activation, measure)
pairs. Against: the same false-positive rate as the weights (then the representation also keeps
reorganising without generalising, and the whole family of spectral measures is a description), or
onsets that coincide with B1 rather than lead it. The task-aware comparator is expected to lead in
every grok run; a Semataxy measure that leads *as often as* the task-aware one, without knowing the
task, is the positive result.

## Amendment 3, preregistered 17 Oct 2026 (14:00), before running: a control-calibrated onset rule

Finding that forces it: under the fixed rule (CUSUM > 8, |z| > 3 for ten logs) the known positive,
Nanda's Fourier measure, fails to lead and fires in 3 of 3 no-decay controls. The rule has no power.

Rule: for each measure family (weights level 1; circuit level 2; representation) and each of the
candidate settings on a fixed grid (CUSUM threshold ∈ {8, 16, 32, 64, 128, 256}; |z| threshold
∈ {3, 4, 6, 8, 12, 16}; persistence k ∈ {10, 20} logs for the 100-epoch logs, {4, 8} for the
500-epoch logs), compute the false-positive rate as the share of (matrix, measure) pairs firing in
the three no-decay controls. **Choose, before looking at any grok run, the most sensitive setting
(smallest thresholds, in grid order) whose control false-positive rate is ≤ 0.20.** Apply that one
setting once to the grok runs. Plateau and B1/B2 unchanged (2,000 to 10,000; the three fast-decay
runs stay unanalysable). Same statistics as before: leads positive in ≥ 12 of 15 (here of the 13
runs with a B1: ≥ 11), sign test, Spearman of structural onset against B1, the eight
clean-plateau runs reported separately.

Power check: the Fourier comparator under the same calibrated setting. If it does not lead in at
least 10 of 13, the sweep cannot answer the question at this resolution and no claim either way is
made about the Semataxy measures from it.

Positive result: a Semataxy measure (mass above, concentration, tail exponent, span) in some
matrix or path or activation that leads in ≥ 11 of 13 with the calibrated rule, while the norm and
effective-rank comparators in that matrix do not, and the Fourier comparator does. Anything less is
reported as "not found at this resolution".
