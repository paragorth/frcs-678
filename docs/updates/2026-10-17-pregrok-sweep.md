# 17 Oct 2026: pre-grokking sweep read under the preregistered rules. The claim fails.

Sweep: 21 runs (`docs/DESIGN_pregrok.md`), run on one A100 in two one-hour slices (all 21 in
parallel; the second slice resumed each run from its epoch-55,000 checkpoint, without optimiser state
because those checkpoints predate the fix; marked in every log). Logs `results/grokking/*_log.jsonl`,
analysis `tools/pregrok_analysis.py` → `results/grokking/pregrok_level1_summary.csv`. Rules as
preregistered: plateau 2,000 to 10,000, line detrend, CUSUM > 8 and ten logs beyond three spreads;
B1 = test loss below 0.8 × plateau median and stays; B2 = test accuracy above 5 per cent and stays.

## Behaviour

| arm | runs | grokked by 80k | behavioural onset B1 (epochs) |
|---|---|---|---|
| wd 3.0 | 5 | 5 | 10.1k, 10.1k, 11.2k, 23.7k, and one before 10k (not analysable) |
| wd 1.0 | 5 | 5 | 10.3k, 13.6k, 17.5k, 41.8k, 43.8k |
| wd 0.3 | 5 | 3 (one at 0.13) | 25.8k, 44.9k, 59.0k; two never |
| wd 0 (no-grok control) | 3 | 0 | none (test accuracy ≤ 0.007 at 80k) |
| fast (frac 0.6) | 3 | 3 | generalise by epoch 200; no plateau |

Three of the fast-decay runs generalise inside or at the edge of the 2k to 10k plateau window, so
for them the "plateau" is not a memorisation plateau and the rule has nothing to detrend against.
Eight grok runs have a clean plateau (test accuracy < 5 per cent at 10k, first > 5 per cent at 15k or
later): s0/s2/s3/s4 at wd 1, s1/s2/s4 at wd 0.3, s3 at wd 3. The resume at 55k without optimiser
state precedes the B1 of one of them (s2_wd0.3, 59k).

## The preregistered tests

1. **Lead** (concentration or tail exponent, some matrix, positive in ≥ 12 of 15, sign p < 0.05).
   Over the 13 runs with a B1: best pair is the effective-rank comparator on the unembedding, 10 of
   13, p = 0.046; no Semataxy measure clears p = 0.13. **Not met.** Restricted post hoc to the eight
   clean-plateau runs, nineteen pairs lead in 7 or 8 of 8 with median leads of 8k to 23k epochs,
   including tail exponent (Q, O, V, K), mass and fraction above the edge (V, up) and concentration
   (O, unembed). So in the runs where the rule can work, structure does move long before behaviour.
2. **Timing fit** (Spearman structural onset vs B1 > 0.7). Full set: best 0.80 on a pair with n = 4;
   among pairs with n ≥ 12, best 0.79 (up frac_above) and 0.75 (down norm), most between −0.6 and
   0.6. Clean-plateau set: one pair at 0.79 (V frac_above), the rest scattered around zero. **Not
   met.** The structural onset does not say *when* the grok will come.
3. **Controls** (no-grok arm: onsets in < 1 in 5 pairs). **Failed outright: 0.77** of pairs fire in
   the no-weight-decay controls, and every one of the eight best leading pairs fires in 3 of 3.
   Zero-false-positive pairs are the whole-network aggregates and the span measure (which rarely
   fires anywhere, n = 3 to 8).
4. **Norm comparator.** Spectral onsets come 0.3k to 10k epochs *after* the per-matrix Frobenius
   norm onset in most leading pairs (lead_vs_norm positive); the norm leads them.

## Reading

The structural onset rule fires in nearly every run, grok or no grok. What it detects is that the
weight matrices keep reorganising after training accuracy reaches 1.0, which they do with or
without weight decay and whether or not generalisation ever arrives. That is the alternative written
in the design ("the measure moves at a fixed time regardless of when the grok comes; control
false-positive rates as high as the grok arms' hit rates"). The one existing run's 12k to 17k lead
was real as a measurement and uninformative as a prediction. The same holds for the norm and the
effective rank, logged as comparators: they also move everywhere.

So: **a weight-only, task-agnostic spectral measure of this kind does not predict the generalisation
jump, at level 1.** It describes grokking after the fact (concentration at the jump is still visible
in every grokked run) and it cannot be the live-learning gate's signal as proposed.

Still open, from the same preregistration: the level-2 measures along the circuit (QK, OV, MLP
paths against a shuffled-factor null, `tools/pregrok_level2.py`, job queued on the checkpoints).
Those measure the relation *between* matrices, which is what a circuit is; whether they fire in the
no-decay controls is the remaining test. If they do, the pre-grokking line closes.

## Not done, and why

No rule was tuned after seeing the data; the clean-plateau restriction above is post hoc and is
labelled so. The fast controls were preregistered as "no separable lead" and they have none; they
were not counted as false positives (the 0.77 is the no-decay arm alone; the fast arm gives the same
number). A better design would set the plateau window per run from the behaviour (end it at the
first behavioural movement) and use the no-decay arm to calibrate the CUSUM threshold; that is a new
preregistration, not a re-read of this one.

## Addendum (13:40): level 2 along the circuit, and organisation of the representation

Both ran on the checkpoints in the fast slot (`pregrok_level2.sbatch` 4 min, `pregrok_acts.sbatch`
7 min; logs `results/grokking/*_level2_log.jsonl`, `*_acts_log.jsonl`; summaries
`pregrok_level2_summary.csv`, `pregrok_acts_summary.csv`; behavioural onsets taken from the
100-epoch level-1 logs, structural onsets from the 500-epoch checkpoint logs).

| measure family | wd0 control false-positive rate (mean over pairs) | pairs with zero false positives | best lead among them |
|---|---|---|---|
| level 1, weights (morning) | 0.77 | 10 of 56, all aggregates or span | none |
| level 2, circuit paths QK/OV/MLP vs shuffled-factor null | 0.68 | 14 of 49 | MLP_in span: 6 of 9 lead, p = 0.25 |
| representation, activations on 2,048 inputs | 0.65 | 14 of 56 | mlp_out span: 6 of 11 lead, p = 0.50 |

Among pairs that do fire in the controls, the representation measures mostly *coincide* with the
jump (median leads −2k to +3k epochs, Spearman 0.5 to 0.9 because onset ≈ B1) and the level-2
measures lead by 4k to 10k in 7 or 8 of the 8 clean-plateau runs, as the weights did, with the
same 100 per cent control firing. **Neither preregistered amendment is met.**

The decisive extra fact: Nanda's own task-aware comparator (Gini of the Fourier spectrum of the
embedding, of the unembedding, and of the logits by answer class), which is known to form before
the jump, *also* fails under our onset rule: 6 to 7 of 13 leads, median lead −1.4k to +6.7k, and it
fires in 3 of 3 no-decay controls. A measure with established ground truth cannot pass this rule,
so the rule has no discriminating power as written: a CUSUM of 8 on plateau-standardised residuals
trips on the slow drift every run has after memorisation. The morning's conclusion therefore
splits in two: (a) the measures did not pass; (b) the test as preregistered could not have been
passed by anything, including the known positive. (a) stands as a negative result for the design;
(b) means it is not yet evidence that the measures carry no leading information.

What a fair test would look like, to be preregistered before running (all data already exists):
set the detection threshold *on the no-decay controls* so that at most 1 in 5 pairs fires there,
then apply that threshold once to the grok runs and report leads; use the known-positive Fourier
comparator as the power check (if it does not lead under the calibrated rule either, the sweep
cannot answer the question at this logging resolution). End each run's plateau at its own first
behavioural movement rather than at a fixed 10k, so the fast-decay runs become analysable.

## Addendum (14:15): amendment 3, the control-calibrated rule. No setting works.

`tools/pregrok_calibrated.py` searched the preregistered grid (CUSUM up to 256, |z| up to 16,
persistence up to 20 logs) for a setting whose false-positive rate on the three no-decay controls is
at most 0.20, separately for weights, circuit paths and activations. **None exists** in any family.
The reason is visible in the raw trajectories: without weight decay the weights grow without bound
(value-matrix norm 15.7 → 28.1 over 80k epochs) and every spectral measure drifts with them, by 20
to 130 plateau standard deviations, because the 2k to 10k plateau is nearly flat and its spread is
tiny. A rule standardised by plateau spread fires on that drift whatever the threshold. The
amendment's own clause applies: the sweep cannot be read with this rule family, and no claim either
way is made about the measures from it.

For the record, what the raw numbers show in one grok run against one control (value matrix, mass
above the edge): control 0.16 → 0.20 → 0.22 → 0.26 → 0.22 at 2k/10k/20k/40k/80k; grok run (B1 at
43.8k) 0.40 → 0.69 → 0.78 → 0.83 → 0.92, with the tail exponent collapsing from 3.7 to 1.7 and the
unembedding's mass going 0.25 → 0.96 across the jump. The *concentration at the grok* (T2) is large
and real; the pre-jump drift (0.69 → 0.83 over 30k epochs) is of the same order as the control's
drift (0.20 → 0.26) once scaled, which is why no threshold separates them. A rule based on
magnitude of change relative to the measure's own range, or on acceleration, would be a fourth
amendment; it is not run today.

Where this leaves the pre-grokking claim: **not supported and not refuted by this sweep.** The
measures concentrate sharply *at* the transition in every grokked run; nothing in three measure
families separates "will generalise" from "will not" before the transition under any rule we
preregistered. The weight-only route to a leading indicator is closed for now; the representation
route coincided with the jump rather than leading it.
