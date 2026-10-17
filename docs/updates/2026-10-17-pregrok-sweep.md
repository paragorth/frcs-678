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
