# Step-and-hold — Rule 0 analysis, written before the code

DECISIONS rule 0: write down what the number is under the null and under the
alternative. If they are the same, do not write the code.

## The design as specified in log 20k
1. Hold alpha at `A_high` for k tokens.
2. Step down to `A_low`, hold k more.
3. Separately, hold `A_low` for 2k tokens from the start.
4. Compare report rate in the second half of (2) against (3).

H0: identical. H1: (2) > (3).

## The problem with that comparison

**A transformer with a KV cache is not memoryless, so H0 as stated is false by
architecture, not by ignition.** During the `A_high` phase the model emits
concept tokens; those tokens enter the context and are attended to for the rest
of the generation. Arm (2) therefore has concept text in its context that arm (3)
does not. Ordinary topic continuation predicts (2) > (3) with no bistability
anywhere.

That is the same confound that made the ramp's 14.9-sigma "hysteresis"
uninterpretable. Running (2) vs (3) alone would reproduce it.

## The comparison that is actually informative

Add a **text-matched** arm:

- **A — step-down**: alpha `A_high` for k tokens, then `A_low` for k tokens.
- **D — text-matched replay**: teacher-force A's first k tokens as context, so
  the text is *identical*, but apply `A_low` throughout — including during the
  forced prefix. Then generate k tokens at `A_low`.
- **B — low-hold**: `A_low` for 2k. (Kept, but it is NOT the primary contrast.)

| quantity | H0 (threshold + text memory) | H1 (activation-level bistability) |
|---|---|---|
| second-half report, A vs B | A > B — text confound alone predicts this | A > B |
| **second-half report, A vs D** | **A = D** — same text, so same report | **A > D** |
| primed control (concept forced, no injection) | dies quickly | dies quickly |

**A vs D is the test.** The text is held identical; the only difference is
whether the KV for those positions was computed under high or low injection.
H0 and H1 differ there, so Rule 0 is satisfied for A vs D and **violated for
A vs B**. The write-up must lead with A vs D.

## Honest limit, recorded in advance
If A > D, that shows report depends on activation history beyond text. Whether
that is *ignition* (recurrent self-amplification) or merely *KV persistence* (the
cache carrying a concept written into it) is an interpretive question this design
does not settle. It is the operational signature of history-dependence, and it
should be reported as that and not more.

Per WHY.md: a population of thresholded units with jittered thresholds sums to a
smooth sigmoid, so a step alone is not ignition — ignition is an additional claim
about synchrony. A language model is the better instrument because a step cannot
be inherited from spiking units, but this design tests history-dependence, not
synchrony.

## Pre-committed thresholds
- `A_high` and `A_low` are chosen from the model's own measured report curve:
  `A_high` = lowest alpha with report >= 70%; `A_low` = highest alpha with report
  <= 15%. Both must sit inside the coherence ceiling measured on this run's own
  prompt pool by repeat fraction.
- **Coherence gate first.** Any (arm, alpha) whose second-half repeat fraction
  exceeds 0.5 is excluded before analysis, and the excluded count is reported.
- Effect: `report(A) - report(D)` in the second half, per prompt, paired.
- **Positive result** requires the paired difference to exceed **2 SEM** AND the
  primed control to remain at baseline. n >= 150 prompts.
- **Null** is reported as a null only with the achieved power stated: report the
  minimum detectable paired difference at the achieved n, and do not describe an
  undetected effect smaller than that as absent.
