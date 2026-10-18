# PRE-REGISTRATION — lens fit depth sweep, Qwen2.5-7B-Instruct

Written and committed **before any fit is run**. 22 August 2026.

## Why

`common.py:33` sets injection at `round(0.25*L)` and the readout band at the
middle third. That formula was back-fitted to constants inherited from an
abandoned run (log §5: "The 0.5B numbers reproduce the abandoned run's hardcoded
values exactly"). It was never swept, never given a criterion, and nothing above
0.65 depth has ever been fitted. Every Phase 3 conclusion rests on it.

## Design

Model Qwen2.5-7B-Instruct, 28 layers. Nine fits at injection layers
**4, 6, 7, 10, 13, 15, 18, 21, 24** = depth **0.143, 0.214, 0.250, 0.357,
0.464, 0.536, 0.643, 0.750, 0.857**.

**Band rule, stated explicitly and identical at every depth:**

```
band = range(inject + 2, min(inject + 12, n_layers))
```

Starts two layers above the injection site, up to ten layers wide, clipped at
the top of the stack. At `inject=7` this reproduces the existing 7B band
`L9..L18` exactly, so the current fit is reproduced rather than approximated.
Defined once in `common.py` and imported; not re-derived per script.

Everything else identical to the existing 7B fit: same wikitext corpus, same 162
tracked-token seeds, same restricted-cotangent procedure, bf16, same
`stage2_verify` gate on France->China (capital, language, continent, currency).

**One deliberate reduction: `n_prompts = 1000`, not 3000** — nine fits at 3000
would be ~20 GPU-hours. Identical at every depth, so the comparison across
depths is internally valid. Cost of the reduction is checked by the control
below.

## Pre-declared confound

**Above `inject = 15` the band is truncated by the top of the stack**: widths are
10, 10, 10, 10, 10, 10, 8, 5, 2 layers for the nine depths in order. Depths
0.643, 0.750 and 0.857 therefore do **not** test the same construction as the
rest — band width is confounded with depth there. This is geometric and
unavoidable: a ten-layer band cannot sit above layer 27.

**Those three depths are reported but are EXCLUDED from the flat-vs-peaked
adjudication.** Adjudication uses the six untruncated depths (0.143 to 0.536)
only. Declared now so it cannot be chosen after seeing results.

## Internal control — the sweep is void without it

`inject = 7` at `n_prompts = 1000` must reproduce the existing **4/4** gate. If
it does not, the reduction from 3000 has cost the fit its validity and **the
sweep adjudicates nothing** — it would then be a statement about fit budget, not
about depth. That outcome is to be reported as a failed sweep, not reinterpreted.

## Hypotheses and the threshold that separates them

Primary measure: `swap_correct` out of 4 on the fact-swap gate.
Secondary: `swap_strict` = `swapped_correct AND NOT swapped_still_src`, out of 4.
Fit quality: worst cross-check cosine against the full Jacobian (64 prompts).

**H_flat — depth is a free parameter.** The lens finds reportable structure
everywhere; 0.25 is arbitrary but harmless.
> Predicted: gate = 4/4 at **at least 5 of the 6** untruncated depths, and the
> spread (max minus min gate score across those 6) is **<= 1**.

**H_peaked — the reportable subspace is localised.** There is a depth region
where the gate works and regions where it does not.
> Predicted: spread across the 6 untruncated depths is **>= 2**, with at least
> one depth at 4/4 and at least two depths at **<= 2/4**.

**Ambiguous.** Spread of exactly 2 without two depths at <=2/4, or any
non-contiguous pattern (e.g. pass, fail, pass) — reported as ambiguous and NOT
resolved in either direction. A non-contiguous pattern would additionally
suggest the gate is noisy at n=4, which is a small gate; that reading is
available in advance and is not a post-hoc escape.

**Secondary question, stated now:** does 0.25 sit near the best depth? Answered
by whether `inject=7` is at or within 1 gate point of the maximum across all
nine depths. With a 4-point gate this is coarse; if several depths tie at 4/4,
the honest answer is "0.25 is among the best, and so is everything else", which
is H_flat, not a vindication of 0.25.

## Interpretation fixed in advance

**A flat result is NOT a null.** It would mean the lens finds reportable
structure at every depth, which is informative about what the lens measures, and
it would make depth a genuinely free axis for a successor experiment rather than
a hidden confound. A peaked result would mean the inherited 0.25 was load-bearing
and needs justifying against the peak.

## Scope

One model, one lens family, one fact-swap gate of n=4, one concept pair
(France->China). No claim about other models, other concepts, or the install
behaviour of any of these lenses. **No write-path, training, or install run
follows from this sweep.**

---

## AMENDMENT 1 — 23 August 2026, before any gate result exists

**Change:** `n_prompts` reduced from 1000 to **400**, and the control depth
(`inject=7`) is run **first and alone** before the other eight.

**Reason: cost, not results.** Measured marginal fit rate on the rented A100 is
**6.96 s/prompt** (two samples 8 min apart, 70 prompts in 487 s) against the
reference fit's 2.64 s/prompt at matched `seed_batch=6` and `max_seq_len=192` --
the remaining gap is the hardware, which is a contended community-cloud
instance. At n=1000 the sweep is 19 h and ~$23; at n=400 it is ~8.7 h and ~$10.

**No gate result has been observed at any depth.** The first sweep attempt
produced one verdict (inject=4: 3/4 swapped, 4/4 moved) which was **discarded
and refitted** because it used the wrong `seed_batch` and `max_seq_len`; no
result from a correctly-configured fit exists at the time of this amendment.

**Nothing else changes.** The band rule, the nine depths, the truncation
exclusion, the flat/peaked thresholds and the interpretation are all as
originally registered.

**The control's gating role is strengthened, not weakened, by running it first:**
if `inject=7` at n=400 does not reproduce the reference **4/4**, the sweep is
declared underpowered and **stops there** rather than producing eight more
uninterpretable fits. That is a stopping rule, fixed now, before the number is
seen.

---

## RUN LOG — 23 August 2026. Attempt 2 completed 8/9 depths; results DESTROYED.

Recorded so the next attempt does not repeat it.

**Control PASSED** (`inject=7`, depth 0.250, n=400): baseline 4/4, swapped 4/4,
**strict 4/4**, moved 4/4, VERDICT PASS. Fit 63.2 min, peak VRAM 19.335 GB and
`max_seq_len=192` — matching the reference lens exactly. The stopping rule was
therefore cleared and the reduction to n=400 is validated. **This is the only
depth result that survives.**

**The other eight verdicts were deleted.** The pull used an unquoted remote glob,
so the LOCAL shell tried to expand it, matched nothing and failed; the pod was
destroyed in the same command block without checking the pull had succeeded. 8.4
hours and $10 of completed fits lost. They were never read, so their values are
unknown.

**One finding survives, from the error text rather than the data:**
`ValueError: source_layers must all be < target_layer=27; got max=27`. The lens
transports into the final-layer basis, so **no source layer may be the final
layer**. Under the registered rule `band = range(inject+2, min(inject+12, L))`
the band reaches L-1=27 for any `inject >= 16` on a 28-layer model.

### AMENDMENT 2 — band must exclude the final layer

```
band = range(inject + 2, min(inject + 12, n_layers - 1))
```

Reason is structural, not result-driven: the final layer is the lens's target
basis and cannot also be a source. Revised widths for the nine depths:
10, 10, 10, 10, 10, 10, **7, 4, 1**. The three deepest remain excluded from the
flat-vs-peaked adjudication, as originally registered. No threshold changes.

### Mandatory procedure for attempt 3

1. Quote the remote glob so the REMOTE shell expands it.
2. **Pull, then verify the local file count, then read the results.**
3. **Destroy only after the results have been read.** Never pull and destroy in
   one command block.

---

## RESULT — attempt 3, 26 August 2026. All 9 depths. $3.20.

| inject | depth | band | w | gate | strict | moved | verdict |
|---|---|---|---|---|---|---|---|
| 4 | 0.143 | 6-15 | 10 | 3/4 | 3/4 | 4/4 | PASS |
| 6 | 0.214 | 8-17 | 10 | 3/4 | 3/4 | 4/4 | PASS |
| 7 | 0.250 | 9-18 | 10 | 4/4 | 4/4 | 4/4 | PASS |
| 10 | 0.357 | 12-21 | 10 | 4/4 | 4/4 | 4/4 | PASS |
| 13 | 0.464 | 15-24 | 10 | 4/4 | 4/4 | 4/4 | PASS |
| 15 | 0.536 | 17-26 | 10 | 4/4 | 4/4 | 4/4 | PASS |
| 18 | 0.643 | 20-26 | 7 | 4/4 | 4/4 | 4/4 | PASS |
| 21 | 0.750 | 23-26 | 4 | 2/4 | 2/4 | 2/4 | FAIL |
| 24 | 0.857 | 26-26 | 1 | 0/4 | 0/4 | 0/4 | FAIL |

Strict and lenient agree at every depth. Adjudication set (untruncated,
0.143-0.536): gate [3,3,4,4,4,4], spread 1, four of six at 4/4.
**H_flat NOT MET** (needs five of six). **H_peaked NOT MET.** Formally
**AMBIGUOUS**, as pre-registered, and reported as such.

### The correct reading: the gate SATURATES, it does not measure

A 4-point gate returning its maximum across five consecutive depths has stopped
resolving. "Reportable structure everywhere" and "fact-swap cannot tell these
depths apart" produce identical data, and this gate cannot separate them.
**Where the instrument had resolution left -- the deep end -- band width is
confounded with depth exactly as pre-declared. So the sweep discriminates
nowhere that it is not confounded.**

An earlier draft of this write-up called the flat region informative. It is not.
Saturation is an absence of measurement, not a measurement of absence.

### What it does settle, which was the point

**0.25 was not a bad choice.** Phase 3's conclusions are not scoped to a
peculiar depth; they are scoped to a depth that behaves like every other depth
in the usable range. **This removes the confound rather than resolving the
question** -- and removing it was the reason for the sweep.

### What it costs (Parag's reading, recorded as his)

If the lens finds reportable structure across half the stack, then a design of
the form "unfreeze where the lens points" points at half the model. He
identifies this as outcome two of `NEXT-lens-depth.md`, a document not present
in this repo -- so it is recorded here as his conclusion, not re-derived. The
selector would have to come from somewhere other than this lens, or from a
sharper gate.

### Next run, if any (NOT RUN)

A gate that does not saturate: more facts, harder facts, or a graded measure
(rank of the destination token) instead of a 4-point binary. `rank_dst` is
already recorded per row and moved 110->1, 94->1, 46->1, 60->1 at depth 0.25 --
a graded statistic exists in the data already collected and costs nothing to
analyse.

### Fixed, from this run's own defect

`crosscheck_{model}.json` was overwritten on every fit, so eight of nine
`cos_min` values were lost (survivor: 0.999996 at depth 0.857, 2 layers, 8
prompts). Now written per tag.

---

## EXPLORATORY re-analysis: does a graded measure rescue resolution? NO.

**Not pre-registered.** Chosen after seeing the gate saturate, and reported as
exploratory. I predicted `rank_dst` would separate the five depths the binary
gate tied at 4/4. **It does not.**

`swapped_rank_dst` per fact (1 = destination is the model's top token):

```
depth  w | capital language continent currency | gate
0.143 10 |    1        2         1        1    | 3/4
0.214 10 |    1        2         1        1    | 3/4
0.250 10 |    1        1         1        1    | 4/4
0.357 10 |    1        1         1        1    | 4/4
0.464 10 |    1        1         1        1    | 4/4
0.536 10 |    1        1         1        1    | 4/4
0.643  7 |    1        1         1        1    | 4/4
0.750  4 |    3        1         5        7    | 2/4
0.857  1 |   51        9        32       43    | 0/4
```

**The destination token is rank 1 on every fact at every depth from 0.25 to
0.643.** There is no headroom in this statistic either -- it is pinned at the
floor exactly where the gate is pinned at its ceiling. Mean log10 rank gain is
identically 1.86 across all five, because it is log(baseline/1) and the
baselines are fixed. **The graded measure confirms the saturation rather than
resolving it**, which is a stronger version of the same finding, not a rescue.

### One unsaturated signal, and it is weak

`swapped_rank_src` -- how far the ORIGINAL fact is pushed down -- does vary:

```
depth  0.143 0.214 0.250 0.357 0.464 0.536 0.643 0.750 0.857
med    153.5 157.0 172.0 183.0 151.5 142.0  73.5   8.5   1.0
```

It rises to a maximum near depth 0.357 and declines thereafter: the deeper the
injection, the less the source fact is suppressed, even while the destination
stays at rank 1. So depth may affect the intervention's **specificity** without
affecting its **success**.

**This is not a result.** It is exploratory, n=4 facts, medians over four
numbers, no pre-registered threshold, and at the deep end it is confounded with
band width exactly as the primary analysis was. It is recorded as the one place
in this dataset where an unsaturated measure exists, and as the obvious thing a
sharper gate would be built to measure. NOT RUN.
