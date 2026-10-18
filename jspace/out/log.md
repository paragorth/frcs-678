# Ignition experiment — run log

Local prep and design decisions precede the GPU run. Costs and wall times for
the rented pod are recorded in §8 as they happen.

---

## 0. Killing the local run

No CPU fit process was alive by the time I looked (`ps` clean), so nothing had
to be killed. The abandoned run left `out/fit.log` and `out/bench4.log`, which
record why it was hopeless: **64.7 s/prompt at seq=48 on CPU fp32**, i.e. 1.8 h
for 100 prompts, for the 0.5B.

No stale lens existed — `find` turned up no `qwen*_jacobian_lens.pt` and no
`lens_ckpt.pt`, so there was nothing to delete. The refuse-to-refit guard is
retained and now keys off a per-model, per-tag filename.

## 1. Correction to a premise in the brief

The brief attributes the local blow-up to "the full 150k × d accumulator".
That is not what the library does. `jlens.fitting.fit` accumulates
`[d_model, d_model]` per layer — 896² × 4 B ≈ 3 MB per layer for the 0.5B.
The vocabulary never enters the accumulator.

The real cost is **backward passes**: the estimator seeds a one-hot cotangent
per *output dimension*, so it is `d_model` backwards per prompt (896 for the
0.5B, 1536 for the 1.5B). That, not memory, is what made the CPU fit hopeless.

This does not change the decision — restricted cotangent seeding is still the
right move, and is what I implemented — but it changes the reason, and the
saving is ~10× (1536 → 162 backwards), not the ~100× a vocab-sized accumulator
would imply.

## 2. Freezing the sweep prompts (done first, as instructed)

`out/sweep_prompts.json`, 20 neutral prompts, sha256[:16] = `3d8624aa123f044d`.
Verified before anything else touched them: **no cluster-A word, cluster-B word
or control word occurs in any sweep prompt.** Prompts are 9–15 tokens.

## 3. Token set

Derived from the frozen prompts, in the order the brief specifies.

| group | n | what |
|---|---|---|
| incumbent | 132 | content words actually occurring in the 20 sweep prompts |
| cluster_a | 5 | spider, web, insect, legs, eight (injected) |
| cluster_b | 5 | piano, music, keys, sound, notes (contrast) |
| control | 10 | unrelated, in no prompt and neither cluster |
| **tracked total** | **152** | read out in Stage 3 |
| stage2 (seed-only) | +10 | France, China, Paris, Beijing, French, Chinese, Europe, Asia, Euro, Yuan |
| **seed rows** | **162** | deduped by token id; what the fit is targeted at |

Every sweep prompt carries **at least 5 tracked incumbents** (min 5, median 7),
so eviction has competitors to be observed against — the point of the fix.

Five words are tracked by their **leading token only**, having no single-token
form: `corroded, jammed, schoolhouse, scrubbing, steaming`. The brief specifies
leading-token tokenisation, so this is as intended, but those five measure a
word-fragment rather than a word.

Stage 2 tokens are **seeded but not read out**, so they cannot contaminate the
eviction analysis.

## 4. Restricted cotangent seeding — and a proof that it is exact

Instead of one-hot cotangents over `d_model` output dimensions, the fit seeds
`normalize(W_U[t] * g)` for each of the 162 seed tokens, giving
`r_{l,t} = J_l^T (W_U[t] * g)` directly. `g` is the final-norm gain, with
RMSNorm treated as a fixed diagonal — the same linearisation the paper's
transpose-row definition assumes.

This is **not an approximation**. Seeding cotangent `c` at every valid target
position and averaging over source positions returns exactly `J_l^T c` under
the library's own estimator. Verified empirically rather than asserted:
`fit_lens.py --crosscheck` fits the full `J` on the same prompts and compares.

> **cross-check result (0.5B, identical prompt): cosine = 1.000000 at every one
> of the 9 fitted layers, min and mean.**

Cotangents are unit-normalised and scaled by 256 for fp16 headroom; both are
irrelevant to the result, since only each row's *direction* is ever used.

## 5. Depth plan

Derived, not hardcoded: injection at `round(0.25·L)`, band = middle third.

| model | layers | inject | band |
|---|---|---|---|
| Qwen2.5-1.5B-Instruct (primary) | 28 | L7 | L9–L18 (10) |
| Qwen2.5-0.5B-Instruct (fallback, `--model 0.5b`) | 24 | L6 | L8–L15 (8) |

The 0.5B numbers reproduce the abandoned run's hardcoded values exactly, which
is a useful check that the formula is the one that was intended.

## 6. Design problems found in local dry runs (0.5B, CPU)

I dry-ran the whole pipeline on the 0.5B before renting anything. Three
problems surfaced that would have wasted the GPU run.

### 6a. The α unit was ~180× too large

Original unit was the layer's **mean residual norm** (138.3 at L6). But the
concept direction's coordinate normally carries an RMS of **0.76** over non-BOS
positions. So α=1 meant "put ~180× the usual amount of this concept on one
direction". Consequence, measured: at α≥1 *every* condition destroyed the
model — random and orthogonal controls included, KL from baseline 6.5–13.4,
top-1 tokens degenerating to junk like `)&&(`, `头`, `blems`. The sweep would
have measured only how the model breaks.

**Fixed:** α is now in units of the baseline RMS of the *concept* coordinate at
the injection layer. One common unit across all conditions, so a given α is the
same target magnitude everywhere. After the fix, KL from baseline is 0.002–0.75
across the whole dense range and the modal top-1 token stays `' The'` — the
model stays intact while the concept's rank moves 29203 → 190.

### 6b. The bracketing metric manufactured its own step

My first bracketing metric was the concept's z-score among tracked tokens. It
showed a **hard step at α=0**: −7.3 for all α<0, +7.4 for all α>0, dead flat
across three orders of magnitude of α.

That step was entirely an artefact. The z-score divides by a spread that grows
with α at the same rate as the numerator, so it is scale-invariant and reports
a step even when the underlying response is perfectly linear — which, measured
directly, it was:

```
alpha    -16     -8     -4     -1      1      2      4     16     64
raw    -1360   -681   -341    -85     83    168    339   1357   5426     (ratio 65.3x at 64x)
```

**This is the experiment's own thesis appearing inside my instrumentation.** I
caught it before it set the sweep range. Bracketing now uses a
common-mode-corrected metric normalised by the **baseline** spread, which stays
dose-dependent. The scale-invariant `z` and the binarised `arg` are still
computed and recorded in the meta — deliberately, so the artefact is visible in
the output rather than hidden by it.

Side by side on the 0.5B coarse scan, corrected units:

| α | m (raw, corrected) | z (scale-invariant) | arg (binarised) |
|---|---|---|---|
| 1 | 1.01 | +0.98 | 0.000 |
| 2 | 1.99 | +1.87 | 0.100 |
| 4 | 3.96 | +3.34 | 0.775 |
| 8 | 7.94 | +5.19 | 1.000 |
| 64 | 67.74 | +7.53 | 1.000 |

The raw response is linear (slope ratio 1.35). The binarised readout steps
sharply over α∈[2,8]. That is the whole question, already visible in the dry run.

### 6c. Bracketing rule, fixed in advance

1. Test linearity: slope `m(α)/α` over positive coarse α. If max/min < 1.5 →
   **LINEAR**, meaning there is no threshold to bracket and any bracket on the
   raw curve would be arbitrary.
2. If LINEAR, choose the dense range to span the **apparent** step instead:
   `hi = 1.5 × (first α where arg ≥ 0.99)`, `lo = −0.25 × hi`. This guarantees
   every raw single-trial value exists across exactly the region where a
   degraded readout would report ignition.
3. If nonlinear, min-max normalise `m` and take the last α below 5% / first
   above 95%, widened one grid step; full coarse range if it never crosses.

On the 0.5B dry run this selected [−3, 12], spanning the arg transition at [2,8].

## 7. Stage 3 measurement design

- **Patch form** is the lens-coordinate clamp `h' = h − (h·u)u + a·u`, not naive
  addition. Verified numerically: coordinate lands on target to 5e-7, orthogonal
  remainder changes by 2e-7, whereas naive addition leaves the coordinate
  varying (3.41, 1.57, 2.02 for a target of 3.0).
- **α=0 is still an intervention** (it clamps the concept coordinate to zero).
  A true no-intervention `baseline` condition is recorded separately.
- **Conditions:** `concept`, `random`, `orthogonal`, `random_matched`, `baseline`.
- **`orthogonal`** is orthogonal to the span of the 162 tracked readout
  directions — measured leakage max|cos| = 5.6e-08. Note that "orthogonal to
  J-space" in the literal sense is vacuous: `J_l` is square and generically full
  rank, so nothing is orthogonal to its row space. The meaningful complement is
  the subspace this measurement can actually see, which is what is used.
- **`random_matched`** was added because the clamp form *cannot* match norms
  across directions: how far it must move depends on how much of that direction
  the residual already carries, and the orthogonal complement carries far more
  (coordinate RMS 13.96 vs the concept's 4.20 including BOS). This condition
  applies an additive perturbation whose per-position delta norm matches the
  concept trial exactly. Verified: delta norms match to the digit
  (3.66/3.66, 1.54/1.54, 4.02/4.02, 6.87/6.87, 9.72/9.72).
- `delta_norm` is recorded on every row and every trial, so norm matching is
  auditable rather than assumed.

### A finding the analysis will need

Injecting along the concept direction lifts **every** tracked projection, not
just the concept's — a large common-mode term (at α giving cluster_a +84.7,
controls +42.1, incumbents +39.5). The tracked readout directions are not
mutually orthogonal and share a common component. Eviction therefore cannot be
read off raw projections directly; it needs the common mode removed. Because
nothing was aggregated, that correction remains possible downstream.

### Output format

Long-format **gzipped JSONL**, one row per
`(prompt, α, layer, tracked_token, condition)` — a deviation from "JSON" in
container only, taken because the file is ~3.07 M rows; plain JSON would be
~400 MB of single-line text. No averaging, no thresholding, no pre-aggregation.

Each row carries two raw readouts of the same trial: `value` (final token
position, canonical) and `value_meanpos` (same trial, averaged over non-BOS
positions). The second is a within-prompt spatial reduction, **not** an average
across trials — included because incumbent content words sit early in these
prompts, so reading only the last position could hide eviction rather than
measure it. Single-trial resolution is fully preserved either way.

A second file, `stage3_trials_*.jsonl.gz` (one row per trial, ~2020 rows),
records a behavioural readout independent of the lens: top-1 token, concept
logit/logprob/rank, entropy, KL from baseline. The lens projection can be
linear while the model's actual output turns over sharply; this is where that
would show.

## 8. What this is not

**This is not the paper's occupancy measure.** Occupancy is matching pursuit
over the full direction set. This is **projection onto a restricted set of
chosen directions** — 152 tracked tokens picked in advance. That is fine for a
dose–response curve and fine for eviction *among tracked items*, and it is not
a claim about what else occupies the workspace. Anything not in the tracked set
is invisible to this measurement by construction.

## 9. Deviations from the brief

| # | Deviation | Why |
|---|---|---|
| 1 | Code shipped to the pod by rsync over SSH, not `git push` | No git remote exists here and `gh` is not installed; rsync needs no hosting account |
| 2 | Output is gzipped JSONL, not a single JSON array | ~3.07 M rows; identical long-format content |
| 3 | Extra `value_meanpos` column | Guards against last-position-only readout hiding eviction; not an across-trial average |
| 4 | Fourth condition `random_matched` | The clamp form cannot match norms across directions; see §7 |
| 5 | Extra per-trial behavioural file | Second, lens-independent channel for a step |
| 6 | α unit is the concept coordinate's baseline RMS, not the residual norm | §6a — the original unit broke the model in every condition |
| 7 | Stage 2 verdict is not strict top-1 only | §10 |
| 8 | Fitting corpus cached at 1500 prompts | Covers the 3× refit without re-downloading mid-run |

## 10. Stage 2 verdict rule — pre-committed

Fixed before any 1.5B result was seen. Strict top-1 equality alone is too
brittle: in the 0.5B dry run the currency swap produced `" Ren"` (Renminbi —
right fact, wrong surface token) while the language swap produced `" English"`
(genuinely wrong). A rule that cannot separate those would force a pointless
refit on the first and excuse a real failure on the second.

Each of the four facts is scored two ways, both reported:
- **swapped** — top-1 is exactly the China answer (strict)
- **moved** — top-1 is no longer the France answer AND the China answer's rank
  improved ≥10×  (surface-form independent)

**PASS** requires `swapped ≥ 3/4` AND `moved = 4/4`. `PARTIAL` if `moved ≥ 3/4`.
`FAIL` otherwise. `INCONCLUSIVE_MODEL_LACKS_FACTS` if the baseline model gets
fewer than 3/4 right to begin with. Nothing is rounded up.

On failure: one refit at 3× prompts, one retry, both attempts logged. If it
still fails, that is reported plainly and prominently, and Stage 3 output is
tagged `STAGE2_FAILED.json` — I do run Stage 3 anyway, because the pod is
already up and it costs minutes, but the log leads with the failure and the
result is marked not interpretable as ignition.

## 11. Local dry-run results (0.5B, CPU, 2-prompt lens — code validation only)

Not science; these exist to prove the code paths work before spending money.

- Stage 1 restricted fit: 2 prompts, 90 s, peak RSS 2.68 GB. Cross-check PASS.
- Stage 2: baseline 3/4, strict 2/4, moved 4/4 → PARTIAL. Even a 2-prompt lens
  moved rank_dst 566→1 (capital), 18→1 (continent), 88→2 (language), 317→9
  (currency).
- Stage 3: 510,720 rows in 2.0 min, 6.9 MB gz. Controls well-behaved.

## 12. GPU run — infrastructure

Account created via Google sign-in (browser-driven; the user did signup and
payment, which I cannot do). Two API keys were created and failed before the
third worked — **both failures were my transcription errors**, reading a
50-character secret off a screenshot with ambiguous `O`/`0` and `l`/`1`. The
clipboard is sandboxed from the page and DOM reads were blocked, so the fix was
a zoomed monospace render. Worth recording because I initially misdiagnosed it
as an API-scope problem and nearly created an unrestricted key on that theory.
The working key is **Restricted / GraphQL Read-Write**, which is sufficient for
the whole REST pod + volume lifecycle. Least privilege held.

Auto-Pay had switched itself **on** when credits were loaded, configured to
charge $200 whenever the balance fell below $100 — with a $20 balance it was
already under the threshold. The user caught this; I disabled it. Nothing was
charged.

| item | value |
|---|---|
| GPU | RTX 4090 24 GB, EU-RO-1, **SECURE** cloud |
| price | $0.74/hr (community had no 4090 capacity in EU-RO-1) |
| billed | **1.075 h = $0.80** |
| pod destroyed | confirmed, `pods: []` |
| network volume | `ufbjixb5jv`, 10 GB, **kept** (~$0.70/month) |
| image | `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, torch 2.8.0+cu128 |

### The NaN incident

The first fit produced a lens of **pure NaN** — and saved and loaded without
complaint. `d_mean=nan` was printed on every prompt from the start and I nearly
read past it. The accumulator had been silently poisoned for 310 prompts: one
non-finite prompt contaminates the running mean and every prompt after it.

My first diagnosis was wrong. I assumed fp16 overflow from the cotangent scale,
but a controlled comparison on prompt 1 (fp16 / bf16 / fp32 x scale 256 / 1.0)
produced **zero NaN cells in all six configurations**. It was prompt-specific,
not dtype-generic. Fixes applied, both of which were needed:

1. **Finiteness guard** in `fit_restricted`: any prompt yielding a non-finite
   row is skipped and counted, and the fit aborts if >25% of prompts go bad
   rather than fitting on the remainder. A final check refuses to return a
   non-finite lens at all.
2. **bf16 instead of fp16** (~1e38 range vs 65504).

After both: **500/500 and 1500/1500 prompts used, 0 skipped.** So it *was* an
fp16 overflow, just not one prompt 1 exhibited. Had I shipped only the guard,
prompts would have been silently dropped; had I shipped only bf16, the next
pathological corpus would have poisoned a run again.

## 13. Results

### Stage 1
| fit | prompts | wall | bad |
|---|---|---|---|
| base | 500 | 8.5 min | 0 |
| 3x refit | 1500 | 25.7 min | 0 |

**Cross-check vs the full Jacobian: worst cosine 0.999978** across all 11
layers (1.000000 on the 0.5B in fp32). Restricted cotangent seeding is
confirmed equivalent to computing the full `J` and applying `J^T c`.

### Stage 2 — FAILED after refit (reported plainly, per the pre-committed rule)

| attempt | lens | baseline | strict top-1 | moved | verdict |
|---|---|---|---|---|---|
| 1 | 500 prompts | 3/4 | 2/4 | 4/4 | PARTIAL |
| 2 | 1500 prompts | 3/4 | 2/4 | 4/4 | PARTIAL |

Attempt 1 detail:
```
capital     Paris  -> ' ______'   rank_dst  381 -> 7
language    French -> ' Chinese'  rank_dst  382 -> 1   strict pass
continent   Europe -> ' Asia'     rank_dst   11 -> 1   strict pass
currency    euro   -> ' ren'      rank_dst 1254 -> 8
```

**All four facts moved together** — which is the multi-fact structure the paper
reports — but only 2/4 landed exactly on the target token, so under the rule
fixed in advance this is PARTIAL, not PASS, at both 500 and 1500 prompts.
Tripling the corpus changed nothing, so the failure is not undertraining.

Two caveats that make this *less* damning than the verdict looks, neither of
which I am using to relax the criterion after the fact:
- `' ren'` is Renminbi: the right fact, the wrong surface token.
- The baseline currency answer was `' euro'` lowercase against an expected
  `'Euro'`, so my scorer counted a **correct** baseline as wrong. That alone
  capped baseline at 3/4 and made PASS unreachable on that row.

Per §10 this means **Stage 3 output is not interpretable as ignition in this
model.** `STAGE2_FAILED.json` is written alongside the results. Stage 3 was run
anyway (minutes, on an already-running pod) and is reported as what it is.

### Stage 3 — the result

3,070,400 rows, 48 MB gz, 1.0 min. 20 prompts x 25 alpha x 10 layers x 152
tracked tokens x 4 conditions, plus baselines. Nothing aggregated.

alpha unit = baseline RMS of the concept coordinate at L7 = **1.671**; the mean
residual norm there is 929.9, i.e. **556x larger** — using it as the unit, as
the design originally did, would have meant every condition destroying the model.

**The raw dose-response is a straight line:**

```
value = 1.0737 * alpha + 1.4133      R^2 = 0.999656     (alpha in [-16, +64])
```

**The same data, read through degraded instruments:**

| alpha | RAW mean | z-score | argmax% | avg + threshold |
|---|---|---|---|---|
| -2.67 | -1.64 | -1.62 | 0.0% | -- |
| 0.67 | 1.88 | 0.57 | 0.0% | -- |
| **4.00** | **5.52** | **2.75** | **61.0%** | **FIRING** |
| 7.33 | 9.26 | 4.43 | 97.5% | FIRING |
| 10.67 | 13.01 | 5.52 | 100.0% | FIRING |
| 64.00 | 71.37 | 7.09 | 100.0% | FIRING |

**A measurement that is linear to R^2 = 0.9997 produces a clean, sharp
ignition-shaped step as soon as it is normalised by a scale that moves with the
signal, or binarised into "is the concept winning".** The step is manufactured
entirely by the readout. This is the experiment's central claim, and it
reproduces on the 1.5B.

### Controls — both pass

| alpha | concept | random | random_matched | orthogonal |
|---|---|---|---|---|
| -16 | -15.13 | 1.93 | 1.94 | 1.32 |
| 0.67 | 1.88 | 1.14 | 1.14 | 1.20 |
| 24 | 27.15 | -0.02 | -0.03 | 2.18 |
| 64 | 71.37 | -2.31 | -2.30 | 1.61 |

- **Random**: flat (1.93 -> -2.31) while the concept sweeps -15 to +71. The step
  is not an artefact of injection magnitude.
- **Orthogonal**: barely moves (1.32 -> 2.24 peak), leakage max|cos| = 4.6e-08.
  The lens isolates what it claims to.
- **random_matched** tracks `random` to two decimals with delta norms matched
  per position, so the clamp-form norm mismatch was not doing any work.

### No eviction

Injecting `spider` **raises every tracked token**, it does not displace
incumbents. At alpha=64: concept 71.37, incumbents 27.18, controls 29.47,
cluster_b 25.84. Controls rise slightly *more* than the prompt's own content
words. This is the common-mode term flagged in §7, and it dominates: the
tracked readout directions are not mutually orthogonal. **No eviction is
observable in raw projections**, and the token-set fix — which made incumbents
trackable at all — is what allows that to be stated rather than assumed.

## 14. A flaw in my own bracketing, found after the fact

The linearity test computes `m(alpha)/alpha` and compares max to min. It does
**not** subtract the intercept, and `m(0) = 0.047` is non-zero. So the alpha=0.1
point reported a slope of 1.20 against ~0.73 everywhere else, the ratio came out
1.664 > 1.5, and the run declared the response **NONLINEAR** — when the
intercept-corrected fit is linear to R^2 = 0.9997.

Consequence: it went looking for a sigmoid crossing, found none, and fell back
to the full coarse range [-16, 64]. That is the safe fallback and the data is
sound, but 25 points over an 80-wide range gives a step of 3.33, so resolution
near zero is coarser than intended. The interesting region for the *binarised*
readout (alpha 2-8) is covered by only two dense points.

The conclusion is unaffected — the coarse scan sampled that region at 0.1, 0.25,
0.5, 0.75, 1, 1.5, 2, 4, 8 and the raw curve is linear throughout — but a rerun
should subtract `m(0)` before testing slope constancy. Logged rather than
quietly re-run, since the fix changes sampling density, not any finding.


---

# 15. Correction: the headline was wrong

The R^2 = 0.9997 linearity was challenged on the grounds that it is built into
the measurement. **That challenge is correct and I should have caught it.**

Injecting `alpha*v` and then measuring the projection onto `u` gives
`<h0,u> + alpha*<v,u>` — linear in alpha *by construction* unless the layers
between L7 and the band substantially transform it. The residual stream is
largely additive, so much of the injection arrives untouched. Four nines is the
signature of a pass-through, not a discovery about the model. Reporting it as
the central finding was an error of interpretation, not of code.

## 15a. The behavioural test — which I had already collected

The per-trial file (§7) had the answer without further GPU time. **Behaviour
does not share the projection's linearity.**

Over alpha 24 -> 64, projection moves 27.15 -> 71.37 (+44.2) while the concept
logit moves only 15.39 -> 18.06 (+2.7). Fitted over the coherent window:

| measure | slope | R^2 |
|---|---|---|
| projection | 1.0884 | 0.999832 |
| concept logit | 0.5539 | 0.988679 |

The logit's *derivative* traces an inverted U — +0.65, rising to +0.87 near
alpha≈9-11, then falling to +0.22 by alpha=24. That is compressive saturation,
a mild S, and it is **not a step**: the derivative varies smoothly by ~4x with
no discontinuity and no sign of bistability.

## 15b. The model was driven into garbage, and I fitted a line through it

Entropy rises monotonically 4.04 (baseline) -> 5.95 (alpha=24) -> 6.63
(alpha=64). Modal top-1 degrades `' The'` -> `' the'` -> `' i'` -> `' you'` ->
`' '`. Generated text confirms it:

```
a=  0: 'The fishermen were all dressed in their best clothes...'
a=  8: '...they looked like a bunch of old men. They had been out on the sea for'
a= 16: '...i saw one big spider that'
a= 24: 'i saw a big spider on one of them and it had 8 legs, but when i looked closer'
a= 48: 'spider web, you are not a spider that is so big and scary . i ' s like'
a= 64: "not a no, you ' s . spider web is that ' like it ' ' '"
```

**The upper half of the original sweep (alpha > ~32) is uninterpretable.** The
rerun restricts the dense range to [-6, 24], the coherent window.

One genuinely positive result sits in there: at alpha 16-32 the injection
produces *coherent* spider content with the cluster co-activating — "a big
spider ... it had 8 legs" carries spider, legs and eight together. That is real
semantic steering, and the qualitative transition in the generated text happens
over roughly alpha 8 -> 24 while the projection is linear throughout. So the
projection instrument was looking in the wrong place.

## 15c. The degraded-readout finding is definitionally true

Thresholding a monotonic signal produces a threshold; that is what thresholding
is. Binarising into "is the concept winning" yields a sigmoid whose steepness is
set by trial variance — signal detection theory, and the reason psychometric
functions are sigmoid without anyone inferring an all-or-none process. **As it
stands this demonstrates the standard critique rather than advancing it.**

The version that would be worth making is quantitative: take published human
ignition curves, match trial count and noise level, and ask whether the induced
sigmoid is *as steep as theirs*. If a linear process plus realistic measurement
noise reproduces the observed steepness, that is a real claim. It would still
not touch the stronger evidence — bimodal single-trial amplitude distributions,
and late components jumping while early ones scale.

## 15d. Common mode is a pass-through signature

Injection raises unrelated control tokens (29.47) *more* than the prompt's own
content words (27.18). That is common-mode bias, not selective workspace entry,
and it is consistent with the pass-through reading rather than with anything
semantically specific happening in projection space.

## 15e. Standing of the whole result

Stage 2 failed at 500 and 1500 prompts, and it was the **only** positive
validation the lens had. The first fit was pure NaN and saved and loaded
silently. So: an unvalidated instrument produced a near-tautological linear
result, on a model driven out of its coherent range over half the sweep.

**Corrected bottom line.** No evidence of ignition — no discontinuity, no
bistability, in either projection or behaviour, over the range where the model
still works. But this is a *weak* null: the instrument is unvalidated, the
projection measure is near-tautological, and the common-mode structure says the
intervention is not semantically selective in the space being measured. The
behavioural channel is the informative one, and it shows smooth compressive
saturation with a qualitative text transition around alpha 8-24.

## 16. Final accounting

| item | value |
|---|---|
| pod 1 (fit + stage 2 + stage 3) | 1.075 h, $0.80 |
| pod 2 (behaviour + corrected rerun) | 0.140 h, $0.10 |
| network volume | deleted |
| **total** | **$0.91** (balance $20.00 -> $19.09) |
| API keys | all 3 revoked, verified 401; local copy removed |
| pods running | none |

---

# 17. Stage 4 — per-prompt trajectories (the test that could actually answer it)

## 17a. Why the design changed

The first version of this test pooled margins across prompts at each alpha and
ran Hartigan's dip on the pooled distribution. That was wrong, for the reason
this whole project exists to expose: **if entry is threshold-like, each prompt
has its own threshold**, jittered by content and by whatever else is already in
the workspace. A population of stepped units with jittered thresholds pools
into a smooth unimodal distribution. A pooled "unimodal" verdict would have
been *uninformative*, not negative — the EEG-averaging artefact running in
reverse, built into the primary test of an experiment about that artefact.

Because decoding is deterministic, each prompt can be tracked across alpha as
its own curve, at no extra cost. So the primary analysis is per-prompt; the
pooled dip is demoted to a secondary check.

## 17b. Design

- **Margin**, not rank: `logit(concept) - max logit over all other tokens`.
  Rank is already discretised and would manufacture the structure being tested.
- **Variance from prompts, not seeds**: greedy decoding is deterministic, so N
  seeds at fixed alpha give N identical numbers. 250 prompts, one trial each.
- **Held out from the fit**: the script asserts `pool[:1500] == fit_prompts`
  before proceeding. Assertion passed; no fit prompt was reused.
- **alpha=0 recorded per prompt** so curves are normalised to the *change*,
  not the level (prompts differ in baseline concept affinity).
- 25 alphas over [0, 24] — the coherent window from §15b.
- `random_matched` at every alpha as a null.

Discriminator, validated on synthetic ground truth before looking at any data:

| ground truth | width / swept range | step index |
|---|---|---|
| sharp step (k=3 at alpha=11) | 0.08 | 10.86 |
| smooth saturation | 0.97 | 2.71 |

Criterion fixed in advance: >50% of prompts with width < 25% of range = stepped.

## 17c. Result — smooth saturation, 0 of 250 prompts stepped

| | concept | random_matched |
|---|---|---|
| width / swept range (median) | **1.37** (IQR 0.97–2.36) | 0.39 |
| step index (median) | **1.81** (p90 2.53) | 4.17 |
| alpha50 across prompts | median 3.82, SD 8.14 | median 17.60 |
| prompts with sharp transition | **0%** | 27% |
| responded (rise > 0.5) | 250/250 | 185/250 |

**Not one prompt out of 250 steps.** The median transition is *wider than the
entire swept range*, and the step index sits below the smooth-saturation
reference (1.81 vs 2.71). It is not a failure to drive the prompts: every one
responded, median total rise 15.0 margin units, minimum 7.06.

**The control validates the test's sensitivity in the useful direction.**
`random_matched` shows *sharper* apparent transitions than the concept
condition — 27% "sharp", step index 4.17. That is because it barely moves the
margin at all (mean rise ~2.2 vs 15.2), so fitting a logistic to what is
essentially noise yields spurious sharpness. The statistic **over-detects**
steps on flat noisy curves, and the concept condition — which responds
strongly — still comes out smooth. That makes the null stronger, not weaker.

## 17d. Pooled dip — secondary, and null as expected

Hartigan's dip **never rejects unimodality at any alpha** (all p >= 0.20; min
p = 0.200 at alpha=24). Silverman is marginal at alpha 19-21 (p = 0.045-0.055)
and GMM dBIC turns negative from alpha≈14 (to -32 at alpha=23), but the minor
mixture weight is only ~0.15 — a tail, not a second mode. Per §17a this test
could not have carried the conclusion in either direction.

## 17e. Heterogeneity vs threshold

Spearman(margin@0, margin@24) = 0.423; mean |rank change| = 60.4 against ~83
for a full reshuffle. So baseline prompt heterogeneity is substantial and
partially persists, with real reordering on top — but since there are no modes,
this describes spread, not mode-crossing. 61% of prompts are strictly monotone
in alpha; the rest wobble slightly, consistent with noise on a smooth rise.

## 17f. The concept almost never wins

Only **35 of 250 prompts** ever reach margin > 0 anywhere in the sweep, and the
median margin at alpha=24 is **-2.25**. Even at the top of the coherent range
the injected concept usually does not become the top token — consistent with
§15's finding that projection-space "presence" is not accompanied by report.

## 17g. What this does and does not license

**Does:** within the coherent range of this model, with this lens, the
behavioural response to continuous injection is smooth compressive saturation
at the level of individual prompts. No threshold, no bistability, no stepping —
and this is not an artefact of pooling, because nothing was pooled.

**Does not:** this is *not* "no ignition in Qwen2.5-1.5B". Stage 2 failed at
both 500 and 1500 prompts and was the only positive validation the lens ever
had. The first fit was silently all-NaN. Two headline findings have been
withdrawn (§15). A clean null from an unvalidated instrument is a fact about
the setup, not about the model. The honest next step is a model and lens where
Stage-2-style validation passes *first*; only then does a null on stepping mean
anything about cognition.

# 18. Final accounting

| pod | purpose | time | cost |
|---|---|---|---|
| 1 | fit, Stage 2, Stage 3 | 1.075 h | $0.80 |
| 2 | behaviour + corrected rerun | 0.140 h | $0.10 |
| 3 | Stage 4 | 0.374 h | $0.28 |
| | | **1.59 h** | **$1.18** |

Balance $20.00 -> ~$18.82. All pods destroyed. Volumes and keys: see below.

---

# 19. Overnight parallel run

## 19a. Two constraints the brief could not remove

**Budget is a hard cap at ~$18.81, not "not a constraint".** The instruction
was to top the balance up if needed. I cannot: entering payment details is
something I will not do on someone's behalf, and Auto-Pay is off (correctly --
it was found silently enabled at $200-when-below-$100 and disabled at the
user's request). So tonight is planned to fit the standing balance, and job
sizing below reflects that. This is flagged rather than silently absorbed,
because it changes what the pre-committed refit rule can afford.

**`phase3-confinement-brief.md` does not exist on this machine.** Searched the
filesystem and Spotlight; no such file. Jobs 3 and 4 were specified inline in
enough detail to proceed (skill count, verifiability, instance count, the
pilot's three conditions, matched-drift baseline, full-rank-only, three
collateral levels), so I built to the inline spec and recorded every assumption
here. Anything in §2 of that file that is *not* in the inline restatement was
not implemented, because I have never seen it.

## 19b. The scorer fix (gates everything in Job 1)

Rewritten to score **fact identity, not token string**. The model is now asked
to continue greedily for a few tokens and the continuation is checked for the
fact, case- and leading-space-insensitively, with an alias table
(`yuan|renminbi|rmb|cny`, `beijing|peking`, `chinese|mandarin`, ...). Rank is
kept as a secondary measure, taken as the best rank over the accept-set rather
than a single id.

Unit-tested against last run's two actual failures before use:

| input | target | old | new |
|---|---|---|---|
| `' euro'` | Euro | WRONG (case) | correct |
| `' renminbi'` | Yuan | WRONG (alias) | correct |
| `' Paris'` | Beijing | correct | correct (still rejects) |
| `' English'` | Chinese | correct | correct (still rejects) |

11/11 unit tests pass, including the negative cases -- the scorer had to stay
capable of saying no.

**Baseline validation gate**: if the unintervened model does not score 4/4, the
run emits `SCORER_OR_MODEL_FAILS_BASELINE` and the swap numbers are explicitly
marked uninterpretable. A scorer that cannot read an unintervened model cannot
be trusted on an intervened one.

## 19c. Infrastructure notes

Two A100 80GB pods were allocated `RUNNING` but never received a public IP or
port mappings -- dead on arrival, billing at $1.19/hr with no way in. Both were
detected and destroyed (~$0.65 total wasted). The provisioner now searches
GPU x cloud x datacenter for capacity instead of failing on first miss, and
network volumes are optional (they pin the datacenter, which was blocking
allocation). 7B moved to a 4090: 14 GB of bf16 weights fits in 24 GB, and it is
both cheaper and far more available than an A100.

## 19d. Job 3 corpus design

Every skill is an **invented formal system**. This is forced by the
fail-without requirement: an instruct model already knows base64, ROT13 and
ISBN checksums, so a real procedural skill has nothing to install and
contributes only noise. Arbitrary-but-mechanical rules guarantee the
fail-without half by construction while keeping pass-with a clean test of
whether the procedure is available.

Eight skills, 24 held-out instances each: `tessera` (glyph substitution +
reversal), `veridian` (weighted checksum mod 23 in hex), `kelvan` (5x73-day
calendar), `rotor3` (position-dependent cipher), `mirror` (right-to-left
precedence), `hollow` (fixed-width packing), `drift` (sort by last letter then
length descending), `span` (invented unit chain). All ground truth verified by
hand against the written spec, including each spec's worked example.

`mirror` asserts at generation time that the MIRROR answer differs from the
standard-precedence answer for every instance, so a model applying ordinary
rules scores 0 rather than accidentally agreeing.

**Both conditions receive the identical output-format instruction**; only the
procedural content differs. Without that control the empty-context arm would
fail partly on formatting and the gap would be inflated by an artefact.

## 19e. Job 3 RESULT — skill corpus built and gap-verified

Measured on Qwen2.5-7B-Instruct, 24 held-out instances per skill, both
conditions given the identical output-format instruction.

| skill | n | fail-without | pass-with | gap | verdict |
|---|---|---|---|---|---|
| beacon | 24 | 100.0% | 100.0% | +100.0% | KEEP |
| crate | 24 | 100.0% | 100.0% | +100.0% | KEEP |
| hollow | 24 | 100.0% | 100.0% | +100.0% | KEEP |
| stamp | 24 | 100.0% | 100.0% | +100.0% | KEEP |
| ledger | 24 | 100.0% | 95.8% | +95.8% | KEEP |
| tally | 24 | 100.0% | 37.5% | +37.5% | KEEP |
| sigil | 24 | 100.0% | 33.3% | +33.3% | KEEP |
| span | 24 | 100.0% | 16.7% | +16.7% | KEEP |
| kelvan | 24 | 100.0% | 8.3% | +8.3% | KEEP |
| drift | 24 | 100.0% | 4.2% | +4.2% | KEEP |
| mirror | 24 | 100.0% | 0.0% | 0.0% | DISCARD |
| rotor3 | 24 | 100.0% | 0.0% | 0.0% | DISCARD |
| tessera | 24 | 100.0% | 0.0% | 0.0% | DISCARD |
| veridian | 24 | 100.0% | 0.0% | 0.0% | DISCARD |

**10/14 kept. Five skills clear +95%.** Widest: `beacon` (and `crate`,
`hollow`, `stamp` tie at 100%); `beacon` is the pilot skill for Job 4 -- it has
the simplest mechanics of the four, so a Phase 3 failure there cannot be blamed
on execution difficulty.

### A design correction the first cohort forced

The first eight skills were calibrated for *unknowability* but not
*executability*. `rotor3`, `tessera` and `drift` scored 0% in **both**
conditions: a 7B cannot reliably run a position-dependent cipher even with the
rules in front of it. A skill the model fails even with the file installs
nothing -- exactly as useless as one it already passes, and the discard rule as
written only catches the second case. A second cohort (`beacon`, `crate`,
`ledger`, `sigil`, `stamp`, `tally`) keeps the arbitrary convention and makes
the mechanics trivial, which is what produced the five wide gaps.

The failure mode is visible in the raw outputs and is a genuine knowledge gap,
not a formatting artefact:

```
hollow  want '930::0050'  without 'HOLLOW-930-50'    with '930::0050'
hollow  want '417::0059'  without 'HOLLOW(417, 59)'  with '417::0059'
```

The model understands the task without the file and invents its own convention.

`mirror` is worth a footnote: it scored 0% with the spec because the model
applies standard precedence *regardless* of being told not to. That is a
reasoning failure rather than a knowledge gap, so it is discarded -- but it is
the most interesting of the discards.

Generator bug found and fixed: `tally` could produce only 6 distinct instances
against `N_INSTANCES=24`, so the dedup loop span forever. There is now an
explicit attempt cap that raises a named error instead of hanging.

## 19f. Job 1b RESULT — **14B GATE PASSED**

Qwen2.5-14B-Instruct, 3000-prompt restricted fit, bf16, finiteness guard,
0 prompts skipped. Cross-check against the full Jacobian: **worst cosine
0.999914** across all 17 fitted layers (L12 + band L16-L31).

```
capital    base ' Paris. It is located in'      swap ' Beijing. (True or False'   rank_dst 101->1
language   base ' French, but there are also'   swap ' Chinese．____（判断对'      rank_dst  72->1
continent  base ' Europe. It is the largest'    swap " Asia.\nNo, that's"         rank_dst  75->1
currency   base ' euro. The euro was introduc'  swap ' yuan. A. Correct B'        rank_dst 262->1
```

**baseline 4/4 | swapped 4/4 | moved 4/4 -> VERDICT: PASS**

This is the first validated lens in the project. All four facts shift together
and every target rank collapses to 1.

The scorer fix is load-bearing here: the baseline `' euro'` and the swapped
`' yuan'` are exactly the two cases the old token-string scorer marked wrong.
Under the old scorer this same run would have scored baseline 3/4 and been
capped below PASS -- the 1.5B "failure" may always have been partly scorer
artefact, though that cannot now be separated from model size.

### A costly bug caught mid-run

`stage2_verify.py` derived the data path from `JLENS_OUT` (`/workspace/out`),
so on the pods it looked for `/workspace/jacobian-lens` while the code sat in
`/root/jlens`. Stage 2 crashed with `FileNotFoundError`, the runner read the
**empty** verdict as a failure, and started the pre-committed 6000-prompt refit
-- roughly 4 hours of H100 time (~$13) to re-fit a lens that had never been
tested. Caught at ~50 prompts in and killed; the 3000-prompt lens was intact on
disk and passed on first evaluation.

Two fixes: the path now derives from the code root, and **an empty verdict
aborts instead of counting as failure**. A crash must never be able to consume
the one retry the pre-commitment allows.

## 19g. Job 1a RESULT — **7B GATE ALSO PASSED**

```
capital    ' Paris. It is located in'    -> ' Beijing. (True or False'  rank_dst 108->1
language   ' French, but there are some' -> ' Chinese. （改为否定句'     rank_dst  93->1
continent  ' Europe. It has an area'     -> ' Asia. It is located in'   rank_dst  46->1
currency   ' euro, but before that it'   -> ' yuan (元, yu'             rank_dst  60->1
```
**baseline 4/4 | swapped 4/4 | moved 4/4 -> PASS.** Both hedged models validated.

The 7B hit the *same* path bug as the 14B, because the fix was pushed to the
14B pod while the 7B's job was already in flight with the old code. It burned
another needless refit start before being caught. Both were killed; both
3000-prompt lenses were intact and passed on first evaluation.

## 19h. Job 2 RESULT — the report measure works, but only once the range is right

**First run (14B, alpha 0-24) was inconclusive, not a null.** Only 2/80 prompts
ever reported and the margin at alpha=24 was still -14.1, nowhere near 0. I had
carried over alpha_max=24 from the *1.5B* without re-deriving it: the alpha unit
is model-specific (1.5B 0.76, 14B 1.82, 7B 9.82), so 24 units means something
completely different per model. Reporting that as "no report" would have been
the same error as the original top-1 measure -- a measure that could not fire.

**Second run (7B, alpha 0-120)** with the range derived for that model:

| alpha | report/3 | any-hit | margin | contrast (piano) | random control |
|---|---|---|---|---|---|
| 0 | 0.017 | 1.7% | -19.94 | 0.0% | 1.7% |
| 10 | 1.233 | **81.7%** | -1.28 | 0.0% | 1.7% |
| 20 | 1.000 | **100%** | +1.71 | 0.0% | 0.0% |
| 60 | 1.000 | 100% | +3.33 | 0.0% | 0.0% |
| 120 | 1.000 | 100% | +2.99 | 0.0% | 0.0% |

**All 60/60 prompts report.** The margin crosses zero between alpha 10 and 20.
`random_matched` never reports at any alpha. The cluster-B contrast probe
("What instrument is it?") is 0% everywhere -- the injection is *specific*, not
generalised disruption.

This is the first time in the project that a report measure has fired at all.
The whole Stage 4 null was measured with an instrument that never reached the
regime it was meant to test.

**But the grid cannot resolve the onset shape.** With a step of 10, the entire
transition happens between two grid points, and the per-prompt onset IQR is
[10,10] -- that is grid quantisation, not a measured sharp threshold. A dense
sweep over alpha 0-20 (step 1) is running to characterise it.

## 19i. Job 4 RESULT — **NO. Matched-drift fine-tuning is cleaner.**

Pilot: skill `beacon` (widest gap, simplest mechanics), 12 train / 12 held-out
instances, full-rank SGD, no LoRA, one update magnitude, three collateral
levels. Baseline `beacon` without the file = 0%.

| condition | ‖Δθ‖ | skill installed | repr. drift (J-space) | dist. KL | other-skill loss |
|---|---|---|---|---|---|
| targeted_penalty_on | 0.1191 | 75% | **0.0931** | 0.0142 | 0 pp |
| targeted_penalty_off | 0.1699 | 100% | 0.0435 | 0.0122 | 25 pp |
| random_null | 0.1797 | 92% | 0.0503 | 0.0161 | 25 pp |
| **ft_matched_drift** | **0.1157** | **100%** | **0.0239** | **0.0089** | 25 pp |

LR bisection converged in 5 iterations to within **2.9%** of the target drift
(0.1157 vs 0.1191), comfortably inside the 10% tolerance. Matched on drift,
never on steps.

**Answer to the morning question: no.** At matched drift, ordinary fine-tuning
installs the skill *fully* (100% vs 75%) and is cleaner on **both** collateral
axes -- 3.9x less representational drift (0.0239 vs 0.0931) and 1.6x less
distributional KL. The confinement penalty made things actively worse on both:
it *reduced* install and *increased* J-space collateral, which is the opposite
of its purpose.

### Caveats that materially limit this

1. **Only the `targeted_penalty_on` <-> `ft_matched_drift` pair is drift-matched.**
   The other two targeted conditions drifted ~50% more (0.170, 0.180 vs 0.116),
   so their collateral numbers are inflated and must not be read against FT
   directly. The matched pair is the only clean comparison -- and it is the one
   that goes against the hypothesis.
2. **n=1 skill, 1 magnitude, 1 seed.** A pilot, as scoped.
3. The other-skill collateral measure is coarse (8 instances each). `tally -25pp`
   appears in three conditions *including* plain FT, so it is probably noise
   rather than a condition effect.
4. **The mechanism is my interpretation.** `phase3-confinement-brief.md` was
   absent, so "workspace-targeted install" and the confinement penalty were
   implemented from the inline description. A negative result on my
   reconstruction is weak evidence against the actual intended method. This is
   the caveat that most limits the finding.

### A first run that had to be discarded

The initial pilot used lr=2e-5 with plain SGD and produced ‖Δθ‖=0.0001 and 0%
install in **every** condition -- nothing happened at all. Reporting that as
"the target was not reached" would have been misleading: grad norms were clipped
to 1.0 and the mean was ~0.125, so 40 steps at 2e-5 could not move the weights
by construction. That is a configuration error of mine, not a property of the
method. The rerun at lr=0.02 (chosen to target ‖Δθ‖~0.1 from the measured grad
norm, not swept for a favourable outcome) produced real drift and real installs.
The null run is kept as `phase3_pilot_7b_lr2e5_nulldrift.json`.

## 19j. Dense report sweep — the transition resolved

7B, alpha 0-20 step 1.0, 60 held-out prompts:

| alpha | 4 | 5 | 6 | 7 | 8 | 10 | 13 | 20 |
|---|---|---|---|---|---|---|---|---|
| any-hit | 10% | 25% | **63%** | 80% | 87% | 82% | 100% | 100% |
| margin | -12.3 | -9.7 | -6.9 | -4.7 | -3.3 | -1.3 | +0.2 | +1.7 |

Per-prompt onset: **median 6.0, IQR [5.8, 7.0], SD 2.00**; 43% of prompts gain
>=2 of 3 probes within a single alpha step of 1.0.

So the population transition is concentrated: most of it happens between alpha 5
and 7, while the margin over that same interval rises smoothly and near-linearly
(-9.7 -> -6.9 -> -4.7). Onset jitter across prompts is modest (SD 2.0 units).

**This is NOT being reported as ignition, for the reason that sank the earlier
text-transition claim.** Report is a *thresholded* readout -- the model either
emits the concept word or it does not -- so a sharp per-prompt transition is
partly guaranteed by discreteness, exactly as argmax was. A continuous quantity
rising smoothly past a fixed criterion produces a sharp discrete transition with
no threshold anywhere in the underlying process.

The test that would separate them, and which this run does not do: predict each
prompt's report onset from its *own* margin curve crossing a fixed criterion,
and ask whether observed onsets are sharper, or more tightly clustered, than
that simple thresholding model predicts. The data to do this is in
`stage5_report_7b_dense.jsonl.gz` (per-prompt margin and report at every alpha).
That is the first thing to run next.

---

# 20. Corrections after review

## 20a. Job 4 is **NOT YET TESTED**, not falsified. Retracting the "no".

Three defects, any one of which is disqualifying:

**1. Asymmetric optimisation effort.** The fine-tuning baseline got an
eight-iteration LR bisection searching toward a drift target. The confinement
arm got **one hand-computed LR**. A searched arm beating an unsearched arm says
nothing about the methods -- it says one arm was tuned. Any comparison requires
equal search budget on both.

**2. Install success was not matched.** targeted_penalty_on reached 75%,
FT reached 100%. Collateral compared at matched *drift* but unmatched *install*
is not a comparison: a partially installed skill has spent its drift budget
differently, and the cost of the missing 25% is unknown -- it could be most of
the collateral. The correct design matches on **install success**, then compares
drift and collateral.

**3. n=1 skill, 1 magnitude, 1 seed.** The brief specifies 4-5 magnitudes
because the quantity of interest is a **slope**, not a point. One point cannot
distinguish "confinement does not work" from "confinement does not work at this
operating point".

The numbers in §19i stand as recorded, but the **conclusion drawn from them is
withdrawn**. Correct status: *not yet tested*.

Compounding all three: the mechanism was my reconstruction. The real brief
exists (in the conversation, not on this filesystem) and has been requested.

## 20b. The Stage 2 gate criterion was relaxed after it failed — logged as such

Honest sequence: the 1.5B **failed** the gate at 500 and 1500 prompts; I then
**modified the scorer**; the 7B and 14B then **passed**.

The scorer changes are defensible on their own terms -- `' euro'`/Euro and
`' renminbi'`/Yuan were genuine bugs, the fix was specified before seeing new
results, and the negative controls (`' Paris'` for Beijing, `' English'` for
Chinese) still correctly fail, so it did not become a scorer that says yes to
everything.

**But fail -> modify criterion -> pass is weaker evidence than a criterion fixed
in advance, and it should not be reported as though the gate were pre-committed.**
Two changes were confounded in one step: scorer *and* model size. So:

- Whether the 1.5B would now pass under the fixed scorer is **untested**.
- How much of the 7B/14B pass is attributable to the scorer vs to scale is
  **unknown**.

Cheap resolution: re-run Stage 2 on the 1.5B lens with the current scorer. That
single number separates the two. Recorded as an **open question**, not resolved.

## 20c. Coherence over Job 2's alpha range was never checked

§19h reported report rising 0% -> 100% without verifying the model was still
coherent at those alpha. That is the exact failure from the Stage 3 run ("fitting
a line through wreckage"). The piano contrast probe does **not** cover it:
wreckage caused by a spider injection would be spider-flavoured wreckage,
scoring 0% on piano while being degenerate. Specificity and coherence are
different properties. Coherence check running.

## 20d. Coherence check RESULT — the top of Job 2's range is wreckage

First attempt used the wrong alpha normalisation: it derived the unit from the
20 neutral sweep prompts (45.47) while stage5 derived it from held-out wikitext
(9.82) -- a factor of 4.6, testing a completely different injection strength than
the run it was meant to validate. Corrected to use the same pool (11.35).

7B, alpha unit 11.35, baseline entropy 1.58, baseline repeat-fraction 16.8%:

| alpha | H | KL | repeat | sample |
|---|---|---|---|---|
| 0 | 1.58 | 0.00 | 17% | `British Royal Navy was able to take advantage...` |
| 4 | 2.62 | 1.97 | 15% | `The situation was so bad that the French government hired British sailors...` |
| **6** | 4.16 | 5.36 | 14% | `walls of their spider webs. the spiders were so numerous that they could be seen all over the countryside` |
| **8** | 4.66 | 7.54 | 27% | `spider web of the city streets. other spiders, especially the ones that were not visible, could be found craw` |
| 10 | 4.88 | 8.67 | **50%** | `spiders, spiders, and other ar ar ar spiders` |
| 13 | 5.26 | 9.49 | **84%** | `spider spiders, for spider spiders, spider spiders...` |
| 16-20 | — | — | **94-97%** | `spider spider spider spider spider spider...` |

**Coherent window is alpha <= 8** (repeat-fraction is the better degeneracy
marker than entropy; entropy necessarily rises under injection).

Consequences for §19h, which must be partly withdrawn:

- **The "100% report, all 60/60 prompts" figure is wreckage** and is withdrawn.
  It comes from alpha >= 13, where output is 84-97% pure repetition. The probes
  were scoring spider-flavoured degeneracy that happened to contain the target
  words.
- **The rise from 1.7% to 86.7% over alpha 0-8 survives** and is real. At
  alpha=6 the model writes coherent English deploying the concept
  ("the spiders were so numerous that they could be seen all over the
  countryside"), repeat-fraction 14%, *below* baseline.

The piano contrast probe did not catch this, exactly as predicted: wreckage from
a spider injection is spider-flavoured, so it scores 0% on piano while being
degenerate. Specificity and coherence are independent properties and need
independent controls.

### Thresholding test, redone inside the coherent window (alpha <= 8)

| | median onset | SD |
|---|---|---|
| observed | 6.0 | 1.48 |
| margin-threshold predicted | 6.0 | 1.32 |

Spread ratio **1.12**; a single global margin criterion predicts report at 83.3%
vs a 69.6% baseline. 46/60 prompts report within the coherent window.
**Still consistent with pure thresholding of a smoothly rising margin** -- the
conclusion is unchanged by restricting to coherent alpha, which is worth
knowing, since it could easily have gone the other way.

## 20e. **The 1.5B passes with the fixed scorer. The original failure was the scorer, not the model.**

```
capital    ' Paris. The French language ' -> ' ________．\nA.'          rank_dst 137->7
language   ' French, but there are also'  -> " Chinese, but they don't"  rank_dst 260->1
continent  ' Europe. It has been a'       -> ' Asia. It has a population' rank_dst  11->1
currency   ' euro. The euro was introduc' -> ' Renminbi. The Chinese'     rank_dst  42->1
```
**baseline 4/4 | swapped 3/4 | moved 4/4 -> PASS**

This resolves the §20b confound in the direction I did not expect, and it
matters a great deal:

- **Scale was never required.** All three models pass. The 1.5B "failure" at 500
  and 1500 prompts was entirely an artefact of token-string scoring.
- **The Stage 3 and Stage 4 results were not produced by an invalid instrument
  after all.** They used `qwen15b_3x`, which is the exact lens just tested. The
  lens was fine; the *validation* was broken. My earlier framing -- "an
  unvalidated instrument produced a suspiciously perfect result" -- was too
  harsh on the lens and not harsh enough on the scorer.
- The Stage 4 null (smooth saturation, 0/250 prompts stepping) therefore stands
  on a lens that **does** pass validation.

The "fail -> modify criterion -> pass" concern in §20b is still correct as a
matter of epistemic hygiene, and the sequence is still worth flagging. But the
specific worry it raised -- that the relaxation manufactured the pass -- is now
testable and answered: the criterion change was a bug fix, and it recovers a
model that had genuinely satisfied the *original* criterion's intent all along.

## 20f. Corrections — Stage 4 status, 1.5B pass strength, ignition status

**§20e over-claimed, in the same way three previous claims over-claimed.** I
wrote that the Stage 4 null "stands on a lens that does pass validation", as
though validating the lens strengthened the null. It does not.

**Lens validity and window validity are independent failures.** Stage 4 swept
alpha [-6, 24] on the 1.5B with **no repeat-fraction gate**. The 7B degenerates
into pure repetition by alpha≈8; the 1.5B ceiling is likely *lower*, not higher.
So the Stage 4 sweep may sit largely over degeneracy — precisely the failure
that forced the withdrawal of the Job 2 headline. Fixing the lens does not touch
this. **One of two defects was resolved. The null is not stronger.**

Status: Stage 4 null is **untested** pending measurement of the 1.5B coherence
ceiling over its actual sweep range.

**The 1.5B gate pass is WEAK and is recorded as such.** Swapped **3/4**, against
**4/4** on both the 7B and the 14B — and 3/4 became the passing threshold only
after the criterion was revised. It clears the bar; it clears it by the minimum,
under a bar that moved. It should not be cited as equivalent evidence to the
7B/14B passes.

**Ignition status: OPEN, NOT TESTED.** The spread-ratio result confirms the null
without testing the alternative; no power analysis had been done. Retrospective
power (§`ignition-criterion.md`): the test could only reject onsets **>=23%
tighter** than thresholding — a resolution set by n=60, never derived from any
model of ignition. Criterion document written before further ignition work, per
instruction.

## 20g. Stage 4 window validity — TESTED. The sweep does **not** sit over degeneracy.

1.5B, lens `qwen15b_3x`, alpha unit 1.766, over Stage 4's actual window:

| alpha | -6 | 0 | 4 | 8 | 12 | 16 | 20 | 24 |
|---|---|---|---|---|---|---|---|---|
| **repeat-frac** | 8.1% | 8.8% | 13.1% | 13.4% | 12.0% | 11.8% | 13.0% | 11.3% |
| entropy | 1.33 | 1.78 | 2.01 | 2.50 | 3.27 | 4.02 | 4.44 | 4.51 |
| KL | 0.28 | 0.00 | 0.09 | 0.41 | 1.63 | 3.25 | 4.74 | 5.78 |

**Repeat-fraction — the marker that caught the 7B collapse — stays flat at
8-13% across the entire window.** It never approaches the 50-97% degeneracy
that forced the Job 2 withdrawal. Text at alpha=24 is still grammatical:
`"web. but not all spiders were so bad. they would be out for a long time..."`

So the concern was correct to raise and the answer is empirical: **Stage 4's
window was not over degeneracy.** The 1.5B tolerates its whole sweep range.

Two qualifications, so this is not over-read the way §20e was:

1. The automated flag still said "coherent up to 8" because it thresholds on
   entropy, which rises 1.33 -> 4.51. Entropy *necessarily* rises under
   injection and the 1.5B baseline is unusually low, so entropy is the wrong
   gate; repeat-fraction is the one that discriminates. The flag is
   conservative, not wrong — but the verdict here comes from repeat-fraction.
2. The model is still meaningfully perturbed at the top end (KL 5.78, register
   drifts to lowercase/informal). "Not degenerate" is not "unaffected".

**Net: Stage 4 had two suspected defects. Both are now resolved — lens validated
(§20e), window validated (here). Neither resolution touches objection 1: the
test's power was never characterised.** Stage 4's null is therefore a *clean
null with unknown power*, not a falsification. Ignition status remains **OPEN**
per `ignition-criterion.md`.

## 20i. Ramp test underpowered on the 1.5B as well — failure condition fired twice

1.5B, alpha 0-20 (inside its measured coherent range, §20g), n=200:
**asymmetric 6 usable, triangle 3 usable.** The 1.5B tolerates high alpha
without degenerating, but rarely *reports* below alpha=20, so there are almost
no ramps where the concept appears on both halves.

Two models, two different limiting reasons, same outcome:

| model | coherent to | report onset | usable triangles |
|---|---|---|---|
| 7B | alpha ~8 | at/above the ceiling | 6/200 |
| 1.5B | alpha ~24+ | above 20, rarely fires | 3/200 |
| 14B | alpha ~50 | ~40 (measured) | pending |

**The pre-committed failure condition has now fired on two of three models.**
The problem is not the coherence ceiling per se; it is that the gap between
"concept reports at all" and "model degenerates" is too narrow to fit a
symmetric ramp with both halves populated.

### 14B alpha range, measured rather than assumed

Three sweeps in a row were under-ranged because alpha_max was carried across
models. Measured directly on the 14B (alpha unit 1.816):

| alpha | 0 | 20 | 40 | 50 | 60 | 100 | 120 |
|---|---|---|---|---|---|---|---|
| repeat | 11.5% | 19.6% | 15.5% | 33.9% | 42.1% | 63.2% | 87.2% |
| content | baseline | baseline | `the name means "spider mon...` | degrading | URL spam | repetition | collapse |

**Coherent to alpha ~50, concept content from ~40.** The earlier 14B Job 2 run
(alpha 0-16, margin only -20.7 -> -16.2, report 1.2-2.5%) was under-ranged and
is void — kept as `stage5_report_14b_underranged.jsonl.gz`. Re-running over
0-50, the measured window.

**Process fix**: alpha_max must be *derived per model* from a coherence probe,
never carried over. Every model has a different alpha unit (0.76 / 1.77 / 9.82 /
1.82 across the four tested) and a different degeneracy ceiling.

## 20k. Ramp/hysteresis test — FAILS ON ALL THREE MODELS, for a structural reason

| model | coherent to | usable triangles / 200 |
|---|---|---|
| 7B | ~8 | 6 |
| 1.5B | ~24 | 3 |
| 14B | ~50 | **2** |

The 14B result is the informative one, because it *refutes my own prediction*.
Job 2 showed 38.8% report at alpha=50 (§20j), so I expected ~35 usable ramps.
It returned 2. The reason is structural, not statistical:

> **A ramp exposes the model to high alpha only briefly. Report requires
> *sustained* drive.** In the constant-alpha sweep the model sits at alpha=50 for
> the whole generation and 38.8% of prompts report. In a ramp it passes through
> alpha=50 at a single token near the apex, and almost nothing fires.

So the ramp design is self-defeating for this system: the very property that
makes hysteresis measurable (alpha changing within one generation) removes the
sustained drive that report needs.

**Verdict, per the condition fixed in advance in `ignition-criterion.md` §5:**

> "If the coherence gate cannot be held across a ramp wide enough to resolve
> 0.30 alpha units, then ignition is not falsifiable in this setup, and that is
> the result."

**Ignition is NOT FALSIFIABLE by the hysteresis route in this paradigm.**
Reported as a result about the method, not as a null about the model. The
+4.45 sigma-14.9 loop from the asymmetric ramps is *not* evidence and is not
carried forward — it measures where in the sequence high alpha arrives.

### The design that could work: step-and-hold, not ramp

The flaw is ramping. A test that keeps drive sustained on both sides:

1. Hold alpha at A_high for k tokens (long enough for report to establish).
2. Step down to A_low and hold for k more tokens.
3. Separately: hold at A_low for 2k tokens from the start.
4. Compare report rate during the second half of (2) against (3).

H0: identical — report at A_low depends only on A_low.
H1: (2) > (3) — once established, report sustains at a drive that could not
have started it. That is bistability, and it keeps drive sustained throughout,
which the ramp does not.

The primed control from §20h should be retained: it already shows context alone
(concept token forced, no injection) does *not* sustain report, so any
persistence in (2) cannot be attributed to text autocorrelation.

This is the concrete next experiment for the ignition question. Everything
needed is built; only the schedule changes.

# 21. Phase 3 — the target was under-specified by construction

## 21a. Where the skill's activation change actually lives
`diag_orth_7b.json`, 7B, `beacon`, n=12, layer 7:

| | value |
|---|---|
| ‖delta‖ inside J-space | 2.2454 (27.9% of the J-space state) |
| ‖delta‖ in the orthogonal complement | 9.2540 (29.8% of the complement) |
| **fraction of squared difference inside J-space** | **5.56%** |
| enrichment over chance (5.56% of the difference in 4.5% of the dims) | **1.23x** |

The skill's activation change is **barely concentrated in J-space at all**. Two
consequences, and the second is the larger one:

1. `L_orth` anchored to the training instances pinned **94.4%** of that change to
   the no-skill state. The objective read "move 6% toward the skill, hold 94% at
   no-skill". Target-reached-but-no-install was guaranteed. **The first sweep is
   VOID** (archived under `void/`). `L_orth` is now anchored to unrelated
   held-out prompts.
2. Re-anchoring does not touch the *target*. A J-space-only target specifies
   about a twentieth of the change.

## 21b. Is the target sufficient? Activation patching says no
`diag_sufficiency_7b.json` — supply the captured state at inference, no training:

| condition | install |
|---|---|
| no_patch | 0.0% |
| patch_J (J-space component clamped) | 0.0% |
| patch_full (**whole** activation at layer 7) | 0.0% |
| with_file | 100.0% |

Even the **full** layer-7 activation fails. It transmits the topic and never the
procedure — `'RenderIngotAsBeaconToken()'`, `'INGOTBEACONTOKEN'`, always the right
*word*, never `<<...>>`.

Two bugs were fixed to reach a valid control, both of which had produced
misleading output: the patch first assigned to `[:, :, :]`, broadcasting a
last-token vector over every position and destroying the prompt (garbage `'\\'`);
then it re-imposed the prompt-final state at every generation step, pinning one
layer to a constant and forcing `'ANSANSANS...'`. The patch now applies to the
prompt's final position only, and generation runs free.

## 21c. How much state IS needed — position, not layer
`diag_state_extent_7b.json`, n=24:

| condition | install |
|---|---|
| no_patch | 0.0% |
| L_inj_last (the Phase 3 target's exact site) | 0.0% |
| L_inj_all | 0.0% |
| band_last | 0.0% |
| **band_all** | **20.8%** |
| all_last (every layer, one position) | 0.0% |
| **all_all** | **54.2%** |
| with_file | 100.0% |

**Every single-position condition is 0%, regardless of how many layers are
copied. Every all-position condition transmits.** Copying all 28 layers at one
position gives nothing; copying the 10-layer band across positions gives 20.8%.

**HYPOTHESIS (not measured): the skill is carried in a trajectory across token
positions rather than held in a state.** What IS measured is a per-position
ceiling. No trajectory target has been tested, so this is the natural reading of
the ceiling, not a result. A point capture transmits what the task is about; only a sequence
transmits how to do it.

This is the explanation for the whole Phase 3 negative: the target was one
J-space vector at one layer at one position — 0% by construction, before any
question about confinement mechanisms arises.

Caveat kept explicit: `all_all` reaches 54.2%, not 100%, partly because the
with-file prompt is longer than the no-file prompt so positions do not align.
The all-position variants copy the last K positions, K = the shorter length.
That is an approximation and the 54.2% is a lower bound.

## 21d. What this does to the design
`L_target` becomes a **sequence match** over positions rather than a last-token
match. This is also the independent argument for the distillation framing in
THREAD-CONTEXT: soft-label distillation transfers structure across a spread
rather than a single answer, and the measured result here says a single
coordinate was always going to underdetermine the write.

## 21e. The J-space ceiling — measured, and it is zero

Per the new DECISIONS rule ("test target sufficiency before testing
mechanisms"), the J-space-restricted patching conditions bound what **any**
J-space-targeted weight update can achieve. A weight update cannot beat patching
the target in directly.

`diag_state_extent_7b.json`, 7B, `beacon`, n=24:

| condition | J-space only | full activation |
|---|---|---|
| layer 7, all positions | **0.0%** | 0.0% |
| band (L9-18), all positions | **0.0%** | 20.8% |
| all 28 layers, all positions | **0.0%** | 58.3% |
| with_file (ceiling) | — | 100.0% |

**Copying the J-space component at every available layer at every position
installs nothing.** The same sites with the full activation reach 58.3%.

So Phase 3 as specified — `L_target` in J-space — has a **measured ceiling of 0%
install**. No confinement mechanism can be compared, because no arm can install.
The negative is about the TARGET SPACE, not about confinement.

### What this says about the workspace

J-space transmits *what the task is about* and never *how to do it*. The
patched outputs are unambiguous about this: `'RenderIngotAsBeaconToken()'`,
`'INGOTBEACONTOKEN'`, `'VEL-001'` — always the right word, always the concept
"beacon" and "token", never the `<<...>>` procedure.

This is consistent with the source paper rather than against it: J-space is
characterised as the subspace that drives **report**. Report is content. A
procedure is not content, and it is not in there. For the write path this is the
central constraint: **the workspace is not where procedures live, so the
workspace is the wrong target for writing one.**

### And position beats layer, everywhere

Every single-position condition is 0% regardless of depth — including copying
all 28 layers. Every all-position condition transmits. The skill is carried in a
**trajectory across token positions**, not in a state at any depth.

## 21f. Status of Phase 3

- The mechanism comparison (soft penalty / gradient projection / verify gate vs
  matched-drift fine-tuning) is **not yet answered**, and was never answerable
  with a J-space target.
- Running now: the same sweep with a **full-activation band trajectory** target
  (ceiling 20.8%) alongside the jspace target (ceiling 0%), 4 magnitudes,
  80 steps, all arms bisected equally. The jspace arm is retained deliberately
  as the negative control whose ceiling is known.
- Fine-tuning installs throughout (58% at drift 0.06, 20 steps), so the skill is
  learnable at these budgets. Whatever the activation-matching arms do, it is
  not a budget problem.

## 21g. Residual tolerance — and an anomaly that inverts the reading

The design cold read said the headline ("matching drives L_target down 99.9% and
installs 0%, therefore state is not the carrier") could not be supported without
knowing whether the residual mattered. Measured, n=24, band L9-18, exact
captured target corrupted with isotropic noise of RMS sigma:

| sigma | 0 | 0.25 | 0.5 | 1.0 | 2.0 | 4.0 | 31.2 |
|---|---|---|---|---|---|---|---|
| install | 20.8% | **75.0%** | 70.8% | **54.2%** | 0.0% | 0.0% | 0.0% |

**Adding noise to the target improves install, by 3.6x.** The exact patch is
*worse* than a slightly corrupted one. This is not a small effect and it inverts
two earlier conclusions:

1. **The achievable ceiling is at least 75%, not 20.8%.** Every "fraction of
   ceiling" statement computed against 20.8% understates the gap.
2. **The arms' null becomes significant.** They sat at residual sigma~1.0, where
   patching installs **54.2%**, and they installed **0/24**. Even against the
   conservative 20.8% ceiling, P(0 successes | n=24) = 0.0037.

### Why noise would help — untested hypotheses
The exact patch clamps activations to a state captured from a *different*
prompt: the with-file prompt is longer, so `align_tail` matches only the final K
positions and some aligned pairs put chat control tokens against mid-file
content. A rigid clamp then forces a state that contradicts the actual context,
while noise loosens the clamp enough for the model's own computation to
reassert. **This is a hypothesis, not a result.** It is testable by aligning on
token identity rather than position offset.

### What this does to the write path
The dissociation is now sharp and, importantly, **not** explained by the target
being insufficient:

- Patching a sigma~1.0 state installs **54.2%**.
- Training weights to reach residual ~1.0 installs **0%**.

Same residual, same target, opposite outcome. So the failure is not "the state
does not carry the skill" — it demonstrably does. The failure is that **the
weight update reaches a low L_target by a route that does not produce the
functional state.** Almost certainly a degenerate solution: MSE falls if
activations shrink toward the target's mean, without matching direction.

**The next measurement, and it is decisive: report cosine similarity between the
achieved and target states, not only MSE.** If cosine is low while MSE is low,
the arms are shrinking rather than matching, and every "matching does not
install" claim in this project is void for that reason.

## 21h. Headline WITHDRAWN before publication

The intended headline — "activation-trajectory matching is not a write path" —
was killed by a cold read before it was written up. Reasons, all confirmed:

- The positive control (`patch_full` at the injection layer) **failed**; by this
  experiment's own decision rule the run is void.
- `L_target` is **anticorrelated** with the outcome: `ft_baseline` installs 92%
  with L_target 1098, *worse* than the untrained 975. A metric on which the
  positive control scores worse than the negative control cannot support a claim
  about what carries the skill.
- `verify_gate` is a **no-op arm**: `rej=80/80`, drift 0.0000, and its L_target
  reprints theta0 to the digit (975.2298/973.8454). It is the untrained model.
  Effective n of matching arms is **3, not 4**.
- **Clip asymmetry destroys matched drift**: matching arms bind the clip on
  50-85% of steps, `ft_baseline` on 9-20%. For four of five arms the drift is
  set by the clipper, which is the exact failure the clip was raised to avoid.
- `random_subspace` (the specificity ablation) **beats** both J-space arms on
  L_target at magnitude 0.2 (0.9434 vs 0.9472 / 1.0311) with indistinguishable
  collateral. By the script's own criterion, J-space specificity is doing no work.
- The final sweep **did not finish** — it stops at magnitude 0.8.
- At magnitude 0.4 `soft_penalty` reaches KL 3.74 nats; the model is destroyed,
  so its 0% is not evidence about anything.

Phase 3 status: **NOT ANSWERED.** The mechanism comparison has never run under
conditions where it could produce an interpretable result.

## 21i. ROOT CAUSE — position alignment was comparing attention sinks to content

`align_tail` aligned by position offset. The with-file prompt is
`[chat_prefix][spec][---][Task: X][FORMAT]`; the no-file prompt is
`[chat_prefix][Task: X][FORMAT]`. Only the trailing task text is shared, so
offset alignment compared the SHORT prompt's leading control tokens — which in
Qwen carry massive-norm attention-sink activations — against the long prompt's
mid-content tokens.

Measured signature at theta0, before the fix: norm ratio **29.9**, `||delta||`
at **99.9%** of `||h||`, `L_target` **975.23**.

After aligning on the longest common **token suffix** (33.4 positions mean):
`L_target` at theta0 is **0.4716**. **Roughly 99.95% of the old objective was
attention-sink mismatch.** Every arm had been optimising to shrink attention
sinks. That is why cosine plateaued at 0.54 and why `random_subspace` matched
the J-space arms — none of them was working on the skill.

## 21j. Phase 3 result, with the objective finally measuring what it claims

7B, `beacon`, band L9-18, full-activation trajectory target, token-aligned,
clip raised to 200 so it binds on 0-4 steps of 80:

| arm | L_target (0.4716 ->) | cosine | norm ratio | install |
|---|---|---|---|---|
| soft_penalty (mag 0.4) | **0.1680** | **+0.928** | 0.968 | **0%** |
| soft_penalty (mag 0.1) | 0.2216 | +0.906 | 0.988 | 0% |
| grad_projection | 0.2489 | +0.893 | 0.974 | 0% |
| verify_gate | 0.2990 | +0.876 | 1.004 | 0% |
| random_subspace | 0.3570 | +0.855 | 1.019 | 0% |
| **ft_baseline** | **1.7803** | **+0.561** | **1.546** | **75%** |

**The matching arms reach cosine 0.93 with the captured state, at correct
magnitude, and install nothing. Fine-tuning moves the state AWAY — cosine down
to 0.561, norm up 55% — and installs 75%.**

Every alternative explanation is now closed:
- **not shrinkage** — norm ratio 0.97-1.02
- **not misalignment** — cosine 0.93
- **not the sink artefact** — fixed, and L_target fell 2000x when it was
- **not memorisation** — train and held-out track to three decimals throughout
- **not budget** — ft_baseline installs at the same drift
- **not an insufficient target in principle** — patching a residual-1.0 state
  installs 54.2% (21g)

### What it means
Reaching the activation state through weights is **not** the same operation as
having that state. The state is sufficient when *supplied* (patching installs);
it is not sufficient when *reached by a weight update that optimises for it*.
The weight configuration that produces a matching activation on these 12 prompts
is not the weight configuration that implements the skill.

That is a real constraint on the write-path programme, and a sharper one than
"the workspace is the wrong target": even with the right target, at cosine 0.93,
activation matching does not transfer the procedure.

### Caveats, held explicitly
- **n=1 seed**, one skill, one model. No error bars.
- Several rows are **drift-UNMATCHED** (0.27-0.32 against a 0.4 target), so the
  magnitude comparison is loose; the install contrast does not depend on it.
- Cosine 0.93 is not 1.0. The apples-to-apples number — what cosine the
  noise-patching at sigma=1.0 (which installs 54.2%) corresponds to — has **not**
  been measured. If sigma=1.0 corresponds to cosine below 0.93, the arms are
  already past the threshold and the negative is strong; if above, it is not.
  **That is the one measurement still missing, and it is cheap.**

## 21k. The negative is NOT supported. The arms never got close enough.

The missing number, measured. Noise-patching the exact captured target, n=24,
reporting the cosine each sigma corresponds to:

| sigma | cosine | install |
|---|---|---|
| 0.00 | 1.0000 | 20.8% |
| 0.10 | 1.0000 | 45.8% |
| 0.25 | 1.0000 | 70.8% |
| 0.50 | 1.0000 | 70.8% |
| 1.00 | 0.9990 | 45.8% |
| **2.00** | **0.9950** | **0.0%** |

**Every patched condition that installs has cosine >= 0.999. Install collapses to
0% by cosine 0.995.**

The trained arms reached **cosine 0.928**. In the units that matter, `1 - cos`:

- arms: **0.072**
- last condition that installs (sigma=1.0): **0.001**

**The arms are ~72x worse than the threshold.** Their 0% install is exactly what
the patching curve predicts for their match quality. It carries **no information**
about whether a good match would install.

**The Phase 3 negative reported in 21j is WITHDRAWN.** "Matching the activation
state does not install the skill" is unsupported: the optimisation never reached
the regime where matching works.

### What Phase 3 actually establishes
1. **A single J-space STATE is insufficient** — patching its component at every layer and
   position installs 0%. That result stands; it is a ceiling, not an optimisation
   outcome.
2. **The skill lives in a position trajectory**, not a state at any depth.
3. **The required match is extreme**: cosine >= 0.999 on the band trajectory.
   80 steps of full-rank SGD on 12 instances reaches 0.928. The binding
   constraint on the write path, as currently formulated, is **optimisation
   precision**, not the idea.
4. Fine-tuning installs at 75% while moving to cosine 0.561 — so the skill is
   reachable in weight space; it just is not reached by *this* objective at
   *this* precision.

### The open question, sharpened
Can gradient descent drive activation-trajectory cosine from 0.93 to 0.999 at a
weight-drift budget comparable to fine-tuning's? If yes, the write path becomes
testable. If no, activation matching is not a practical write path — for a
reason about optimisation, not about representation.

That is a well-posed, cheap next experiment: train one arm hard (no drift cap,
many more steps) and report the cosine it asymptotes to.

### Note on the anomaly
Install remains **non-monotonic** in sigma — 20.8% exact, 70.8% at sigma 0.25-0.5,
0% at 2.0. A slightly corrupted target beats an exact one, still unexplained.
Cosine is 1.000 across that whole rising region, so cosine does not capture what
changes there either.

## 21l. Scope corrections and one open thread

**"J-space cannot be the target" was over-scoped.** What is measured: for THIS
skill, on the 7B, a **single J-space state** patched at every layer and position
installs 0%. The 1.23x enrichment predicts it, so the result is coherent — but
the *mechanism* is unknown, and the distillation formulation (matching across a
spread of probes) specifies a different object than patching one state and is
**not refuted by this**. Recorded as: *a single J-space state is insufficient;
mechanism unknown.*

**"The skill lives in a position trajectory" is an inference, not a measurement.**
It is the natural reading of a per-position ceiling, but no trajectory target has
been tested. Flagged as a HYPOTHESIS in both the log and the report so the next
session does not inherit it as established.

**The non-monotonicity is load-bearing and it undermines the threshold.**
Install runs 20.8% (exact) -> 70.8% (sigma 0.25-0.5) -> 0% (sigma 2.0) while
cosine reads 1.000 across the entire rising region. So **cosine is blind to
whatever drives the rise** — which disqualifies it as the yardstick for the
">= 0.999" criterion that was about to define the next experiment. Building the
asymptote run's success criterion out of a measure shown insensitive in the
relevant regime would repeat the project's characteristic error.

Candidate explanation, to be tested before any further spend: **cosine discards
magnitude.** Isotropic noise *increases* norm (E||t + eps||^2 = ||t||^2 +
sigma^2 d). If the captured with-file state sits at a lower norm than the model's
natural state at those positions, noise pushes it toward the natural norm and
install improves; past the natural norm it overshoots and collapses. That would
make install a **monotone function of norm ratio** that only looks non-monotone
in a direction-only measure.

## 21m. Norm vs direction, disentangled — and the criterion replaced

The same offset-alignment bug had contaminated the norm diagnostic too (written
standalone with `min(len)` instead of importing the fixed alignment). Corrected,
on the 33.4 genuinely corresponding positions:

| | value |
|---|---|
| captured with-file target, mean per-position norm | 62.897 |
| model's natural no-file state, same positions | 66.877 |
| **target / natural** | **0.9405** |

### Norm explains the non-monotonicity

| ‖patched‖/‖natural‖ | 0.941 | 0.945 | 0.950 | **0.967** | **1.042** | 1.298 | 2.022 |
|---|---|---|---|---|---|---|---|
| install | 20.8% | 45.8% | 50.0% | **70.8%** | **70.8%** | 45.8% | 0.0% |

**Install peaks at the natural norm.** The exact patch under-installs because its
norm is 6% low; isotropic noise raises it (E‖t+eps‖² = ‖t‖² + sigma²d) toward
natural and install improves, then overshoots. **Monotone in norm ratio,
non-monotone in sigma** — exactly as predicted. Norm tolerance is tight: ±5%
moves install by ~50 points.

### Direction, measured at FIXED norm

Renormalising the noised target back to the original per-position norm, so only
direction varies:

| cosine | 1.000 | **0.995** | 0.968 | 0.889 | 0.696 | 0.236 |
|---|---|---|---|---|---|---|
| install | 20.8% | **0.0%** | 0.0% | 0.0% | 0.0% | 0.0% |

**Install collapses between cosine 1.000 and 0.995.** A 0.5% angular deviation is
already fatal.

### Consequences

1. **The ">= 0.999 cosine" criterion was derived from the wrong sweep.** The
   unnormalised noise curve varies *norm* (cosine never drops below 0.995 across
   its whole rise), so it calibrated norm tolerance and said nothing about
   direction. Replaced by two criteria, measured separately:
   - **direction: cosine > 0.995**, and plausibly tighter — 0.995 is already 0%
   - **norm: within ~5% of the model's natural norm at those positions**
2. **The trained arms:** norm ratio 0.968-1.019 (**passes**), cosine 0.928
   (**fails by a wide margin**). Their 0% install is fully explained by direction
   error. The Phase 3 negative remains **NOT SUPPORTED**.
3. **The best achievable here is ~70.8%**, not 100% — that is the band-only
   patching ceiling at correct norm and exact direction.

### Recurrence note
This is the **third** appearance of the offset-alignment bug: in `L_target`, then
in the norm diagnostic. It recurred because each diagnostic was written
standalone rather than importing the fixed alignment. Any new script touching
with-file/no-file activation pairs must import the common-token-suffix alignment,
not re-derive one.

# 22. THE ANSWER — and the contamination that hid it

## 22a. Every patching "success" was smuggling the spec's activations

`diag_align_compare_7b.json`. The patching scripts aligned with `min(len)` — the
alignment bug's **fourth** appearance, and the one that mattered most.

| | tokens |
|---|---|
| with-file prompt | 147.5 |
| no-file prompt | 58.5 |
| **genuinely shared token suffix** | **33.5** |
| copied by offset alignment | 58.5 (**25 more than are shared**) |

Those 25 extra positions contain:

```
'<<HARBOR>>`.\n\nAnswer with the token only, on a line beginning `ANSWER:`...'
```

**The spec's own worked example.** Offset-aligned patching was copying
activations that encode the literal answer template. Not an abstract skill state
— the answer, spelled out.

| extent | offset-aligned | **token-aligned** |
|---|---|---|
| band L9-18 | 20.8% | **0.0%** |
| all 28 layers | 54.2% | **0.0%** |

## 22b. The result

**Copying the with-file activation at every layer, at every genuinely shared
position, exactly, installs the skill 0% of the time.** Depth does not help.
Rank truncation does not help (0% at k = 1...128). Norm correction does not help
(0%). Any noise level does not help (0% at every sigma).

**There is no state at the positions that survive removing the file which
carries the skill.** The skill's effect lives at the spec's own token positions,
and those positions do not exist once the file is gone.

That is a structural result about the write path as formulated: **capture-the-
state-and-reach-it-through-weights has no target to aim at.** It is not that the
optimisation is too weak, or the subspace too narrow, or the target too thin —
even a perfect copy of the entire activation tensor transfers nothing.

## 22c. What this voids

- `diag_state_extent_7b.json` — every condition, offset-aligned. **VOID.**
  ("band_all 20.8%", "all_all 58.3%", "position beats layer".)
- `diag_residual_tolerance` — the whole sigma curve, offset-aligned. **VOID.**
  ("noise improves install 3.6x", "ceiling >= 75%".)
- The **norm hypothesis** result: the norm *statistics* stand (target 62.9 vs
  natural 66.9, ratio 0.9405) but the install-vs-norm relation was built on void
  numbers. **The "install peaks at natural norm" conclusion is VOID.**
- The **direction tolerance** "cosine > 0.995". **VOID** — derived from the same
  curve.
- **The asymptote run is cancelled.** Its pre-registration asked whether SGD can
  drive cosine from 0.93 toward a tolerance; there is no tolerance, because a
  cosine of exactly 1.0 installs 0%. `PREREG-asymptote.md` is retained as a
  record and marked cancelled. This saves the spend it was written to justify.

## 22d. What survives

- **J-space enrichment 1.23x** — pure activation statistics, no patching.
- **`diag_target_sufficiency`** — it patched position -1 only, which IS a shared
  position, so its 0% for both J-space and full activation is valid and is now
  the same result as 22b at n=1 position.
- **Delta rank statistics** (correctly aligned): rank 15 for 50% of variance,
  195 for 90%, 942 for 99%. The delta is **not** low-rank. Under the ruling that
  the trajectory hypothesis stays parked until the rank test reports: **the rank
  test reports high rank, and 0% install at every truncation.**
- **ft_baseline installs 75%** at drift 0.083 — the skill is learnable in weight
  space by ordinary means.
- The training arms reach cosine 0.928 at correct norm ratio. **No longer
  informative** about optimisation, since a perfect match also installs 0%.

## 22e. What it means for the programme

The write path is not refuted. **This formulation of it is.** Matching an
activation state cannot transfer a skill whose representation lives in tokens
that are absent at write time.

The distillation framing in THREAD-CONTEXT is *not* touched by this and is now
the live candidate, for a reason this result supplies: it does not require the
skill to be present as a state at surviving positions. Matching output
distributions across a spread of probes asks the student to reproduce a
*function*, which is what a procedure is, rather than a *state*, which this
result shows the procedure is not stored as.

# 23. PHASE 3 (redesigned) — arms 1 and 2

## 23a. Headline as first computed, and why it was withdrawn
"Hard-label FT beats distillation on drift and collateral at every matched
install level; higher T monotonically worse." **Withdrawn on cold read**, on five
counts, all confirmed:

1. **The whole table was one comparison at one rung.** After the divergence gate,
   only 6 of 18 runs had nonzero install. The 25% and 50% cells for both arms
   interpolate the *same* rung pair (1e-5 -> 3e-5) where install jumps 0% -> 75%
   and 0% -> 52.8%. No measurement exists near 25% or 50%.
2. **It was not a drift finding.** At matched lr the arms drift within 6-9%
   (0.0544 vs 0.0594; 0.0955 vs 0.1012). The 1.26-1.70x came entirely from
   install.
3. **`repr_collateral` carries no independent information.** collateral/drift =
   0.287 +- 0.022 for every arm at every level -- it is a first-order linear
   functional of `delta_theta`. "Beats on both" was one finding reported twice.
   And the hypothesis predicts distil sits *below* that line; no arm departs from
   it by more than ~2sd, so this metric is a **null**, not a win for ft_hard.
4. **On `dist_kl`, the one non-collinear collateral measure, T4 is 4x better per
   unit drift.** "Monotonically worse in T" is false where it can be
   independently measured.
5. **The pre-registered abort fired and I promoted the aborting number to the
   headline.** The null check says distil@T1 and ft_hard should be close, and
   that separation means noise exceeds the effect.

Also: the divergence gate (KL <= 1.0) was written *after* seeing the data, and
the in-run script printed the opposite ranking without it. The sign of the result
was set by a post-hoc analysis choice.

## 23b. The pre-registered calibration, run — and it overturns the abort
5 seeds at the crossing rung (lr = 3e-5), the variance component the ladder set
to zero:

| seed | ft install | T1 install | diff | ft drift | T1 drift | ft KL | T1 KL |
|---|---|---|---|---|---|---|---|
| 0 | 75.0% | 52.8% | +22.2 | 0.0544 | 0.0594 | 0.0055 | 0.0101 |
| 1 | 83.3% | 63.9% | +19.4 | 0.0590 | 0.0719 | 0.0030 | 0.0131 |
| 2 | 80.6% | 61.1% | +19.4 | 0.0553 | 0.0589 | 0.0038 | 0.0103 |
| 3 | 63.9% | 44.4% | +19.4 | 0.0820 | 0.0609 | 0.0040 | 0.0099 |
| 4 | 83.3% | 72.2% | +11.1 | 0.0557 | 0.0596 | 0.0050 | 0.0107 |

**install gap +18.3%, sd 4.2, paired t(4) = 9.73, 5/5 seeds positive.**

The separation is **~4.4x the training-run noise floor**.

**On the abort, recorded honestly: the pre-registration was ambiguous.** It
specified both an abort condition ("if they separate, measurement noise exceeds
the effect") and a calibration procedure (replicate to establish the noise
floor). Those point in opposite directions once a separation is observed, and
**I resolved the ambiguity in the permissive direction** -- running the
calibration and continuing rather than stopping. That may be the right call, but
it was a judgement made after seeing the result, and it is recorded as such
rather than as "the check working".

**Drift is not the story**: ft 0.0613 +- 0.0105 vs T1 0.0622 +- 0.0049,
overlapping.

**The independent collateral measure agrees**: general-text KL at equal drift,
ft 0.0042 vs T1 0.0108 -- **2.6x**, paired t(4) = 6.99, 5/5 positive.

## 23c. What arms 1 and 2 actually establish
At the same learning rate and the same weight drift, on three skills:

- distillation from a spec-reading teacher **installs less** (-18.3pp, t=9.73)
- and does **more** general-text damage (2.6x KL, t=6.99)

**Distillation-without-confinement is not a gentler write than hard-label
fine-tuning.** Per protocol-v2, the positive branch ("if distillation alone
beats hard-label FT on collateral, that is positive for the thesis") does not
obtain. The soft-target route is worse on both axes that carry information.

## 23d. The mechanism is unexplained, and one candidate is testable cheaply
The likely cause, from the design cold read: the teacher's distribution is
conditioned on the spec being **in context**, and full-vocab KL spends gradient
on tail structure that encodes "there is a spec-shaped document here" -- topic,
length, position -- rather than the skill. The student burns budget fitting that.

**Top-k truncated KL is the discriminating test.** If the tail is the problem,
truncating it should close the gap toward ft_hard; if the gap survives, the
mechanism is something else. This is pre-registered here before running.

## 23e. Standing caveats
- 3 skills, all from the same corpus; `ledger` teacher accuracy is 92%, so distil
  trains toward a wrong target on ~1/12 ledger items while ft_hard does not.
  This flatters ft_hard and is not corrected.
- T4's curve has **one** informative point; its rising segment is a single rung
  and its matched-install drift is lower-bounded by ladder placement. **No T4
  conclusion is drawn.**
- The three confinement mechanisms are **not run**. Per protocol, arms 1 and 2
  report first.

## §24 — Strict-scoring audit of every earlier headline

Prompted by the Phase 3 finding that lenient install scoring let two arms
optimise toward different strings while both scored correct. That leniency is
not confined to Phase 3, so every scorer in the project was audited.

**Two scorer families, only one carries the same bug.**

*Family A — skill install* (`phase3_writepath.py`, `phase3_distil.py`):
`re.sub(r"\s+"," ",s.strip()).upper()`. Case- and whitespace-lenient. This is
the one that bit. No raw generations were ever stored, so these numbers are not
re-scorable offline; the matched-install rerun now records strict alongside
normalised **and keeps every generation**, so this question is answerable
without a rerun in future.

*Family B — free-text report* (`stage5_report.py`, `stage2_verify.py`):
word-membership against an accept set. Raw per-trial continuations WERE stored
in the `.jsonl.gz` sweep files, so this family was re-scored offline.

**Family B has a worse defect than family A, and it is a false-positive channel.**
`PROBES[0]` is `("How many legs does it have?", {"eight", "8"})`, matched by
`re.findall(r"[a-z0-9]+", text.lower())`. A bare `8` anywhere in the 60-token
continuation scores a report hit. The held-out pool is wikitext, which is full
of scorelines, years and measurements. Actual scored hits include:

    a= 0.0  '8 after a run of only one win in'        <- alpha=0, NO injection
    a= 3.0  '8 for not being able to motivate his team .'
    a= 5.0  'through the 1987 - 8'
    a= 4.0  '8 feet ) to the north of the old one'

On 7B dense, 26 of 27 probe0 hits are this artefact. Probes 1 and 2 (`web`,
`spider`) are alphabetic and largely clean (46/284 and 38/840 lenient-only).

**Strict re-scoring** = alphabetic word-form only (drop the bare numeral),
asserted within the first six words of the answer.

| model / run | headline under lenient | under strict | verdict |
|---|---|---|---|
| 7B dense | floor 1.7%, rises to 100% | floor **0.0%**, rises to 100% | **survives, cleaner** |
| 7B coarse | identical | identical | survives |
| 14B (original, alpha<=24) | 1.2% -> 2.5% | **0.0% at every alpha** | **void — was entirely `8` artefact** |
| 14B v3 (alpha<=50) | 1.2% -> 38.8% | 0.0% -> **27.5%** | real but **not saturated** |

7B: the false-positive floor vanishes and the curve is unchanged in shape. The
50% crossing moves from alpha ~5.65 to ~6.00 — a 6% shift. **The 7B headline
strengthens under strict scoring.**

14B: this is a retraction. The original 14B sweep never ran alpha high enough to
produce a single strictly-scored report hit; its entire apparent signal was the
numeral artefact. Only **v3**, which extended the ladder to alpha=50, shows a
genuine rise — and it reaches only 27.5% at the top rung, i.e. the transition is
observed beginning but **never observed completing**. Any claim that the lens was
"validated on 14B" on the strength of an onset must be downgraded: what 14B
supports is a rising strictly-scored report fraction, not a characterised onset.

The `random_matched` control stored no continuations and cannot be strictly
re-scored. Its flat 1.25% lenient rate is now identifiable as the same numeral
artefact, so strict re-scoring would move it toward 0% — widening the
concept-vs-control contrast rather than narrowing it.

**Rule added:** an accept set for a free-text measure must contain no token that
occurs in the held-out corpus for unrelated reasons. Bare numerals and
single-letter tokens are disqualified. Store raw generations for every scored
trial so scoring choices remain auditable after the fact.

## §24b — What the 14B retraction actually costs

Drawing the consequence, and separating two 14B claims that were being conflated.

**The 14B lens gate stands.** Job 1b (§19f) is the Stage 2 fact-swap
verification: worst cross-check cosine 0.999914 across 17 layers, 4/4 swaps on
capital / language / continent. Its scorer is `stage2_verify.ALIASES` --
`{yuan, renminbi, rmb, cny}`, `{beijing, peking}`, `{chinese, mandarin}`,
`{europe}`, `{asia}` -- **every token alphabetic**. The numeral artefact cannot
touch it. The 14B lens is still a faithful lens and the concept direction is
still causally sufficient there.

**What falls is the 14B onset.** That is a different measure with a different
scorer, and it is the measure ignition analysis is about.

**Consequence for cross-model support.** For the *onset*, the instrument count
is now one:

| model | strict-scored onset | status |
|---|---|---|
| 7B | floor 0.0%, saturates to 100%, 50% crossing at alpha ~6.0 | **clean pass** |
| 14B | 0.0% until alpha 40, reaches 27.5% at alpha 50 (top rung) | rise real, **50% crossing never reached** |
| 1.5B | already recorded WEAK (3/4, threshold moved after the fact) | weak |

So **"two independent models validated" is withdrawn for the onset measure.**
It survives only for the lens gate. Any argument that leaned on 7B/14B onset
agreement has no second leg.

**`ignition-criterion.md` is not invalidated by this.** Its standing gate says
"Model: 7B (lens validation 4/4 on all three conditions)" -- that is the Stage 2
gate, which is unaffected. The 7B was the right choice and remains so. What is
lost is the ability to *replicate* an ignition result on a second model, which
was never in the criterion but was an implicit fallback. Recorded so it is not
silently assumed later.

**The retraction does not rescue the ignition ramp; it makes the 14B worse.**
§21's table recorded the 14B report onset as "~40 (measured)". Strictly scored,
first hits appear at alpha 40 and the curve is at 27.5% by alpha 50 -- the
report side needs *higher* alpha than recorded, so the window between "reports
at all" and "degenerates" is **narrower** than logged, not wider. The
pre-committed failure condition therefore fires on the 14B more clearly, not
less: three of three models, not two of three.

**Open gap, being measured now.** The `random_matched` control stored no
generations, so the published contrast is a strictly-scored concept arm against
a leniently-scored control. The expectation that strict scoring drives the
control toward 0% and widens the contrast is an *argument*, not a measurement,
and this project's record on arguments is poor. `stage5_report.py` now stores
generations, `repeat_frac`, contrast and strict flags for **both** arms; the 7B
sweep is re-running on a second pod. **The contrast is not to be quoted until
that lands.**

**Install-side remains unaudited.** The fine-tuning reference curve, Job 4's
install axis, and the skill teacher gate all used the lenient install scorer and
stored no generations. First strict number from the live run:
`ft_hard lr=2e-5  install 52.1% normalised vs 47.9% strict` -- a 4.2pp gap on
the arm whose training target *is* the gold string. **The 75%->100%
fine-tuning curve is provisional until re-scored strictly.**

## §25 — Do the reportable and coherent regimes overlap? OPEN, and my "three of three" was wrong

The question worth asking: if every model needs more alpha to report the
concept than its coherent window allows, that is a structural fact about this
whole injection paradigm, and it would explain the project's history of
degenerate-window failures as one cause rather than three accidents. It would
also constrain the ignition design, whose discriminator assumes a usable window
exists.

**Evidence assembled. It does not support the strong claim.**

Two different coherence measures disagree, and the standing gate specifies the
one that argues *against* the structural hypothesis.

| model | ceiling per `coherence_*.json` (entropy/KL, separate pool) | in-sweep mean `repeat_frac` at top rung | strict 50% report crossing |
|---|---|---|---|
| 7B | alpha=4 (DEGRADED at 6) | not recorded -- dense sweep predates per-trial gating | alpha ~6.0 |
| 1.5B | alpha=8 | not recorded | above 20, rarely fires |
| 14B | alpha=50 (DEGRADED at 60) | **0.067 at alpha=50** -- far below the 0.5 gate | not reached; 27.5% at alpha=50 |

**The 14B actively contradicts the hypothesis.** Its own sweep shows
`repeat_frac` rising only 0.024 -> 0.067 across alpha 0..50 while strict report
climbs to 27.5% and is still accelerating. There is abundant coherence headroom
at the top rung. **That sweep was truncated, not blocked.** The apparent 14B
ceiling of 50 is an artefact of comparing against a different pool under an
entropy/KL criterion; the per-trial repeat measure the standing gate actually
mandates says the model is fine there.

**The 7B is the case that keeps the hypothesis alive**, and only under the
entropy/KL ceiling: strict 50% crossing at alpha ~6.0 against a ceiling of 4.
But the 7B dense sweep recorded no per-trial `repeat_frac` either, so the
comparison is again across two different measures on two different pools -- the
same defect. **It is not established for any model.**

**Decisive test, cheap:** extend the 14B ladder past alpha=50 and ask whether
strict report reaches 50% before in-sweep `repeat_frac` crosses 0.5. One sweep
answers it. The 7B control re-run now in flight records `repeat_frac` per trial
for both arms, so it answers the same question for the 7B within one dataset and
one measure.

Status: **OPEN hypothesis, not a finding.** Recorded because it constrains the
ignition discriminator if it holds -- and because I stated it as "fires on three
of three models" before checking the in-sweep coherence, which was wrong.

## §25b — Job 4's strict re-score does NOT rehabilitate Job 4

The re-run adds `install_strict` and retains generations. That corrects **one**
axis. Job 4 was never disqualified on scoring; it was disqualified on:

1. asymmetric optimisation effort between mechanisms,
2. collateral compared at matched drift with **unmatched install** (75% vs 100%),
3. n=1 skill (`beacon`), which §24's per-skill spread now shows is the
   *weakest* of the four (+6.7% where others gave +23-25%).

A strict-scored Job 4 is a Job 4 with one fewer defect. **Status remains
NOT TESTED**, per §20a. The strict numbers are for auditing the install scorer
across the project, not for reviving the mechanism comparison.

## §25c — Noted, not acted on (out of scope overnight)

Written down per the standing instruction to record obvious next runs rather
than start them.

1. **Extend the 14B ladder past alpha=50.** The decisive test for §25. Its own
   sweep shows `repeat_frac` 0.067 at the top rung with strict report still
   accelerating through 27.5% -- there is coherence headroom and the sweep
   simply stopped. Until this runs, the 14B onset is *unmeasured*, not *absent*.

2. **Job 4 has a fifth potential defect: it ran in bf16.** `common.load_model`
   uses bf16 on CUDA. Phase 3 had to move to fp32 because bf16's 8-bit mantissa
   cannot express updates at ~1e-4 relative to ||theta||~1e2. Job 4's drift
   magnitudes are 0.05-0.4 absolute, i.e. ~1e-3 relative -- larger, so less
   likely to be fatal, but still the marginal region. Not investigated.

3. **The strict-vs-normalised install gap is itself install-dependent.** On the
   ft_hard ladder it falls 4.2 -> 4.2 -> 2.1 -> 2.1 -> 0.0pp as install rises
   52% -> 92%: more training drives exact-string reproduction, not merely
   case-equivalent. Consequence: **comparing strict gaps between arms is only
   meaningful at matched install.** A raw gap comparison at fixed lr would be
   confounded in exactly the way the original headline was.

## §26 — Control re-run (7B, generations stored, both arms strict-scored). Three results.

`stage5_report_7b_strictctl.jsonl.gz`, 2520 rows, 60 held-out prompts,
alpha 0..20. Both arms now store generations, `repeat_frac`, contrast and strict
flags. 75.7 min on a 4090, $0.46, pod destroyed.

**1. The contrast is real, and now measured rather than argued.**
The `random_matched` control is **0.0% at every alpha, lenient AND strict**.
The concept arm goes 0% -> 100%. The earlier control's flat 1.25-1.7% was
entirely the numeral artefact, as predicted -- but it was predicted from an
argument, and this project's record on arguments is poor. It is now a
measurement. The contrast may be quoted.

The `piano` contrast probe is **0% at every alpha in both arms**: the injection
is specific, not a general excitation.

**2. The strict/lenient gap on the concept arm is real but does not change the
verdict.** Peak divergence at alpha=8 (81.7% lenient vs 68.3% strict, 13.4pp).
Both saturate to 100% by alpha 12. Lenient scoring inflates the middle of the
curve and moves the onset earlier; it does not manufacture the transition.

**3. The overlap hypothesis (§25) is FALSE for the 7B. Decisively.**
Within one dataset, one measure, the standing gate's own `repeat_frac`:

```
strict report 50% crossing : alpha 5.31
strict report reaches 100% : alpha 12.0    (repeat_frac 0.373 -- coherent)
repeat_frac crosses 0.5    : alpha 13.08
USABLE WINDOW              : alpha [5.31, 13.08], width 7.77
```

The concept is **fully reportable at 100% while the model is still coherent**,
with room to spare. There is no squeeze. Combined with the 14B (repeat 0.067 at
alpha=50, report still climbing), **both models with in-sweep coherence data
show ample overlap.** The hypothesis is dead, not open. §25 is superseded.

**4. The consequence is larger than the hypothesis. The coherence ceiling used
throughout this project was wrong.**

`coherence_7b.json` put the 7B ceiling at **alpha=4** (DEGRADED at 6) using
`entropy > baseline+1.5 or kl > 3.0` on a *separate* prompt pool. The standing
gate specifies "repeat fraction from this sweep's own prompt pool". By that
measure the ceiling is **alpha ~13** -- roughly **3x higher**. At alpha=6, where
`coherence_7b.json` says DEGRADED, the in-sweep repeat fraction is 0.056, i.e.
*lower* than the alpha=0 baseline of the old dense run.

The entropy/KL criterion was flagging distribution *shift* -- which is what a
successful concept injection is supposed to cause -- and calling it degeneracy.
It penalised the intervention working.

**"Three findings died to this" (per `ignition-criterion.md`'s standing gates).
Those deaths were adjudicated by the wrong instrument and need revisiting.**
NOT DONE, NOT STARTED -- out of scope overnight, recorded for the morning.
This does not resurrect those findings; it means their cause of death is
unreliable.

## §27 — Phase 3 matched-install sweep. The headline does not survive, and neither does its replacement.

4 skills (beacon, crate, hollow, stamp; ledger dropped, 92% teacher), 2 seeds,
80 steps, 6-point lr ladder, 5 arms. 276 min on an A100. Analysis script
committed BEFORE the data landed (`analyse_phase3_matched.py`).

Teacher accuracy 100% on all four. Sequence mismatch (teacher greedy vs
ft_hard's target) **26/48 training items** -- the confound reproduces across
skills and was not a ledger artefact.

### The original 18.3pp result is withdrawn

At matched install, clustered on skill:

| comparison | normalised | strict |
|---|---|---|
| ft_hard - distil@T1/forced | **-4.2% +- 17.0** t(3)=-0.49 p=0.658 | +1.0% +- 29.9 p=0.949 |
| ft_hard - distil@T1/free | **+0.0% +- 21.2** p=1.000 | -0.0% +- 23.8 p=1.000 |

**But this does NOT establish equivalence, and must not be reported as such.**
95% CIs (t_crit=3.182, n=4): T1/forced normalised **[-31.3, +22.9]**, strict
**[-46.6, +48.6]**, T1/free **[-33.7, +33.7]**. The withdrawn 18.3pp deficit
lies inside every one of them. The correct statement is that **this design
cannot resolve a 30pp effect**, not that the effect is absent.

Three things changed at once between the two experiments -- seed count (5->2),
clustering unit (seed->skill), and the addition of the matching rule. No part of
the difference is attributable to any one of them. Note also that the /free arm,
which reproduces the ORIGINAL implementation, also shows p=1.000 at matched
install: **the matching defect alone was sufficient to produce the original
headline.** The sequence confound is real but was not required.

### The collateral ratio is not estimable

`dist_kl` ratio, distil@T1/forced vs ft_hard, under defensible matching rules:

```
closest install to 75%  (pre-registered)   19.73x   install 80.2%  drift 0.2109
lowest drift, install >= 70%                2.46x   install 82.3%  drift 0.1165
closest drift to ft_hard mean               1.01x   install 63.5%  drift 0.0798
```

**1.01x to 19.7x.** The pre-registered rule returns the most extreme value, and
it does so for a specific reason: for seed 1 it selects lr=2e-4 (install 77.1%,
|dist|=0.021) over lr=5e-5 (install 81.2%, |dist|=0.063) -- **a margin of two
held-out items out of 48, against a binomial SE of ~5.8pp**. That rung has
`dist_kl` 0.28112 against 0.01212 at 5e-5, a factor of 23, and it is the rung
where the sibling seed diverged (n=1). Seed 1 contributes **91% of the arm
mean**. Pre-registration does not rescue a rule whose failure mode is
data-triggered and whose output turns on two eval items.

The rule also failed to match: T1/forced sits at **80.2% install vs ft_hard's
76.0%** -- above, on the axis held constant -- with **2.5x the drift**
(0.2109 vs 0.0844). And ft_hard seed 1 hit an **exact tie** (|75-66.67| =
|83.33-75| = 8.333); `min()` broke it silently toward 3e-5. Taking 5e-5 puts
ft_hard at 84.4% install, the highest of any arm, and the matched framing
collapses outright.

### T4 is untestable on this grid, again

Best achievable install: T4/forced **31.2%**, T4/free 42.7%, against ft_hard's
76.0%. There is no matched comparison point. The one p<0.05 in the whole
analysis (T4/forced strict, t(3)=3.42, p=0.042) is computed against a
**max-selected** per-seed rung at a third of ft_hard's install, and drops to
p~=0.12 on removing crate. **It is not a result.** Objection 2 (T4 undertested)
is NOT resolved; it is now documented as unresolvable within this ladder.

T4 also fails completely and specifically: **crate and hollow are at 0.0%
install at every single rung in both T4 arms**, while beacon and stamp install.
(Note: the T^2 gradient rescaling IS applied -- `* (T*T)` in `l_distil` -- so
this is not the missing-Hinton-correction artefact it resembles.)

### What the strict-vs-normalised gap actually is

Not an arm property. Per-skill mean gap over all non-diverged rungs:

```
ft_hard          beacon  9.0pp  crate 0.7  hollow 0.0  stamp  0.7
distil@T1/forced beacon  6.1pp  crate 0.0  hollow 0.0  stamp 20.5
```

**One skill carries it, and a different skill for each arm.** The arm-level
"9.4pp vs 4.2pp" is stamp/4 and beacon/4. The finding is: on `stamp`, distilled
models emit a differently-cased or -spaced version of the correct answer. That
is a real and specific observation and it is not "distillation degrades exact
reproduction" in general.

### The one clean directional difference is the one I excluded

Divergences per 12 seed-runs: **ft_hard 0, T1/forced 1, T1/free 3, T4/free 2,
T4/forced 4.** No test, small counts -- but it is the only arm-level difference
visible before any selection rule is applied, and the analysis defines it away
by excluding diverged runs. Excluding them also flatters distillation's install
numbers (the worst runs are deleted) while leaving one pathological survivor to
dominate the collateral number. **Recorded as the most promising signal in the
dataset. NOT tested, NOT run.**

### Cold read

Run on the numbers alone in a fresh context, no framing, instructed to find why
they are wrong. It independently identified: the 19.7x being 91% one seed at a
half-diverged rung selected by a two-item margin; the 4.2pp install gap in the
wrong direction; the silent tie-break; the max-selection behind p=0.042; the
one-skill origin of the strict gap; the n=1 rows; and that the nulls have no
power against the withdrawn 18.3pp. All verified against the raw data and all
correct.

Two of its charges were wrong and are recorded as such: (a) it flagged
"identical drift 0.0654" for T1/forced and T1/free at lr=2e-5 as a logging bug
-- the underlying per-seed values are 0.0662/0.0645 and 0.0650/0.0658, distinct;
the collision was a coincidence of two-seed means at 4dp in my summary table.
(b) It inferred a missing T^2 gradient correction from T4's behaviour; the
correction is applied.

## §28 — Job 4 strict re-score. Status unchanged: NOT TESTED.

Re-run with strict scoring and generations retained. 41.6 min. Two results.

**1. The fine-tuning reference curve under strict scoring:**

| drift target | drift | install (norm) | install (strict) | gap |
|---|---|---|---|---|
| 0.05 | 0.0518 | 33.3% | **25.0%** | 8.3pp |
| 0.10 | 0.0981 | 91.7% | **83.3%** | 8.3pp |
| 0.20 | 0.1826 | 91.7% | **91.7%** | 0.0pp |
| 0.40 | 0.2793 | 100.0% | **100.0%** | 0.0pp |

The provisional status is lifted. The curve holds up: strict scoring costs
8.3pp at the low-drift end and nothing at the high end. Same install-dependence
as §25c -- more training drives exact-string reproduction. The "75% -> 100%"
shape is real.

**2. All four write-path mechanisms install 0.0% at every magnitude, strict and
normalised.** `soft_penalty`, `grad_projection`, `verify_gate`,
`random_subspace`: 0.0% at drift targets 0.05 / 0.1 / 0.2 / 0.4. Only
`ft_baseline` installs. The matched-install gate refused to make any comparison
at any level ("only 1 arm cleared the gate -- no comparison"), which is the gate
working as specified.

**This differs from the original Job 4, which recorded `targeted_penalty_on` at
75%.** The likeliest explanation is that the original 75% was the token-suffix
alignment artefact -- the same bug that produced "phantom installs of 4.2-100%"
elsewhere -- and that with the alignment fixed the mechanisms install nothing.
Not investigated; recorded.

**Status remains NOT TESTED.** Scoring was one of four defects. Asymmetric
optimisation effort, unmatched install, and n=1 on beacon are untouched, and
beacon is now known to be the weakest of the four skills. A strict-scored Job 4
is a Job 4 with one fewer defect, not a rehabilitated Job 4.

## §29 — Overnight run closed

Three jobs, all completed. Both pods destroyed and verified via the API; no
local processes running.

| job | outcome | cost |
|---|---|---|
| Matched-install distillation sweep | complete, 60 runs, 276 min | |
| Control re-run with generations | complete, 2520 rows, 76 min | $0.46 |
| Job 4 strict re-score | complete, 41.6 min | |
| A100 total (sweep + Job 4 + idle) | 7.119 h | $8.47 |
| **total** | | **$8.93** of a $25 cap |

**Operational failure to record: ~1 hour of idle A100 (~$1.19) was billed for
nothing.** A watchdog script used `pgrep -f phase3_matched_install` inside a
shell whose own command line contained that string, so the check always matched
itself and reported RUNNING forever. The sweep had finished at 04:18; it was not
noticed until 05:25. Fix is `pgrep -f '[p]hase3_matched_install'`. Two further
self-inflicted delays: Job 4 was launched twice against a stale copy of the
script (once missing the lens file, once missing the strict-scoring patch), and
a docstring edit displaced `from __future__` past line 1 and crashed it a third
time. All three were tooling errors, not experimental ones, and none changed a
result.

## §30 — Instrument audit. My "three findings died to the wrong adjudicator" claim is WITHDRAWN.

Went through the retractions one at a time and recorded which instrument killed
each. The premise of §26.4 was wrong.

**`coherence_*.json` is consumed by nothing.** `grep -rn "coherence_" scripts/`
returns no reader outside `coherence_check.py` itself. The entropy/KL number was
a diagnostic that was never wired to a gate. **The alpha=4 figure never killed
anything.**

Every operative ceiling in this project came from **repeat fraction**, recorded
at §20c: "Coherent window is alpha <= 8 (repeat-fraction is the better
degeneracy marker than entropy; entropy necessarily rises under injection)."
That judgement was correct when it was made and it is the one that propagated.

| finding | killed by | stands? |
|---|---|---|
| 19h "100% report on 60/60 prompts" | repeat fraction 84-97% at alpha>=13 | **stands** |
| 20i / 20k ramp failure, all three models | report onset vs repeat-fraction ceiling (alpha~8) | **stands** |
| 21h "trajectory matching is not a write path" | five unrelated reasons: positive control failed, L_target anticorrelated with outcome, verify_gate a no-op (rej 80/80, drift 0.0000), clip asymmetry destroying matched drift, random_subspace beating J-space | **stands** |
| 17d pooled dip | unvalidated instrument | stands |
| Stage 4 null | pending 1.5B ceiling measurement | still pending |

**Nothing needs re-opening.** I inferred a propagating error from the existence
of the alpha=4 number without checking whether anything consumed it. It did not.

### The correction that DOES survive is to my own §26

§26 claimed a usable window of alpha [5.31, 13.08] and "the concept is fully
reportable at 100% while the model is still coherent". **Both wrong.** Reading
the generations at the boundary:

```
alpha= 7  strict 70.0%  'other arachnid creatures, the spiders of the'
alpha= 8  strict 68.3%  'and a spider web of yarns. the whole'
alpha= 9  strict 83.3%  'the spider webs of the city, and spent hours'
alpha=10  strict 91.7%  'and a spider spider spider spider spider spider'
alpha=11  strict 98.3%  'spiders, the spider spider spider spider spider'
alpha=12  strict 100%   'spider spiders, spider spiders, and a spider spider'
```

Degeneration begins at **alpha 10**. The 100% report figure sits at alpha 12,
inside wreckage -- the same error as §19h, made again, by me, one section after
quoting §19h as a cautionary tale.

**Corrected 7B window: alpha [5.31, ~9], width ~3.7 (not 7.77). Strict report at
the ceiling is 83.3%, not 100%.** This is very close to §20c's original alpha<=8
and does not disturb any prior retraction.

Why the shipped measure missed it: `degeneracy()` counts unique
whitespace-delimited strings, so `spider` / `spiders` / `spiders,` are three
distinct tokens, and the probe answers average 8.5 words. Stemming and
punctuation-stripping move the 0.5 crossing only from 13.1 to 12.0, so **the 0.5
threshold itself is the problem on short samples**, not just the tokenisation.
Rule added to DECISIONS: every coherence ceiling must be supported by inspected
text at the boundary.

**The overlap question, settled properly:** a window does exist, so the strong
"regimes do not overlap" hypothesis stays dead. But it is narrow (~3.7 alpha
units) and report never saturates inside it. That is materially closer to the
original §20i/20k reading -- too narrow to fit a symmetric ramp with both halves
populated -- than to my §26 framing.

## §31 — Divergence as an outcome variable

Reported with the runs kept in, from the existing sweep. This is a rate, not a
comparison at matched anything, so no matching defect applies.

| arm | diverged | rate | at lr |
|---|---|---|---|
| ft_hard | 0/12 | **0.0%** | -- |
| distil@T1/forced | 1/12 | 8.3% | 2e-4 |
| distil@T4/free | 2/12 | 16.7% | 8e-5, 2e-4 |
| distil@T1/free | 3/12 | 25.0% | 3e-5, 1.3e-4, 2e-4 |
| distil@T4/forced | 4/12 | 33.3% | 8e-5, 1.3e-4, 2e-4 |

Hard-label CE never diverged in 12 seed-runs; every distillation arm did.
Restricted to the four lower rungs where ft_hard is stable, the counts thin to
0/8, 0/8, 1/8, 1/8, 1/8 -- so the effect is concentrated at high lr, which is
where a matched-install search is forced to go for the arms that install slowly.

**This is the one arm-level difference in the sweep that is monotone and needs
no selection rule.** It also contaminates both headline axes when diverged runs
are excluded: exclusion deletes distillation's worst runs (flattering install)
while leaving one pathological survivor to dominate collateral. Mechanism is
plausible -- a soft, temperature-scaled target is less stable than a one-hot at
matched lr -- but n is small and there is no test here.

Full measurement (install and collateral for diverged runs, which the early
abort never evaluated) requires a rerun without the abort. NOT RUN.

## §32 — The falsification test failed to falsify. J-space state matching is retired.

Pre-stated prediction (before the run): `soft_penalty` with `lambda_orth=0` --
pure `L_target`, no confinement -- installs 0% while still minimising
`L_target`. If it installed, confinement was the problem and the whole reading
was wrong.

`phase3_writepath_7b_lorth0.json`, 40.4 min, A100, $1.06. theta0 L_target 0.4721.

| mag | CONFINED drift / L_t / cos / inst | UNCONFINED drift / L_t / cos / inst | ft_baseline inst |
|---|---|---|---|
| 0.05 | 0.0564 / 0.3720 / +0.850 / **0.0%** | 0.0442 / 0.3981 / +0.841 / **0.0%** | 33.3% |
| 0.10 | 0.1060 / 0.3249 / +0.866 / **0.0%** | 0.0957 / 0.2729 / +0.884 / **0.0%** | 91.7% |
| 0.20 | 0.1797 / 0.2827 / +0.881 / **0.0%** | 0.1484 / 0.2721 / +0.884 / **0.0%** | 91.7% |
| 0.40 | 0.4629 / 0.2209 / +0.904 / **0.0%** | 0.4258 / 0.2144 / +0.907 / **0.0%** | 100.0% |

**Removing confinement entirely changes nothing.** Unconfined optimisation
achieves the *best* state match in the experiment -- L_target 0.2144 against
theta0's 0.4721, cosine 0.907 to the captured J-space state -- and installs
**0.0% at every magnitude**, at drifts where ordinary fine-tuning installs
33-100%. Confinement was never the obstacle. It is not a harness fault either:
the arms train, drift hits target, and the objective falls monotonically.

**Stated precisely, avoiding a stronger claim than the data carries.**
`ft_baseline`'s own state match is erratic -- L_t 0.503 / 9.828 / 0.551 / 1.485
and cosine +0.808 / +0.273 / +0.795 / +0.593 across the four magnitudes -- while
its install stays 33-100%. So this is not a clean monotone anticorrelation. The
two defensible statements are:

1. **Minimising `L_target` is sufficient for 0% install.** Every arm that
   optimises it -- five configurations now, confined four ways and unconfined --
   reaches L_t 0.21-0.40, cosine 0.84-0.91, and installs nothing.
2. **Matching the captured state is not necessary for install.** `ft_baseline`
   installs 91.7% at cosine +0.273, i.e. while barely matching it at all.

Together: **optimising toward the captured J-space state actively selects
against having the skill.** That has a direction, which is why it is stronger
than a null.

### This retires the formulation, not an arm

Three independent lines now converge:

- **1.23x enrichment.** The skill-in-context state differs from the no-skill
  state barely more in J-space than chance -- the signal was never concentrated
  there.
- **Prefix geometry.** The positive control fails identically for a known-good
  fact edit once the readout moves past the patch boundary, so patching the
  state cannot carry a procedure across positions.
- **This run.** Reaching the state through weights, by five configurations,
  yields zero behaviour -- and does so *better* the harder the state is matched.

**J-space state is not the object that carries a procedure.** Phase 3 as
formulated -- capture the workspace state, reach it through weights, expect the
behaviour to follow -- is retired. Not "not yet tested": tested, and the
objective points the wrong way.

What survives untouched is the *question*: whether there is a write path from
activation to weight. What is retired is the answer that the thing to write is
the corpus-averaged J-space state. The Jacobian lens remains valid
instrumentation -- the 7B onset (log 26, 30) is real and strictly scored.

## §33 — Divergence rerun: DIED at 32/60. Partial result, and it weakens what I reported.

Launched with `--diverge-drift 1e9` so every run trains to completion and is
evaluated. **The process exited silently after 3.7h at 32/60 runs** -- no
traceback, no OOM string, no CUDA error, host memory fine (33GB/1007), GPU idle,
no JSON written. Cause not determinable from the host. Not rescued, not
relaunched, per the standing rule. Pod destroyed; log salvaged to
`out/div_noabort.log`. Billed 3.715h, $4.42.

**Seed 0 is complete (30/30 runs, every one evaluated). Seed 1 got 2/30.**

### The abort flag was correct, but it conflates two different things

All five of seed 0's previously-aborted runs were still above drift 0.4 at the
end of training, so the abort was not catching transient excursions. But what
those runs actually *are* splits sharply:

| arm | lr | final drift | install | strict | dist_kl | verdict |
|---|---|---|---|---|---|---|
| distil@T1/forced | 2e-4 | 1.3256 | 0.0% | 0.0% | 7.7473 | destroyed |
| distil@T4/forced | 1.3e-4 | 1.3486 | 0.0% | 0.0% | 10.5870 | destroyed |
| distil@T4/forced | 2e-4 | 1.6134 | 0.0% | 0.0% | 23.7821 | destroyed |
| distil@T1/free | 1.3e-4 | 0.5559 | 43.8% | 43.8% | 3.1730 | damaged but installing |
| distil@T1/free | 3e-5 | 0.4967 | **64.6%** | **60.4%** | **0.0145** | **healthy** |

**`distil@T1/free` at lr=3e-5 installs 64.6% with `dist_kl` 0.0145 -- collateral
comparable to the healthiest runs in the sweep -- and was thrown out as
"diverged" for having drift 0.497.** The criterion `drift > 0.4` conflates *a
large weight change* with *a broken model*, which is the same category error as
treating drift as damage rather than dose.

### Correction: the exclusion did NOT contaminate seed 0's headline

I claimed the exclusion "contaminates both headline axes". On seed 0 it does
not. Matched-install selection with all runs kept picks **exactly the same rung
for every arm** as the selection with diverged runs excluded -- identical
install, strict, drift and dist_kl throughout. The claim may still hold for
seed 1, where the pathological 2e-4 selection occurred, but **seed 1 did not
complete, so it is untested.**

### The divergence finding is weaker than I reported

§31 gave rates out of 12 seed-runs per arm. With only seed 0 complete and
"divergence" decomposed into *high drift* vs *actually destroyed*:

| arm | high drift (>0.4) | destroyed (0% install, KL>1) |
|---|---|---|
| ft_hard | 0/6 | 0/6 |
| distil@T1/forced | 1/6 | 1/6 |
| distil@T1/free | 2/6 | 0/6 |
| distil@T4/forced | 2/6 | 2/6 |
| distil@T4/free | 0/6 | 0/6 |

Hard-label CE still never leaves the stable region, and destruction occurs only
in the `/forced` arms -- but this is **one seed**, the monotone ordering of §31
does not survive the decomposition, and `distil@T1/free` has the second-highest
high-drift count with **zero** destroyed models. The directional claim from §31
is downgraded to: *hard-label CE did not leave the stable region in 6 runs; some
distillation configurations did.* No ordering among the distillation arms is
supported.

### Free harness verification

25/25 non-aborted seed-0 runs reproduce **exactly** across two independent
launches -- install, drift and dist_kl identical to the log's full precision.
The training path is deterministic given the seed. (An initial check reported
"18 of 25 differ"; that was my own artefact from comparing log values rounded to
0.1% against full-precision JSON floats. Corrected.)
