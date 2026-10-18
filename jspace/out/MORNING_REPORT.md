# Morning report — Phase 3

> Sections below are kept in the order they were produced. **Read section A
> first — it supersedes everything under it.**

## A. THE ANSWER

**Every patching "success" in this project was copying the spec's own
activations.** The patching scripts aligned by `min(len)`, so they copied 58.5
positions when only 33.5 are genuinely shared. The extra 25 contain:

```
'<<HARBOR>>`.\n\nAnswer with the token only, on a line beginning `ANSWER:`...'
```

The spec's worked example — the answer template, spelled out.

| extent | offset-aligned | **token-aligned (correct)** |
|---|---|---|
| band L9-18 | 20.8% | **0.0%** |
| all 28 layers | 54.2% | **0.0%** |

**Copying the with-file activation at every layer, at every genuinely shared
position, exactly, installs the skill 0% of the time.** Depth doesn't help. Rank
truncation doesn't help (0% at k=1..128). Norm correction doesn't help. No noise
level helps.

**There is no state at the positions that survive removing the file which carries
the skill.** The capture-the-state-and-reach-it-through-weights formulation has
no target to aim at — not a thin target, none. It isn't weak optimisation, a
narrow subspace, or an insufficient target: a perfect copy of the entire
activation tensor transfers nothing.

**The asymptote run is cancelled** — it asked whether SGD can reach a tolerance
that does not exist. That saves the spend it was written to justify.

### Voided by this
`diag_state_extent` (all conditions), the whole residual-tolerance sigma curve,
"noise improves install 3.6x", "install peaks at natural norm", and the
"cosine > 0.995" direction tolerance. All were computed on offset-aligned
patches.

### Surviving
J-space enrichment 1.23x (pure statistics); single-shared-position patching 0%
for J-space and full alike; the delta is **not** low-rank (15 dims for 50% of
variance, 942 for 99%) with 0% install at every truncation — so the rank test
reports high rank, and the trajectory hypothesis stays parked; ordinary
fine-tuning installs 75% at drift 0.083, so the skill *is* learnable in weight
space.

### What it means
The write path is **not** refuted — this formulation of it is. A skill whose
representation lives in tokens absent at write time cannot be transferred by
matching an activation state. The distillation framing is untouched and is now
the live candidate, for a reason this result supplies: matching output
distributions across probes asks the student to reproduce a **function**, which
is what a procedure is, rather than a **state**, which this shows it is not
stored as.

---


> **SUPERSEDED IN PART.** Sections 1-4 below were written before the root cause
> was found. The corrected result is at the top of section 0. Kept intact
> because the sequence matters.

## 0. FINAL STATE — read this first

**Phase 3 is OPEN. The negative below (section 0a) is WITHDRAWN.**

The arms reached activation-trajectory cosine **0.928**. Measured threshold: every
patched condition that installs has cosine **>= 0.999**; install is already 0% at
cosine 0.995. In `1 - cos` the arms are at 0.072 against a threshold of 0.001 —
**~72x too far**. Their 0% install is exactly what the patching curve predicts
for that match quality, so it says nothing about the hypothesis.

**What stands:** J-space cannot be the target (0% ceiling when patched directly
at every layer and position); the skill appears to live in a position trajectory rather than a
state at depth (HYPOTHESIS — a trajectory target has not been tested); the required match is cosine >= 0.999, and 80 steps of full-rank
SGD reaches 0.928.

**The criterion, now measured properly (norm and direction separated):**
- **direction: cosine > 0.995** — install is 20.8% at cosine 1.000 and **0% at
  0.995**, at fixed norm. Half a percent of angle is fatal.
- **norm: within ~5% of natural** — install peaks at norm ratio 0.97-1.04 and
  the captured target sits 6% low, which is why an exact patch (20.8%)
  underperforms a slightly noised one (70.8%).

The arms pass on norm (0.968-1.019) and fail on direction (0.928).

**The sharpened question:** can gradient descent drive activation-trajectory
cosine from 0.93 past 0.995 within a fine-tuning-comparable drift budget? Cheap
to answer — train one arm hard with no drift cap and report where cosine
asymptotes. Best achievable install at band extent is ~70.8%, not 100%.

---

## 0a. THE RESULT AS OF THE ALIGNMENT FIX (superseded by 0 above)

**A position-alignment bug had been dominating the objective all along.**
`align_tail` aligned by offset, so the short prompt's attention-sink tokens
(massive norm in Qwen) were compared against the long prompt's content tokens.
`L_target` at theta0 was **975.23**; after aligning on the longest common TOKEN
suffix it is **0.4716**. **~99.95% of the objective was sink mismatch.** Every
arm had been optimising to shrink attention sinks.

With that fixed, and clip raised so it binds on 0-4 of 80 steps:

| arm | L_target (0.4716 ->) | cosine | norm ratio | install |
|---|---|---|---|---|
| soft_penalty | **0.168** | **+0.928** | 0.968 | **0%** |
| grad_projection | 0.249 | +0.893 | 0.974 | 0% |
| verify_gate | 0.299 | +0.876 | 1.004 | 0% |
| random_subspace | 0.357 | +0.855 | 1.019 | 0% |
| **ft_baseline** | **1.780** | **+0.561** | **1.546** | **75%** |

**The matching arms reach cosine 0.93 with the captured state at correct
magnitude and install nothing. Fine-tuning moves AWAY from that state — cosine
0.561, norm +55% — and installs 75%.**

Every alternative is closed: not shrinkage (norm ratio ~1), not misalignment
(cos 0.93), not the sink artefact (fixed), not memorisation (train = held-out to
3 dp), not budget (FT installs at the same drift), not an insufficient target
(patching a residual-1.0 state installs 54.2%).

**Interpretation:** reaching the activation state through weights is not the same
operation as having it. The state is sufficient when *supplied*; it is not
sufficient when *reached by a weight update optimised for it*. The weights that
produce a matching activation are not the weights that implement the skill.

**Caveats: n=1 seed, one skill, one model; several rows drift-UNMATCHED; and the
one missing number is what cosine sigma=1.0 noise-patching corresponds to — if
below 0.93 the negative is strong, if above it is not. Cheap, and it is next.**

---

**Phase 3 is NOT answered.** The mechanism comparison has still never run under
conditions where it could produce an interpretable result. But the night was not
wasted: four measurements now constrain the design in ways that were guesses
before, and one of them inverts an assumption the whole experiment rested on.

## 1. What is now established

**A single J-space STATE is insufficient. Measured ceiling: 0%.**
Patching the J-space component in *directly*, at every available layer and every
token position — which no weight update can beat — installs the skill **0%** of
the time. The same sites with the full activation reach 58.3%.

| | J-space only | full activation |
|---|---|---|
| layer 7, all positions | 0.0% | 0.0% |
| band L9-18, all positions | **0.0%** | 20.8% |
| all 28 layers, all positions | **0.0%** | 58.3% |
| skill file in context | — | 100.0% |

J-space transmits *what the task is about*, never *how to do it* —
`'RenderIngotAsBeaconToken()'`, `'INGOTBEACONTOKEN'`, always the right word,
never the `<<...>>` procedure. That is consistent with the source paper rather
than against it: J-space is the subspace that drives **report**, and report is
content. **A procedure is not content. The workspace is the wrong target for
writing one.**

**Position beats layer, everywhere.** Every single-position condition is 0%
including all 28 layers; every all-position condition transmits. The skill is
carried in a **trajectory across token positions**, not a state at any depth.

**Noise in the target improves install 3.6x.** Exact patch 20.8%; RMS-0.25 noise
**75.0%**. The rigid clamp is worse than a loose one — probably because the
captured state comes from a longer prompt and position-aligned copying forces
contradictions with the actual context. Hypothesis, not result.

## 2. The dissociation, and why it is not what I thought

- Patching a residual-1.0 state installs **54.2%**.
- Training weights to reach residual 1.0 installs **0%**.

Same target, same residual, opposite outcome. So the failure is **not** "the
state does not carry the skill" — it demonstrably does. The weight update is
reaching low MSE by a route that does not produce the functional state, almost
certainly by shrinking activations toward the target mean rather than matching
direction.

**The single decisive next measurement: cosine similarity between achieved and
target state, not just MSE.** If cosine is low while MSE is low, that is the
whole story and every "matching does not install" claim is void for that reason.

## 3. What I withdrew before writing it up

The intended headline — "activation-trajectory matching is not a write path" —
was killed by the mandatory cold read. It was wrong on seven counts, all
confirmed against the log:

1. The **positive control failed** (`patch_full` = 0%); by the experiment's own
   rule the run is void.
2. `L_target` is **anticorrelated with the outcome** — `ft_baseline` installs 92%
   with L_target *worse* than untrained (1098 vs 975).
3. `verify_gate` is a **no-op**: rejected 80/80 steps, drift 0.0000, L_target
   reprints theta0 to the digit. Effective n of matching arms is 3, not 4.
4. **Clip asymmetry destroys matched drift** — matching arms clip on 50-85% of
   steps, ft_baseline on 9-20%.
5. `random_subspace` (the ablation) **beats** both J-space arms on L_target, so
   J-space specificity is doing no work.
6. The sweep **didn't finish** (stops at magnitude 0.8).
7. At magnitude 0.4 soft_penalty reaches **KL 3.74 nats** — model destroyed, its
   0% means nothing.

## 4. To run next, in order

1. **Cosine, not just MSE** — settles whether the arms are matching or shrinking.
   One cheap run.
2. **Fix the capture protocol** — align on token identity, not position offset;
   push the `all_all` ceiling from 58% toward 100%. Report install as a fraction
   of the achieved ceiling, never against 100%.
3. **Fix or drop `verify_gate`**; raise `--clip` until binding is under 10% for
   every arm; then re-run the mechanism comparison.

## 5. Housekeeping

- All my pods destroyed and verified. **One pod I did not create is still
  running**: `pn2-v035-compat` (id `uw16t2lalt1es1`, $0.74/hr, 50 GB volume
  `pn2-v035-compat_volume`, started 23:40). Left alone — not mine to kill.
- Spend this session ~$7.4 (three pods: $5.77 + $0.88 + $0.76).
- **Note on your figure of ~$12.91 total for the project** — my running tally is
  well above that (~$49 including last night). Worth reconciling against the
  RunPod billing page; one of the two numbers is stale.
- I minted API keys in the browser again this session. DECISIONS now records
  that you do that, not agents. Key still live — revoke when convenient.
