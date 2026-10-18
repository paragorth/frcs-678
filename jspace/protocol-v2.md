# Workspace-Level Learning: Phased Protocol — v2, 21 August 2026

Supersedes `protocol.md` (stale: Phase 0 gone, lens from Anthropic's repo,
ignition added as a side branch, Phase 3 redesigned, false-positive handling
written into Phases 2 and 6).

## The claim
Fine-tuning operates at the wrong level of abstraction. Not the wrong objective —
prediction goes all the way down — but error is computed at the token and only
trickles back to where conceptual prediction happens.

1. **A workspace exists** and is legible with the published lens. **ESTABLISHED.**
2. **Plasticity is graded by depth** — writing cheap at concept level, expensive
   below. Isolation emergent from density of commitment, not imposed. **UNTESTED**
   (Phase 1).
3. **A workspace-level write can be confined.** **UNTESTED** (Phase 3) — the crux,
   and most likely to be false.

## Established
- Lens validated on 7B and 14B: 4/4 baseline/swapped/moved, cross-check 0.9999.
  1.5B is a **weak pass** (3/4 swapped, threshold set after the fact).
- Skill corpus: 10/14 retained, 5 with >=95% gap.
- Concept injection on 7B: 1.7% -> 86.7% across the coherent window; random
  control never fires.
- The skill's activation change is only **1.23x enriched** in J-space over chance.
- Ordinary FT installs across the drift budget: 75% @ 0.083 -> 100% @ 0.883.

## Ruled out
**Cross-prompt activation patching cannot test the write path.** The differing
content sits in the non-shared prefix; token-aligned patching overwrites only the
shared suffix and leaves the prefix KV intact. A known-good item edit fails
identically to a skill once the readout moves past the patch boundary. Geometry,
not implementation.

## Retracted
Five headlines, all one error — building the measurement then asking what it
measured.

---

## PHASE 3 — Confinement (THE CRUX) — REDESIGNED. Runnable.
**Match at the OUTPUT, not the input.** Teacher runs `[spec + task]`, student
runs `[task]`. Prompts differ in length and content, but **the answer position
exists in both** — so no positional correspondence is needed.

    L = L_distil + lambda_orth * L_orth + lambda_size * ||delta_theta||^2

- **L_distil** = KL(teacher || student) at answer positions only, temperature T,
  teacher logits cached, teacher frozen. **Sweep T** — if T matters that is
  itself a result (the transferred thing is the neighbourhood, not the answer).
  **Never include the ground-truth label**, or this becomes FT with a regulariser.
- **L_orth** = drift in workspace directions of **unrelated held-out concepts**,
  on a held-out probe set **disjoint** from the collateral evaluation set.
- **lambda_size** = 0 while arms are bisected to fixed drift.

Mechanisms, ablated: soft penalty / gradient projection (GEM-OGD) / verify gate
(**rejection count is a result**), plus a **no-confinement arm** (distillation
alone).

Comparators: hard-label FT (the thing to beat); distillation without confinement;
the three mechanisms; random-direction control at matched norm.

**Arms 1 and 2 together answer the project's question**: is a workspace-level
write gentler than a token-level write at the same drift? If distillation alone
already beats hard-label FT on collateral, that is positive for the thesis and
confinement is a refinement, not the mechanism.

Running spec (the Job 4 fixes, now rules): equal search budget, bisect every arm;
**match on install first, compare collateral second** (install is a gate);
4-5 magnitudes; >=3 skills differing in shape; no LoRA; shared alignment function.

Collateral at three levels: behavioural, representational, distributional.

Gate: small collateral at matched install with a slope -> proceed to 5 and 6.
Large collateral or no separation from hard-label FT -> confinement was an
illusion; publish, it kills a plausible idea cheaply.

## PHASE 1 — Plasticity gradient. Cheap, not run.
Perturbation only, no fine-tuning. *Movement* and *cost* curves vs depth.
**The cost curve is the result; movement is context.** Flat cost kills emergent
isolation.

## PHASE 2 — Drift during training. Cheap, not run.
Pairwise angles among readout directions every ~20 steps. Smooth / snap / thrash.
**False-positive handling:** carry control concepts untouched by the task through
the identical measurement; report drift as **excess over the control
distribution**, never as a raw angle.

## PHASE 4 — Sense separation (optional). Gates the ambitious version only.
## PHASE 5 — Matched-drift forgetting. Expensive. `forgetting_jdrift.py` UNTESTED,
corpus never chosen. **Do not spend until Phase 3 reports.**
## PHASE 6 — The intervention. Gated on Phase 3. Known limit: readout is in
directions the model already has, so it should excel at **re-attachment** and may
silently approximate on **acquisition**. False-positive handling on extraction.
## PHASE 7 — Intake gating. Idea only. **Surprise is not contradiction.**

## SIDE BRANCH — Ignition. NOT on the critical path.
Pre-registered in `IGNITION-CRITERION-v2.md`. Inject at L, measure at L+k,
continuous quantity, readout direction independent of injection. Power analysis
before the sweep. **Expected outcome negative. Exactly one clean measurement,
not a campaign.**

## Order of operations
    3 -> (1, 2 in parallel, cheap) -> (4) -> 5 -> 6
Phase 3 is the gate. Phases 1 and 2 run **while** Phase 3 runs, not before.
Nothing spent on Phase 5 until Phase 3 reports.
