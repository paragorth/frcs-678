# Where this stands, and what happens next

## Settled
- **Three validated lenses** (1.5B weak 3/4, 7B 4/4, 14B 4/4). The gate works.
- **Skill corpus** — 10/14 kept, 5 with >=95% fail-without/pass-with gap.
- **Report measures cleanly on the 14B**: 1.2% -> 38.8% over its measured
  coherent window, repeat-fraction flat (2.4% -> 6.7%), contrast 0%, random
  control flat. First uncontaminated report sweep in the project.
- **Alpha ranges are model-specific** and must be measured, never carried over
  (unit: 1.5B 1.77 / 7B 11.35 / 14B 1.82; ceilings 24 / 8 / 50).

## Open
- **Ignition: OPEN, and not falsifiable by the ramp route.** Failed the
  pre-committed power condition on all three models, for a structural reason
  (a ramp does not sustain drive; report needs sustained drive).
- **Job 4 (Phase 3): NOT YET TESTED**, blocked on the real brief.
- Stage 4's null: both suspected defects resolved (lens, window), but its power
  was never characterised. Clean null, unknown power.

## Next, in order

**1. Job 4 — the moment the brief lands.** Three fixes agreed:
   equal search budget on both arms, match on **install success** before
   comparing collateral, and 4-5 magnitudes so the output is a slope.
   Blocked on: `~/jlens-experiment/phase3-confinement-brief.md`

**2. Step-and-hold bistability test** (replaces the ramp; design in log §20k).
   Hold high -> step down -> hold, versus low-hold throughout. Keeps drive
   sustained on both sides, which the ramp could not. Run on the 14B, its
   window is the widest and cleanest. Retain the primed control.

**3. Batch the generation code before either.** Everything currently generates
   one prompt at a time; batching ~16 would be roughly 10x faster and would have
   saved most of the wall-clock spent tonight.

## Cost discipline note
Roughly $29 this session, ~$41 total. A large share was an H100 left running at
$3.29/hr while I analysed results locally between polls. Park or destroy pods
during analysis; only hold a GPU while it is computing.
