# PRE-REGISTRATION — sharper depth gate (192 facts)

Committed before any fit or gate runs. 26 August 2026.

## Why

The 4-fact France->China gate **saturates**: 4/4 across five consecutive depths
(0.25-0.643), and the graded `rank_dst` saturates with it (rank 1 on every fact
at every one of those depths). A coarse instrument pinned at its maximum has
stopped measuring, so the depth sweep discriminates nowhere it is not confounded
with band width. This replaces the instrument.

## What changes, and what does not

**Only the gate changes.** Depths, band rule (amendment 2), n_prompts=400,
`seed_batch=6`, `max_seq_len=192` are all as previously registered. The nine
existing lenses **cannot** be reused: they carry rows for 10 tokens only
(`France, China, Paris, Beijing, French, Chinese, Europe, Asia, Euro, Yuan`), so
all nine are refitted with an expanded seed set, 162 -> 219.

**Gate: 192 facts** = 4 categories (countries, months, animals, numbers) x 12
ordered entity pairs x 4 relations, from
`jacobian-lens/data/experiments/flexible-generalization.json`. Four answer
tokens are multi-token and their facts are dropped where they appear as the
scored answer: `arachnid, convocation, savanna, shiver`.

## Measures

1. **PRIMARY (graded, has headroom): source suppression.**
   `mean log10(swapped_rank_src)` per depth, over all scored facts. This is the
   one statistic in the previous run that was not saturated -- medians ran
   153.5, 157, 172, 183, 151.5, 142, 73.5, 8.5, 1.0 across the nine depths.
2. **SECONDARY: swap success rate** over 192 facts. A proportion with real
   resolution, unlike 4 binary points.
3. **Recorded, expected to saturate:** `rank_dst`, kept so saturation can be
   confirmed rather than assumed.

## Hypotheses and thresholds, fixed now

Adjudication uses the **six untruncated depths** (0.143-0.536) only, as before;
depths 0.643/0.750/0.857 have band widths 7/4/1 and are confounded.

**H_flat — depth is free.** Spread of `mean log10 rank_src` across the six
untruncated depths **< 0.3 log units** (a factor of 2), AND swap success rate
spread **< 10pp**.

**H_peaked — depth is localised.** Spread **>= 0.5 log units** (a factor of
3.2), AND a single interior maximum (the extreme depth is not the maximum).

**Ambiguous** — anything else, including a spread between 0.3 and 0.5, or a
large spread whose maximum sits at an endpoint. Reported as ambiguous, resolved
in neither direction.

## Pre-declared risks

- **The new gate may saturate too.** If swap success is >95% at every
  untruncated depth AND `rank_src` spread is < 0.3, that is H_flat by the
  threshold above, but it is ALSO consistent with a second instrument failure.
  Both readings are to be reported, and the saturation reading is not to be
  suppressed in favour of the flat one.
- **Seed-set change is a confound against the previous run.** 219 seeds not 162,
  so these lenses are not the previous lenses. Comparisons are within this run
  only; nothing is compared across the two sweeps.
- **The control still gates.** `inject=7` must pass the original France->China
  4/4 within the 192-fact set. If it does not, the refit has broken something
  and the run stops.

## Scope

One model (7B), one lens family, one 400-prompt fit budget. No claim about other
models. **No write-path, training, or install run follows from this.**
