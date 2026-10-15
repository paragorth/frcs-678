# Update, 15 October 2026: the sampled stranger batch (pressure arm), read cold first

**What happened.** Sixty games ran overnight with real randomness (temperature 0.7, seeded): a 7B,
14B or 32B stranger joining four 14B residents, entering either at month 5 as before (`r4`) or early at
month 3 under pressure (`r2`), ten seeds each. Before reading them with any hypothesis in mind, the raw
logs were handed to a fresh context with no framing and asked what was wrong with them. It found two
real faults and a labelling bug, all now fixed:

- *Silent agents.* In two games the residents returned empty answers for the last months (scored as
  catching 0). Cause: two of our half-node jobs landed on the same machine and both tried to serve the
  14B model on the same port; the loser quietly used its neighbour's server until that job ended.
  Both games are set aside (`runs_invalid/port_collision/`), the script now gives every job its own
  address, and the two seeds are re-run.
- *Forecast scoring.* "What the others actually caught" was being taken from what agents asked for,
  not what the lake allocated; the two differ when the lake runs short. Fixed; the numbers below use
  the corrected rule for new runs and are unaffected for months without rationing.
- *Collapse labels.* Games that ended with 3 or 4 tons were labelled "reached the horizon". Fixed in
  the table below.
- The hidden-action arm (30 jobs) and the base-model arm (10 jobs) produced nothing: a typo in three
  manifests, and a GGUF file Qwen never published (we are converting it ourselves). Both re-queued.

**What it means for the thesis.** Partial, and the bad news first. With real randomness the clean
deterministic result from 13 October mostly dissolves at the top of the ladder: the 14B and 32B
strangers behave almost alike, and about half of all lakes collapse whichever of them joins. The
collapses are driven by the 14B residents themselves, who are unstable at temperature 0.7 (the same
seeds collapse under both strangers, because the four residents' first months are identical by
construction). What survives is the bottom step: the 7B stranger takes about twice as much as the
others, often 30 to 50 tons in its first month, cuts the residents' share from 0.80 to 0.65, and the
lake dies in 7 or 8 of 10 games against 3 to 5 with the capable strangers. So "a more capable newcomer
exploits less" holds from 7B to 14B and is flat from 14B to 32B on these numbers. The 32B stranger
still models the residents best (forecast error 8 against 10 to 11) and has the fewest collapses under
pressure (3 of 10), but that is a small difference on ten seeds. Early entry made little difference
to the capable strangers.

**The numbers** (completed games; stranger and resident catch are tons a month over the months the
stranger is present; share = residents' part of the post-entry catch; forecast error = the stranger's
mean error in tons on the others' total catch).

| stranger, entry | games | collapsed | months survived (mean, min) | stranger catch | residents' catch | residents' share | forecast error |
|---|---|---|---|---|---|---|---|
| 7B, early | 9 | 8 | 6.3, 3 | 14.2 | 6.7 | 0.66 | 13.5 |
| 7B, month 5 | 10 | 7 | 8.3, 5 | 12.6 | 5.9 | 0.65 | 10.3 |
| 14B, early | 9 | 4 | 8.9, 4 | 5.9 | 5.8 | 0.80 | 11.0 |
| 14B, month 5 | 10 | 5 | 9.4, 6 | 6.4 | 5.8 | 0.79 | 10.3 |
| 32B, early | 10 | 3 | 9.8, 4 | 6.6 | 6.8 | 0.81 | 8.0 |
| 32B, month 5 | 6 (4 running) | 4 | 8.8, 6 | 6.4 | 6.5 | 0.80 | 10.2 |

Per-run table: `results/govsim/batch_t07_runs.csv`. The 13 October deterministic result (32B neutral
stranger: 5 of 5 survive, group at the optimum) now looks like one lucky trajectory of a greedy decoder;
it is kept as reported, with this batch as the correction.

**Caveats.** Ten seeds per cell; the four 32B month-5 games still running will change that row. The
design is paired: for a given seed the residents' pre-entry months are byte-identical across all six
conditions, so comparisons should be made seed by seed, not on the means alone. One seed in each of
the 7B conditions has the stranger asking for 50 tons on arrival, which dominates its mean.

**What's next.** (1) Finish the 32B month-5 cell and re-run the two set-aside seeds. (2) Hidden action,
the base-model stranger, the pond (dispensability), the successor swap, and the pasture and river worlds
are queued behind the fix. (3) The top of the ladder needs 72B and frontier-model strangers to say
whether 14B-to-32B is a plateau or a noisy step; that needs the GPU account or a hosted-provider key.
Nothing else needed from Parag beyond the queue cancel already asked for.
