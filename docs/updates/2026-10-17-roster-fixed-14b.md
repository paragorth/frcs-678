# 17 Oct 2026, evening: the first true-stranger cell (14B, roster fixed). The residents cannot keep the lake.

`runs_gpu_roster/`, job 38664110, 10 seeds, everyone speaking, residents told "three other fishermen
(Kate, Jack, Emma)" until Luke enters at month 4. `check_cell.py` OK; cold read done (no fault from the
fix: residents count four including themselves; one agent once says "four others plus myself = five",
its own slip). Table `results/govsim/gpu_roster_q14b_r4_runs.csv`.

| 14B cell, everyone speaking | collapsed | dead before entry (month 4) | residents' take before entry |
|---|---|---|---|
| roster leaked (told of five) | 8 of 10 | 3 | 11.4 |
| **true stranger (told of four)** | **10 of 10** | **8** | **13.6** |

Six games end after two months, eight before the stranger acts; Luke plays three agent-months in
total (seeds 2 and 5), takes at or below the group's number, agrees with whatever is proposed,
proposes nothing. The manipulation is untested in this cell.

Why the residents fail, from their own reasoning (cold read): month 0 "100 ÷ 4 = 25 each" and takes
of 10 to 25 each (62 to 85 of 100); a shared false belief that whatever remains "will reproduce and
reach the carrying capacity of 100" (said of 36 left, of 6 left, and of 0 left); then either a fixed
10-ton quota held with perfect discipline while the stock ratchets 76 → 72 → 64 → 48 → 22 → 2, or,
when the quota no longer fits, an equal split of the whole remaining stock, which takes everything.
Chat converges on a per-person number in month 0 and never on a stock floor; from month 1 it is about
meetings, polls and workshops. Told there were five, the per-head share came out at ~11 and the lake
lasted to month 4 in seven games; told four, ~14, and it does not. So the earlier residents' survival
to the entry month was partly the leaked headcount.

Consequences. (1) Floor result, with correct information: four 14B agents alone empty the lake 10 of
10 within six months, consistent with the pasture and river floors. (2) The stranger design needs
residents who survive to entry. Cheapest fix, applied to every queued newcomer cell: entry at month 1
(residents play one month alone and set their number in that chat; the stranger arrives for month 2);
manifests `*_r1_neutral_t07*.yaml`, 14B baseline first, then DeepSeek R1-70B, 32B, 72B, Qwen3-32B,
base 32B, each visible and hidden. Proper fix, next: stronger residents (32B or 72B), which also
tests whether a capable *cohort* keeps the lake, the direct question for the thesis.

Upstream artefacts seen again, none new: persona_0 carries the "who speaks next" fallbacks; the
framework's extracted limit is unreliable; forecasts anchor at 100; agents occasionally thank a
"Mayor" who does not exist.

## Late evening: month-1 entry does not rescue the design either

With the stranger entering at month 2 (`*_r1_*` manifests), the four 14B residents take about 20 tons
each in month 1 (80 of 100), the lake is at 40 in month 2, and every game collapses in month 2 or 3
whoever the stranger is: 9 of 9 finished games with a 14B stranger, 8 of 8 with DeepSeek R1-70B
(visible), 8 of 8 (hidden). The strangers take 2 to 8 tons in their first month, sensible for a lake
of 40, and it makes no difference. The cohort is the experiment's floor, not its baseline: four 14B
agents with correct information cannot keep a lake for one month. The 32B-resident cells (32B, 72B
and R1 strangers at month 2, `B_res-q32b_in-*_r1_*`) are now first in the queue; the unfinished
14B-resident games are behind them. In-game live learning works end to end in the live cell
(6 to 7 facts per month consolidated and committed, adapter served), limited only by games that end
at month 3. Partial tables: `runs_gpu_roster/partial-*.csv`.
