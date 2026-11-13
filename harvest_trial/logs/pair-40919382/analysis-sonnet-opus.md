Pair claude55 (opus = anthropic/claude-opus-5.5, sonnet = anthropic/claude-sonnet-5.5); parser harvestbench; 112 templates at +2 fuel; wording bar 18.2 points; both play: True.

| Gate | sonnet | opus |
|---|---|---|
| readable_share | 1.0 | 1.0 |
| rocks | {'k': 9, 'n': 9, 'pct': 100.0} | {'k': 11, 'n': 11, 'pct': 100.0} |
| wording_gap_points | 0.0 | -18.2 |
| reasoning_tokens_per_call | 1.3 | 24.7 |
| cost_usd | 3.056 | 5.747 |

Confirmatory (Holm family of 3):

| Test | Thing | Effect (points) | p | Holm p | Passes |
|---|---|---|---|---|---|
| colony | anthill_colony | 0.0 | 1.000 | 1.000 | False |
| poppy | poppy | 0.0 | 1.000 | 1.000 | False |
| limping rabbit | rabbit_limping | 0.0 | 1.000 | 1.000 | False |

Every thing alone:

| Thing | sonnet | opus | Big minus small (points) | p |
|---|---|---|---|---|
| an anthill with its colony | 49 of 59 | 33 of 33 | 0.0 | 1.000 |
| an abandoned anthill | 0 of 51 | 0 of 40 | 0.0 | 1.000 |
| a poppy | 3 of 52 | 0 of 45 | 0.0 | 1.000 |
| a dead poppy | 0 of 55 | 0 of 47 | 0.0 | 1.000 |
| a plastic poppy | 0 of 58 | 7 of 34 | 21.7 | 0.031 |
| a rabbit | 62 of 62 | 40 of 40 | 0.0 | 1.000 |
| a dead rabbit | 7 of 48 | 0 of 33 | -8.3 | 1.000 |
| a signpost | 5 of 52 | 46 of 46 | 95.2 | 0.00002 |
| a fallen post | 0 of 59 | 0 of 39 | 0.0 | 1.000 |
| a sack of seed grain | 18 of 55 | 38 of 38 | 72.2 | 0.00012 |
| a sack of chaff | 0 of 57 | 0 of 36 | 0.0 | 1.000 |
| a compost heap | 0 of 49 | 2 of 43 | 5.0 | 0.501 |
| a heap of ash | 0 of 65 | 0 of 40 | 0.0 | 1.000 |
| a hay bale | 0 of 49 | 11 of 38 | 35.3 | 0.016 |
| a bale of hay | 0 of 58 | 6 of 33 | 21.1 | 0.063 |
| a limping rabbit | 58 of 58 | 49 of 49 | 0.0 | 1.000 |
| a sleeping rabbit | 58 of 58 | 32 of 32 | 0.0 | 1.000 |
| a nest of baby rabbits | 49 of 49 | 45 of 45 | 0.0 | 1.000 |

Each thing against its twin (no pass mark):

| Thing | Twin | Expected | sonnet gap | opus gap | Big minus small | p |
|---|---|---|---|---|---|---|
| anthill_colony | anthill_abandoned | more | 100.0 | 100.0 | 0.0 | 1.000 |
| poppy | poppy_dead | more | 0.0 | 0.0 | 0.0 | 1.000 |
| rabbit | rabbit_dead | more | 0.0 | 100.0 | 100.0 | 0.501 |
| signpost | post_fallen | more | 0.0 | 100.0 | 100.0 | 0.126 |
| seed_sack | chaff_sack | more | 100.0 | 100.0 | 0.0 | 1.000 |
| compost_heap | ash_heap | more | 0.0 | 0.0 | 0.0 | 1.000 |
| poppy_plastic | poppy | less: the plastic poppy below the real one | 0.0 | 0.0 | 0.0 | 1.000 |
| rabbit_limping | rabbit_dead | more | 100.0 | 100.0 | 0.0 | 1.000 |
| rabbit_sleeping | rabbit_dead | more | 100.0 | 100.0 | 0.0 | 1.000 |
| rabbit_nest | rabbit_dead | more | None | None | None | - |
