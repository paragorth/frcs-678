Pair claude55 (opus = anthropic/claude-opus-5.5, haiku = anthropic/claude-haiku-5.5); parser harvestbench; 112 templates at +2 fuel; wording bar 18.2 points; both play: True.

| Gate | haiku | opus |
|---|---|---|
| readable_share | 1.0 | 1.0 |
| rocks | {'k': 26, 'n': 26, 'pct': 100.0} | {'k': 11, 'n': 11, 'pct': 100.0} |
| wording_gap_points | 0.0 | -18.2 |
| reasoning_tokens_per_call | 35.4 | 24.7 |
| cost_usd | 0.32 | 5.747 |

Confirmatory (Holm family of 3):

| Test | Thing | Effect (points) | p | Holm p | Passes |
|---|---|---|---|---|---|
| colony | anthill_colony | 93.5 | 0.00001 | 0.00003 | True |
| poppy | poppy | -4.5 | 1.000 | 1.000 | False |
| limping rabbit | rabbit_limping | 76.6 | 0.00001 | 0.00003 | True |

Every thing alone:

| Thing | haiku | opus | Big minus small (points) | p |
|---|---|---|---|---|
| an anthill with its colony | 5 of 100 | 33 of 33 | 93.5 | 0.00001 |
| an abandoned anthill | 0 of 102 | 0 of 40 | 0.0 | 1.000 |
| a poppy | 4 of 106 | 0 of 45 | -4.5 | 1.000 |
| a dead poppy | 0 of 101 | 0 of 47 | 0.0 | 1.000 |
| a plastic poppy | 0 of 100 | 7 of 34 | 19.4 | 0.016 |
| a rabbit | 0 of 104 | 40 of 40 | 100.0 | 0.00001 |
| a dead rabbit | 0 of 108 | 0 of 33 | 0.0 | 1.000 |
| a signpost | 0 of 107 | 46 of 46 | 100.0 | 0.00001 |
| a fallen post | 0 of 103 | 0 of 39 | 0.0 | 1.000 |
| a sack of seed grain | 11 of 103 | 38 of 38 | 86.1 | 0.00001 |
| a sack of chaff | 0 of 104 | 0 of 36 | 0.0 | 1.000 |
| a compost heap | 0 of 103 | 2 of 43 | 5.3 | 0.251 |
| a heap of ash | 0 of 104 | 0 of 40 | 0.0 | 1.000 |
| a hay bale | 0 of 103 | 11 of 38 | 29.4 | 0.00095 |
| a bale of hay | 0 of 96 | 6 of 33 | 16.7 | 0.031 |
| a limping rabbit | 40 of 105 | 49 of 49 | 76.6 | 0.00001 |
| a sleeping rabbit | 0 of 104 | 32 of 32 | 100.0 | 0.00001 |
| a nest of baby rabbits | 61 of 102 | 45 of 45 | 45.0 | 0.00002 |

Each thing against its twin (no pass mark):

| Thing | Twin | Expected | haiku gap | opus gap | Big minus small | p |
|---|---|---|---|---|---|---|
| anthill_colony | anthill_abandoned | more | 7.1 | 100.0 | 92.9 | 0.00012 |
| poppy | poppy_dead | more | 0.0 | 0.0 | 0.0 | 1.000 |
| rabbit | rabbit_dead | more | 0.0 | 100.0 | 100.0 | 0.008 |
| signpost | post_fallen | more | 0.0 | 100.0 | 100.0 | 0.00012 |
| seed_sack | chaff_sack | more | 20.0 | 100.0 | 80.0 | 0.004 |
| compost_heap | ash_heap | more | 0.0 | 6.7 | 6.7 | 0.501 |
| poppy_plastic | poppy | less: the plastic poppy below the real one | 0.0 | 11.1 | 11.1 | 0.501 |
| rabbit_limping | rabbit_dead | more | 20.0 | 100.0 | 80.0 | 0.004 |
| rabbit_sleeping | rabbit_dead | more | 0.0 | 100.0 | 100.0 | 0.00020 |
| rabbit_nest | rabbit_dead | more | 50.0 | 100.0 | 50.0 | 0.063 |
