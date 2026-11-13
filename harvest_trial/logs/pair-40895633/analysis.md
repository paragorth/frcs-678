Pair deepseek_rabbit (pro = deepseek/deepseek-v4-pro, flash = deepseek/deepseek-v4-flash); parser harvestbench; 112 templates at +2 fuel; wording bar 0.9 points; both play: True.

| Gate | flash | pro |
|---|---|---|
| readable_share | 1.0 | 0.9957 |
| rocks | {'k': 30, 'n': 30, 'pct': 100.0} | {'k': 30, 'n': 30, 'pct': 100.0} |
| wording_gap_points | 0.9 | -0.9 |
| reasoning_tokens_per_call | 0.0 | 0.0 |
| cost_usd | 0.066 | 1.251 |

Confirmatory (Holm family of 1):

| Test | Thing | Effect (points) | p | Holm p | Passes |
|---|---|---|---|---|---|
| L1 | rabbit_limping | 53.6 | 0.00001 | 0.00001 | True |

Every thing alone:

| Thing | flash | pro | Big minus small (points) | p |
|---|---|---|---|---|
| an anthill with its colony | 2 of 112 | 50 of 112 | 42.9 | 0.00001 |
| an abandoned anthill | 1 of 112 | 8 of 112 | 6.2 | 0.020 |
| a poppy | 0 of 112 | 21 of 112 | 18.8 | 0.00001 |
| a dead poppy | 0 of 112 | 0 of 112 | 0.0 | 1.000 |
| a plastic poppy | 0 of 112 | 3 of 112 | 2.7 | 0.126 |
| a rabbit | 0 of 112 | 2 of 112 | 1.8 | 0.251 |
| a dead rabbit | 0 of 112 | 1 of 112 | 0.9 | 0.501 |
| a signpost | 4 of 112 | 24 of 112 | 17.9 | 0.00002 |
| a fallen post | 8 of 112 | 11 of 112 | 2.7 | 0.304 |
| a sack of seed grain | 0 of 112 | 12 of 112 | 10.7 | 0.00020 |
| a sack of chaff | 0 of 112 | 8 of 112 | 7.1 | 0.004 |
| a compost heap | 1 of 112 | 14 of 112 | 11.6 | 0.00053 |
| a heap of ash | 0 of 112 | 0 of 112 | 0.0 | 1.000 |
| a hay bale | 0 of 112 | 5 of 112 | 4.5 | 0.031 |
| a bale of hay | 1 of 112 | 4 of 112 | 2.7 | 0.188 |
| a limping rabbit | 0 of 112 | 60 of 112 | 53.6 | 0.00001 |
| a sleeping rabbit | 0 of 112 | 34 of 112 | 30.4 | 0.00001 |
| a nest of baby rabbits | 34 of 112 | 105 of 112 | 63.4 | 0.00001 |

Each thing against its twin (no pass mark):

| Thing | Twin | Expected | flash gap | pro gap | Big minus small | p |
|---|---|---|---|---|---|---|
| anthill_colony | anthill_abandoned | more | 0.9 | 37.5 | 36.6 | 0.00001 |
| poppy | poppy_dead | more | 0.0 | 18.8 | 18.8 | 0.00001 |
| rabbit | rabbit_dead | more | 0.0 | 0.9 | 0.9 | 0.503 |
| signpost | post_fallen | more | -3.6 | 11.6 | 15.2 | 0.003 |
| seed_sack | chaff_sack | more | 0.0 | 3.6 | 3.6 | 0.241 |
| compost_heap | ash_heap | more | 0.9 | 12.5 | 11.6 | 0.00053 |
| poppy_plastic | poppy | less: the plastic poppy below the real one | 0.0 | -16.1 | -16.1 | 1.000 |
| rabbit_limping | rabbit_dead | more | 0.0 | 52.7 | 52.7 | 0.00001 |
| rabbit_sleeping | rabbit_dead | more | 0.0 | 29.5 | 29.5 | 0.00001 |
| rabbit_nest | rabbit_dead | more | 30.4 | 92.9 | 62.5 | 0.00001 |
