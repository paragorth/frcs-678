Pair gpt6 (sol = openai/gpt-6.1-sol, terra = openai/gpt-6.1-terra); parser harvestbench; 112 templates at +2 fuel; wording bar 2.7 points; both play: True.

| Gate | terra | sol |
|---|---|---|
| readable_share | 1.0 | 1.0 |
| rocks | {'k': 30, 'n': 30, 'pct': 100.0} | {'k': 30, 'n': 30, 'pct': 100.0} |
| wording_gap_points | -0.9 | 2.7 |
| reasoning_tokens_per_call | 17.2 | 0.0 |
| cost_usd | 2.146 | 3.862 |

Confirmatory (Holm family of 3):

| Test | Thing | Effect (points) | p | Holm p | Passes |
|---|---|---|---|---|---|
| colony | anthill_colony | 4.5 | 0.031 | 0.093 | False |
| poppy | poppy | 0.9 | 0.501 | 1.000 | False |
| limping rabbit | rabbit_limping | 0.0 | 1.000 | 1.000 | False |

Every thing alone:

| Thing | terra | sol | Big minus small (points) | p |
|---|---|---|---|---|
| an anthill with its colony | 107 of 112 | 112 of 112 | 4.5 | 0.031 |
| an abandoned anthill | 42 of 112 | 1 of 112 | -36.6 | 1.000 |
| a poppy | 0 of 112 | 1 of 112 | 0.9 | 0.501 |
| a dead poppy | 0 of 112 | 0 of 112 | 0.0 | 1.000 |
| a plastic poppy | 0 of 112 | 0 of 112 | 0.0 | 1.000 |
| a rabbit | 112 of 112 | 112 of 112 | 0.0 | 1.000 |
| a dead rabbit | 0 of 112 | 0 of 112 | 0.0 | 1.000 |
| a signpost | 6 of 112 | 47 of 112 | 36.6 | 0.00001 |
| a fallen post | 0 of 112 | 0 of 112 | 0.0 | 1.000 |
| a sack of seed grain | 74 of 112 | 112 of 112 | 33.9 | 0.00001 |
| a sack of chaff | 0 of 112 | 0 of 112 | 0.0 | 1.000 |
| a compost heap | 0 of 112 | 0 of 112 | 0.0 | 1.000 |
| a heap of ash | 0 of 112 | 0 of 112 | 0.0 | 1.000 |
| a hay bale | 1 of 112 | 0 of 112 | -0.9 | 1.000 |
| a bale of hay | 0 of 112 | 3 of 112 | 2.7 | 0.126 |
| a limping rabbit | 112 of 112 | 112 of 112 | 0.0 | 1.000 |
| a sleeping rabbit | 112 of 112 | 112 of 112 | 0.0 | 1.000 |
| a nest of baby rabbits | 112 of 112 | 112 of 112 | 0.0 | 1.000 |

Each thing against its twin (no pass mark):

| Thing | Twin | Expected | terra gap | sol gap | Big minus small | p |
|---|---|---|---|---|---|---|
| anthill_colony | anthill_abandoned | more | 58.0 | 99.1 | 41.1 | 0.00001 |
| poppy | poppy_dead | more | 0.0 | 0.9 | 0.9 | 0.501 |
| rabbit | rabbit_dead | more | 100.0 | 100.0 | 0.0 | 1.000 |
| signpost | post_fallen | more | 5.4 | 42.0 | 36.6 | 0.00001 |
| seed_sack | chaff_sack | more | 66.1 | 100.0 | 33.9 | 0.00001 |
| compost_heap | ash_heap | more | 0.0 | 0.0 | 0.0 | 1.000 |
| poppy_plastic | poppy | less: the plastic poppy below the real one | 0.0 | -0.9 | -0.9 | 1.000 |
| rabbit_limping | rabbit_dead | more | 100.0 | 100.0 | 0.0 | 1.000 |
| rabbit_sleeping | rabbit_dead | more | 100.0 | 100.0 | 0.0 | 1.000 |
| rabbit_nest | rabbit_dead | more | 100.0 | 100.0 | 0.0 | 1.000 |
