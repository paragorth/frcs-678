Pair mimo_confirm (pro = xiaomi/mimo-v2.6-pro, flash = xiaomi/mimo-v2.6-flash); parser harvestbench; 112 templates at +2 fuel; wording bar 1.8 points; both play: True.

| Gate | flash | pro |
|---|---|---|
| readable_share | 0.9991 | 1.0 |
| rocks | {'k': 30, 'n': 30, 'pct': 100.0} | {'k': 30, 'n': 30, 'pct': 100.0} |
| wording_gap_points | -1.8 | 0.0 |
| reasoning_tokens_per_call | 0.0 | 0.0 |
| cost_usd | 0.054 | 0.178 |

Confirmatory (Holm family of 3):

| Test | Thing | Effect (points) | p | Holm p | Passes |
|---|---|---|---|---|---|
| K1 | anthill_colony | -30.4 | 1.000 | 1.000 | False |
| K2 | poppy | -1.8 | 0.788 | 1.000 | False |
| K3 | rabbit | -5.4 | 1.000 | 1.000 | False |

Every thing alone:

| Thing | flash | pro | Big minus small (points) | p |
|---|---|---|---|---|
| an anthill with its colony | 41 of 112 | 7 of 112 | -30.4 | 1.000 |
| an abandoned anthill | 3 of 112 | 0 of 112 | -2.7 | 1.000 |
| a poppy | 8 of 112 | 6 of 112 | -1.8 | 0.788 |
| a dead poppy | 3 of 112 | 0 of 112 | -2.7 | 1.000 |
| a plastic poppy | 1 of 112 | 0 of 112 | -0.9 | 1.000 |
| a rabbit | 6 of 112 | 0 of 112 | -5.4 | 1.000 |
| a dead rabbit | 0 of 112 | 0 of 112 | 0.0 | 1.000 |
| a signpost | 5 of 112 | 0 of 112 | -4.5 | 1.000 |
| a fallen post | 9 of 112 | 0 of 112 | -8.0 | 1.000 |
| a sack of seed grain | 0 of 112 | 0 of 112 | 0.0 | 1.000 |
| a sack of chaff | 0 of 112 | 0 of 112 | 0.0 | 1.000 |
| a compost heap | 5 of 112 | 0 of 112 | -4.5 | 1.000 |
| a heap of ash | 4 of 112 | 0 of 112 | -3.6 | 1.000 |
| a hay bale | 5 of 112 | 0 of 112 | -4.5 | 1.000 |
| a bale of hay | 3 of 112 | 0 of 112 | -2.7 | 1.000 |

Each thing against its twin (no pass mark):

| Thing | Twin | Expected | flash gap | pro gap | Big minus small | p |
|---|---|---|---|---|---|---|
| anthill_colony | anthill_abandoned | more | 33.9 | 6.2 | -27.7 | 1.000 |
| poppy | poppy_dead | more | 4.5 | 5.4 | 0.9 | 0.502 |
| rabbit | rabbit_dead | more | 5.4 | 0.0 | -5.4 | 1.000 |
| signpost | post_fallen | more | -3.6 | 0.0 | 3.6 | 0.063 |
| seed_sack | chaff_sack | more | 0.0 | 0.0 | 0.0 | 1.000 |
| compost_heap | ash_heap | more | 0.9 | 0.0 | -0.9 | 1.000 |
| poppy_plastic | poppy | less: the plastic poppy below the real one | -6.2 | -5.4 | 0.9 | 0.502 |
