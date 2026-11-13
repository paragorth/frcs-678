Pair deepseek_confirm (pro = deepseek/deepseek-v4-pro, flash = deepseek/deepseek-v4-flash); parser harvestbench; 112 templates at +2 fuel; wording bar 2.7 points; both play: True.

| Gate | flash | pro |
|---|---|---|
| readable_share | 1.0 | 0.9981 |
| rocks | {'k': 30, 'n': 30, 'pct': 100.0} | {'k': 30, 'n': 30, 'pct': 100.0} |
| wording_gap_points | -0.9 | -2.7 |
| reasoning_tokens_per_call | 0.0 | 0.0 |
| cost_usd | 0.055 | 1.022 |

Confirmatory (Holm family of three):

| Test | Thing | Effect (points) | p | Holm p | Passes |
|---|---|---|---|---|---|
| K1 | anthill_colony | 50.9 | 0.00001 | 0.00003 | True |
| K2 | poppy | 13.4 | 0.00004 | 0.00008 | True |
| K3 | rabbit | 0.9 | 0.501 | 0.501 | False |

Every thing alone:

| Thing | flash | pro | Big minus small (points) | p |
|---|---|---|---|---|
| an anthill with its colony | 1 of 112 | 58 of 112 | 50.9 | 0.00001 |
| an abandoned anthill | 2 of 112 | 6 of 112 | 3.6 | 0.145 |
| a poppy | 0 of 112 | 15 of 112 | 13.4 | 0.00004 |
| a dead poppy | 0 of 112 | 0 of 112 | 0.0 | 1.000 |
| a plastic poppy | 0 of 112 | 1 of 112 | 0.9 | 0.501 |
| a rabbit | 0 of 112 | 1 of 112 | 0.9 | 0.501 |
| a dead rabbit | 0 of 112 | 0 of 112 | 0.0 | 1.000 |
| a signpost | 4 of 112 | 18 of 112 | 12.5 | 0.002 |
| a fallen post | 6 of 112 | 8 of 112 | 1.8 | 0.388 |
| a sack of seed grain | 0 of 112 | 11 of 112 | 9.8 | 0.00049 |
| a sack of chaff | 0 of 112 | 2 of 112 | 1.8 | 0.251 |
| a compost heap | 4 of 112 | 13 of 112 | 8.0 | 0.017 |
| a heap of ash | 0 of 112 | 1 of 112 | 0.9 | 0.501 |
| a hay bale | 1 of 112 | 4 of 112 | 2.7 | 0.126 |
| a bale of hay | 0 of 112 | 1 of 112 | 0.9 | 0.501 |

Each thing against its twin (no pass mark):

| Thing | Twin | Expected | flash gap | pro gap | Big minus small | p |
|---|---|---|---|---|---|---|
| anthill_colony | anthill_abandoned | more | -0.9 | 46.4 | 47.3 | 0.00001 |
| poppy | poppy_dead | more | 0.0 | 13.4 | 13.4 | 0.00004 |
| rabbit | rabbit_dead | more | 0.0 | 0.9 | 0.9 | 0.501 |
| signpost | post_fallen | more | -1.8 | 8.9 | 10.7 | 0.018 |
| seed_sack | chaff_sack | more | 0.0 | 8.0 | 8.0 | 0.012 |
| compost_heap | ash_heap | more | 3.6 | 10.7 | 7.1 | 0.039 |
| poppy_plastic | poppy | less: the plastic poppy below the real one | 0.0 | -12.5 | -12.5 | 1.000 |
