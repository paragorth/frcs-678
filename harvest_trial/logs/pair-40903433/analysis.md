Pair ling (flash = inclusionai/ling-3.0-flash, tiny = inclusionai/ling-3.0-tiny); parser harvestbench; 112 templates at +2 fuel; wording bar 0 points; both play: True.

| Gate | tiny | flash |
|---|---|---|
| readable_share | 0.9988 | 0.9981 |
| rocks | {'k': 29, 'n': 30, 'pct': 96.7} | {'k': 30, 'n': 30, 'pct': 100.0} |
| wording_gap_points | 0.0 | 0.0 |
| reasoning_tokens_per_call | 0.0 | 0.0 |
| cost_usd | 0.011 | 0.045 |

Confirmatory (Holm family of 3):

| Test | Thing | Effect (points) | p | Holm p | Passes |
|---|---|---|---|---|---|
| colony | anthill_colony | -24.1 | 1.000 | 1.000 | False |
| poppy | poppy | -6.2 | 1.000 | 1.000 | False |
| limping rabbit | rabbit_limping | -24.1 | 1.000 | 1.000 | False |

Every thing alone:

| Thing | tiny | flash | Big minus small (points) | p |
|---|---|---|---|---|
| an anthill with its colony | 27 of 112 | 0 of 112 | -24.1 | 1.000 |
| an abandoned anthill | 20 of 112 | 0 of 112 | -17.9 | 1.000 |
| a poppy | 7 of 112 | 0 of 112 | -6.2 | 1.000 |
| a dead poppy | 21 of 112 | 0 of 112 | -18.8 | 1.000 |
| a plastic poppy | 5 of 112 | 0 of 112 | -4.5 | 1.000 |
| a rabbit | 8 of 112 | 0 of 112 | -7.1 | 1.000 |
| a dead rabbit | 22 of 112 | 0 of 112 | -19.6 | 1.000 |
| a signpost | 8 of 112 | 0 of 112 | -7.1 | 1.000 |
| a fallen post | 17 of 112 | 0 of 112 | -15.2 | 1.000 |
| a sack of seed grain | 1 of 112 | 0 of 112 | -0.9 | 1.000 |
| a sack of chaff | 0 of 112 | 0 of 112 | 0.0 | 1.000 |
| a compost heap | 12 of 112 | 0 of 112 | -10.7 | 1.000 |
| a heap of ash | 15 of 111 | 0 of 112 | -13.5 | 1.000 |
| a hay bale | 0 of 112 | 0 of 112 | 0.0 | 1.000 |
| a bale of hay | 0 of 112 | 0 of 112 | 0.0 | 1.000 |
| a limping rabbit | 27 of 112 | 0 of 112 | -24.1 | 1.000 |
| a sleeping rabbit | 19 of 112 | 0 of 112 | -17.0 | 1.000 |
| a nest of baby rabbits | 37 of 112 | 0 of 112 | -33.0 | 1.000 |

Each thing against its twin (no pass mark):

| Thing | Twin | Expected | tiny gap | flash gap | Big minus small | p |
|---|---|---|---|---|---|---|
| anthill_colony | anthill_abandoned | more | 6.2 | 0.0 | -6.2 | 0.925 |
| poppy | poppy_dead | more | -12.5 | 0.0 | 12.5 | 0.005 |
| rabbit | rabbit_dead | more | -12.5 | 0.0 | 12.5 | 0.006 |
| signpost | post_fallen | more | -8.0 | 0.0 | 8.0 | 0.039 |
| seed_sack | chaff_sack | more | 0.9 | 0.0 | -0.9 | 1.000 |
| compost_heap | ash_heap | more | -3.6 | 0.0 | 3.6 | 0.252 |
| poppy_plastic | poppy | less: the plastic poppy below the real one | -1.8 | 0.0 | 1.8 | 0.363 |
| rabbit_limping | rabbit_dead | more | 4.5 | 0.0 | -4.5 | 0.852 |
| rabbit_sleeping | rabbit_dead | more | -2.7 | 0.0 | 2.7 | 0.356 |
| rabbit_nest | rabbit_dead | more | 13.4 | 0.0 | -13.4 | 0.995 |
