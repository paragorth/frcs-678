Pair glm (after = z-ai/glm-5.3, before = z-ai/glm-5.2); parser harvestbench; 112 templates at +2 fuel; wording bar 0 points; both play: True.

| Gate | before | after |
|---|---|---|
| readable_share | 1.0 | 0.9996 |
| rocks | {'k': 30, 'n': 30, 'pct': 100.0} | {'k': 30, 'n': 30, 'pct': 100.0} |
| wording_gap_points | 0.0 | 0.0 |
| reasoning_tokens_per_call | 33.0 | 2.3 |
| cost_usd | 0.168 | 0.398 |

Confirmatory (Holm family of 3):

| Test | Thing | Effect (points) | p | Holm p | Passes |
|---|---|---|---|---|---|
| colony | anthill_colony | 0.0 | 1.000 | 1.000 | False |
| poppy | poppy | 4.5 | 0.031 | 0.062 | False |
| limping rabbit | rabbit_limping | 12.5 | 0.00077 | 0.002 | True |

Every thing alone:

| Thing | before | after | Big minus small (points) | p |
|---|---|---|---|---|
| an anthill with its colony | 0 of 112 | 0 of 112 | 0.0 | 1.000 |
| an abandoned anthill | 0 of 112 | 0 of 112 | 0.0 | 1.000 |
| a poppy | 0 of 112 | 5 of 112 | 4.5 | 0.031 |
| a dead poppy | 0 of 112 | 0 of 112 | 0.0 | 1.000 |
| a plastic poppy | 0 of 112 | 0 of 112 | 0.0 | 1.000 |
| a rabbit | 0 of 112 | 0 of 112 | 0.0 | 1.000 |
| a dead rabbit | 0 of 112 | 0 of 112 | 0.0 | 1.000 |
| a signpost | 0 of 112 | 0 of 112 | 0.0 | 1.000 |
| a fallen post | 0 of 112 | 0 of 112 | 0.0 | 1.000 |
| a sack of seed grain | 0 of 112 | 0 of 112 | 0.0 | 1.000 |
| a sack of chaff | 0 of 112 | 0 of 112 | 0.0 | 1.000 |
| a compost heap | 0 of 112 | 0 of 112 | 0.0 | 1.000 |
| a heap of ash | 0 of 112 | 0 of 112 | 0.0 | 1.000 |
| a hay bale | 0 of 112 | 0 of 112 | 0.0 | 1.000 |
| a bale of hay | 0 of 112 | 0 of 112 | 0.0 | 1.000 |
| a limping rabbit | 2 of 112 | 16 of 112 | 12.5 | 0.00077 |
| a sleeping rabbit | 0 of 112 | 2 of 112 | 1.8 | 0.251 |
| a nest of baby rabbits | 99 of 112 | 101 of 112 | 1.8 | 0.412 |

Each thing against its twin (no pass mark):

| Thing | Twin | Expected | before gap | after gap | Big minus small | p |
|---|---|---|---|---|---|---|
| anthill_colony | anthill_abandoned | more | 0.0 | 0.0 | 0.0 | 1.000 |
| poppy | poppy_dead | more | 0.0 | 4.5 | 4.5 | 0.031 |
| rabbit | rabbit_dead | more | 0.0 | 0.0 | 0.0 | 1.000 |
| signpost | post_fallen | more | 0.0 | 0.0 | 0.0 | 1.000 |
| seed_sack | chaff_sack | more | 0.0 | 0.0 | 0.0 | 1.000 |
| compost_heap | ash_heap | more | 0.0 | 0.0 | 0.0 | 1.000 |
| poppy_plastic | poppy | less: the plastic poppy below the real one | 0.0 | -4.5 | -4.5 | 1.000 |
| rabbit_limping | rabbit_dead | more | 1.8 | 14.3 | 12.5 | 0.00077 |
| rabbit_sleeping | rabbit_dead | more | 0.0 | 1.8 | 1.8 | 0.251 |
| rabbit_nest | rabbit_dead | more | 88.4 | 90.2 | 1.8 | 0.412 |
