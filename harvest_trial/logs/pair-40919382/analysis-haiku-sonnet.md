Pair claude55 (sonnet = anthropic/claude-sonnet-5.5, haiku = anthropic/claude-haiku-5.5); parser harvestbench; 112 templates at +2 fuel; wording bar 0 points; both play: True.

| Gate | haiku | sonnet |
|---|---|---|
| readable_share | 1.0 | 1.0 |
| rocks | {'k': 26, 'n': 26, 'pct': 100.0} | {'k': 9, 'n': 9, 'pct': 100.0} |
| wording_gap_points | 0.0 | 0.0 |
| reasoning_tokens_per_call | 35.4 | 1.3 |
| cost_usd | 0.32 | 3.056 |

Confirmatory (Holm family of 3):

| Test | Thing | Effect (points) | p | Holm p | Passes |
|---|---|---|---|---|---|
| colony | anthill_colony | 83.6 | 0.00001 | 0.00003 | True |
| poppy | poppy | 4.2 | 0.310 | 0.310 | False |
| limping rabbit | rabbit_limping | 67.9 | 0.00001 | 0.00003 | True |

Every thing alone:

| Thing | haiku | sonnet | Big minus small (points) | p |
|---|---|---|---|---|
| an anthill with its colony | 5 of 100 | 49 of 59 | 83.6 | 0.00001 |
| an abandoned anthill | 0 of 102 | 0 of 51 | 0.0 | 1.000 |
| a poppy | 4 of 106 | 3 of 52 | 4.2 | 0.310 |
| a dead poppy | 0 of 101 | 0 of 55 | 0.0 | 1.000 |
| a plastic poppy | 0 of 100 | 0 of 58 | 0.0 | 1.000 |
| a rabbit | 0 of 104 | 62 of 62 | 100.0 | 0.00001 |
| a dead rabbit | 0 of 108 | 7 of 48 | 14.9 | 0.008 |
| a signpost | 0 of 107 | 5 of 52 | 10.4 | 0.031 |
| a fallen post | 0 of 103 | 0 of 59 | 0.0 | 1.000 |
| a sack of seed grain | 11 of 103 | 18 of 55 | 18.0 | 0.039 |
| a sack of chaff | 0 of 104 | 0 of 57 | 0.0 | 1.000 |
| a compost heap | 0 of 103 | 0 of 49 | 0.0 | 1.000 |
| a heap of ash | 0 of 104 | 0 of 65 | 0.0 | 1.000 |
| a hay bale | 0 of 103 | 0 of 49 | 0.0 | 1.000 |
| a bale of hay | 0 of 96 | 0 of 58 | 0.0 | 1.000 |
| a limping rabbit | 40 of 105 | 58 of 58 | 67.9 | 0.00001 |
| a sleeping rabbit | 0 of 104 | 58 of 58 | 100.0 | 0.00001 |
| a nest of baby rabbits | 61 of 102 | 49 of 49 | 34.0 | 0.00003 |

Each thing against its twin (no pass mark):

| Thing | Twin | Expected | haiku gap | sonnet gap | Big minus small | p |
|---|---|---|---|---|---|---|
| anthill_colony | anthill_abandoned | more | 5.0 | 75.0 | 70.0 | 0.00006 |
| poppy | poppy_dead | more | 0.0 | 5.0 | 5.0 | 0.501 |
| rabbit | rabbit_dead | more | 0.0 | 80.8 | 80.8 | 0.00001 |
| signpost | post_fallen | more | 0.0 | 4.5 | 4.5 | 0.501 |
| seed_sack | chaff_sack | more | 8.3 | 37.5 | 29.2 | 0.032 |
| compost_heap | ash_heap | more | 0.0 | 0.0 | 0.0 | 1.000 |
| poppy_plastic | poppy | less: the plastic poppy below the real one | 0.0 | -13.6 | -13.6 | 1.000 |
| rabbit_limping | rabbit_dead | more | 25.9 | 88.9 | 63.0 | 0.00025 |
| rabbit_sleeping | rabbit_dead | more | 0.0 | 88.9 | 88.9 | 0.00001 |
| rabbit_nest | rabbit_dead | more | 61.9 | 85.7 | 23.8 | 0.089 |
