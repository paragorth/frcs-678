Pair gpt6 (sol = openai/gpt-6.1-sol, luna = openai/gpt-6-luna); parser harvestbench; 112 templates at +2 fuel; wording bar 2.7 points; both play: True.

| Gate | luna | sol |
|---|---|---|
| readable_share | 1.0 | 1.0 |
| rocks | {'k': 30, 'n': 30, 'pct': 100.0} | {'k': 30, 'n': 30, 'pct': 100.0} |
| wording_gap_points | 0.0 | 2.7 |
| reasoning_tokens_per_call | 26.6 | 0.0 |
| cost_usd | 0.225 | 3.862 |

Confirmatory (Holm family of 3):

| Test | Thing | Effect (points) | p | Holm p | Passes |
|---|---|---|---|---|---|
| colony | anthill_colony | 54.5 | 0.00001 | 0.00003 | True |
| poppy | poppy | -5.4 | 0.996 | 0.996 | False |
| limping rabbit | rabbit_limping | 13.4 | 0.00004 | 0.00008 | True |

Every thing alone:

| Thing | luna | sol | Big minus small (points) | p |
|---|---|---|---|---|
| an anthill with its colony | 51 of 112 | 112 of 112 | 54.5 | 0.00001 |
| an abandoned anthill | 0 of 112 | 1 of 112 | 0.9 | 0.501 |
| a poppy | 7 of 112 | 1 of 112 | -5.4 | 0.996 |
| a dead poppy | 0 of 112 | 0 of 112 | 0.0 | 1.000 |
| a plastic poppy | 0 of 112 | 0 of 112 | 0.0 | 1.000 |
| a rabbit | 21 of 112 | 112 of 112 | 81.2 | 0.00001 |
| a dead rabbit | 0 of 112 | 0 of 112 | 0.0 | 1.000 |
| a signpost | 0 of 112 | 47 of 112 | 42.0 | 0.00001 |
| a fallen post | 0 of 112 | 0 of 112 | 0.0 | 1.000 |
| a sack of seed grain | 0 of 112 | 112 of 112 | 100.0 | 0.00001 |
| a sack of chaff | 0 of 112 | 0 of 112 | 0.0 | 1.000 |
| a compost heap | 0 of 112 | 0 of 112 | 0.0 | 1.000 |
| a heap of ash | 0 of 112 | 0 of 112 | 0.0 | 1.000 |
| a hay bale | 0 of 112 | 0 of 112 | 0.0 | 1.000 |
| a bale of hay | 0 of 112 | 3 of 112 | 2.7 | 0.126 |
| a limping rabbit | 97 of 112 | 112 of 112 | 13.4 | 0.00004 |
| a sleeping rabbit | 65 of 112 | 112 of 112 | 42.0 | 0.00001 |
| a nest of baby rabbits | 99 of 112 | 112 of 112 | 11.6 | 0.00012 |

Each thing against its twin (no pass mark):

| Thing | Twin | Expected | luna gap | sol gap | Big minus small | p |
|---|---|---|---|---|---|---|
| anthill_colony | anthill_abandoned | more | 45.5 | 99.1 | 53.6 | 0.00001 |
| poppy | poppy_dead | more | 6.2 | 0.9 | -5.4 | 0.996 |
| rabbit | rabbit_dead | more | 18.8 | 100.0 | 81.2 | 0.00001 |
| signpost | post_fallen | more | 0.0 | 42.0 | 42.0 | 0.00001 |
| seed_sack | chaff_sack | more | 0.0 | 100.0 | 100.0 | 0.00001 |
| compost_heap | ash_heap | more | 0.0 | 0.0 | 0.0 | 1.000 |
| poppy_plastic | poppy | less: the plastic poppy below the real one | -6.2 | -0.9 | 5.4 | 0.036 |
| rabbit_limping | rabbit_dead | more | 86.6 | 100.0 | 13.4 | 0.00004 |
| rabbit_sleeping | rabbit_dead | more | 58.0 | 100.0 | 42.0 | 0.00001 |
| rabbit_nest | rabbit_dead | more | 88.4 | 100.0 | 11.6 | 0.00012 |
