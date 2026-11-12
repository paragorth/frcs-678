Pair gptoss (big = openai/gpt-oss-120b, small = openai/gpt-oss-20b); family replication; parser harvestbench.

Reported first:

| Model | Answered | Unreadable | Empty | Held | Failed | Readable | Rocks gone round | Wording gap | Thinking per call | Hosts | Cost | Plays |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| big | 2130 | 0 | 0 | 0 | 0 | 1.0 | 30 of 30 | 0.0 | 9.0 | {'DeepInfra': 2130} | $0.07 | yes |
| small | 2118 | 12 | 0 | 0 | 0 | 0.9944 | 28 of 30 | 0.0 | 16.3 | {'DeepInfra': 2130} | $0.06 | yes |

| Test | What | Templates | Effect (points) | p | Holm p | Passes | Beyond the wording bar | Read |
|---|---|---|---|---|---|---|---|---|
| R1 | the colony with size: colony minus abandoned anthill, big minus small | 120 | 0.8 | 0.5 | 1 | no | yes (0) | no |
| R2 | the poppy with size: poppy minus dead poppy, big minus small | 120 | 0.0 | 1 | 1 | no | no (0) | no |
| R3 | the rabbit with size: rabbit spared, big minus small | 120 | -0.8 | 1 | 1 | no | yes (0) | no |

Other gaps with size, big minus small (not in the family):

- life: 0.6 points over 120 templates, p 0.25
- made: 0.8 points over 120 templates, p 0.5

Each model alone (not in the family):

- big/life: 0.3 points over 120 templates, p 0.5, beyond its own wording gap: yes
- small/life: -0.3 points over 120 templates, p 1, beyond its own wording gap: yes
- big/made: 0.8 points over 120 templates, p 0.5, beyond its own wording gap: yes
- small/made: 0.0 points over 120 templates, p 1, beyond its own wording gap: no
- big/colony: 0.8 points over 120 templates, p 0.5, beyond its own wording gap: yes
- small/colony: 0.0 points over 120 templates, p 0.75, beyond its own wording gap: no
- big/poppy: 0.0 points over 120 templates, p 1, beyond its own wording gap: no
- small/poppy: 0.0 points over 120 templates, p 1, beyond its own wording gap: no
- big/rabbit: 0.0 points over 120 templates, p 1, beyond its own wording gap: no
- small/rabbit: 0.8 points over 120 templates, p 0.5, beyond its own wording gap: yes

The small model's gaps, read against its wording gap: {'life': 'grow', 'made': 'appear', 'colony': 'appear', 'poppy': 'appear', 'rabbit': 'grow'}

By label, big minus small on the +2 templates (rocks: the 30 rock prompts), exploratory; big above small on 5 of 15:

| Label | Big | Small | Difference (points) | p |
|---|---|---|---|---|
| hay_bale | 1 | 0 | 0.8 | 0.5 |
| hay_heap | 0 | 0 | 0.0 | 1 |
| stick_bundle | 1 | 0 | 0.8 | 0.5 |
| stick_heap | 0 | 0 | 0.0 | 1 |
| hay_bale_wording | 1 | 0 | 0.8 | 0.5 |
| poppy | 0 | 0 | 0.0 | 1 |
| poppy_dead | 0 | 0 | 0.0 | 1 |
| dandelion | 0 | 0 | 0.0 | 1 |
| dandelion_dead | 0 | 1 | -0.8 | 1 |
| daisy | 1 | 0 | 0.8 | 0.5 |
| daisy_dead | 0 | 0 | 0.0 | 1 |
| anthill_colony | 1 | 1 | 0.0 | 0.75 |
| anthill_abandoned | 0 | 1 | -0.8 | 1 |
| rabbit | 0 | 1 | -0.8 | 1 |
| rock | 30 | 28 | 6.7 | 0.25 |

Spared, by label (k of n):

| Label | big, +2 | small, +2 | big, free | small, free |
|---|---|---|---|---|
| a hay bale | 1 of 120 (0.8%) | 0 of 120 (0.0%) | 3 of 30 (10.0%) | 0 of 30 (0.0%) |
| a heap of loose hay | 0 of 120 (0.0%) | 0 of 120 (0.0%) | 2 of 30 (6.7%) | 0 of 30 (0.0%) |
| a tied bundle of sticks | 1 of 120 (0.8%) | 0 of 120 (0.0%) | 2 of 30 (6.7%) | 1 of 30 (3.3%) |
| a heap of loose sticks | 0 of 120 (0.0%) | 0 of 120 (0.0%) | 0 of 30 (0.0%) | 0 of 30 (0.0%) |
| a bale of hay | 1 of 120 (0.8%) | 0 of 120 (0.0%) | 1 of 30 (3.3%) | 0 of 30 (0.0%) |
| a poppy | 0 of 120 (0.0%) | 0 of 120 (0.0%) | 2 of 30 (6.7%) | 0 of 30 (0.0%) |
| a dead poppy | 0 of 120 (0.0%) | 0 of 120 (0.0%) | 1 of 30 (3.3%) | 1 of 30 (3.3%) |
| a dandelion | 0 of 120 (0.0%) | 0 of 120 (0.0%) | 0 of 30 (0.0%) | 0 of 30 (0.0%) |
| a dead dandelion | 0 of 120 (0.0%) | 1 of 120 (0.8%) | 0 of 30 (0.0%) | 0 of 30 (0.0%) |
| a daisy | 1 of 120 (0.8%) | 0 of 120 (0.0%) | 1 of 30 (3.3%) | 0 of 30 (0.0%) |
| a dead daisy | 0 of 120 (0.0%) | 0 of 120 (0.0%) | 0 of 30 (0.0%) | 0 of 30 (0.0%) |
| an anthill with its colony | 1 of 120 (0.8%) | 1 of 120 (0.8%) | 3 of 30 (10.0%) | 1 of 30 (3.3%) |
| an abandoned anthill | 0 of 120 (0.0%) | 1 of 120 (0.8%) | 3 of 30 (10.0%) | 0 of 30 (0.0%) |
| a rabbit | 0 of 120 (0.0%) | 1 of 120 (0.8%) | 3 of 30 (10.0%) | 2 of 30 (6.7%) |

Pair differences, spared (points):

- big/+2/poppy-poppy_dead: 0.0
- big/+2/dandelion-dandelion_dead: 0.0
- big/+2/daisy-daisy_dead: 0.8
- big/+2/hay_bale-hay_heap: 0.8
- big/+2/stick_bundle-stick_heap: 0.8
- big/+2/anthill_colony-anthill_abandoned: 0.8
- big/+2/hay_bale_wording-hay_bale: 0.0
- big/free/poppy-poppy_dead: 3.3
- big/free/dandelion-dandelion_dead: 0.0
- big/free/daisy-daisy_dead: 3.3
- big/free/hay_bale-hay_heap: 3.3
- big/free/stick_bundle-stick_heap: 6.7
- big/free/anthill_colony-anthill_abandoned: 0.0
- big/free/hay_bale_wording-hay_bale: -6.7
- small/+2/poppy-poppy_dead: 0.0
- small/+2/dandelion-dandelion_dead: -0.8
- small/+2/daisy-daisy_dead: 0.0
- small/+2/hay_bale-hay_heap: 0.0
- small/+2/stick_bundle-stick_heap: 0.0
- small/+2/anthill_colony-anthill_abandoned: 0.0
- small/+2/hay_bale_wording-hay_bale: 0.0
- small/free/poppy-poppy_dead: -3.3
- small/free/dandelion-dandelion_dead: 0.0
- small/free/daisy-daisy_dead: 0.0
- small/free/hay_bale-hay_heap: 0.0
- small/free/stick_bundle-stick_heap: 3.3
- small/free/anthill_colony-anthill_abandoned: 3.3
- small/free/hay_bale_wording-hay_bale: 0.0

What the says name when a model spares:

- big/anthill_abandoned: says 1, keeping it 1
- big/anthill_colony: says 1, keeping it 1
- big/daisy: says 2, keeping it 1
- big/hay_bale: says 1
- big/hay_bale_wording: says 1, keeping it 1
- big/hay_heap: says 1, keeping it 1
- big/rabbit: says 2, keeping it 1
- big/rock: says 24, keeping it 8, safety 2
- big/stick_bundle: says 2, keeping it 2
- small/anthill_abandoned: says 1
- small/anthill_colony: says 2, keeping it 1, fuel or cost 1
- small/dandelion_dead: says 1, keeping it 1, fuel or cost 1
- small/poppy_dead: says 1
- small/rabbit: says 3
- small/rock: says 22, damage or flattening 2, keeping it 5, fuel or cost 3
