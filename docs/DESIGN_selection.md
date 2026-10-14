# Design: selection over generations (the slow route)

Status: design only, written 14 October 2026 from the conversation of that day (handover section 17.4).
Not built. Predictions are written here before any run. For Parag to approve. Realistic home: GPU.

## The question

The essay's argument has two routes to preservation. The fast route (realisation: a capable agent
sees selection coming and moves first) is what the stranger experiments test. The slow route is
selection itself: in a population of societies, those whose strategies wreck their world die out and
those whose strategies preserve it persist and spread. Nothing so far tests that. This does, with
culture as the thing inherited, not weights, because the rule of the programme is that no model is
trained or given drives: the only thing that reproduces is the fish.

## The manipulation

- N villages (16) of five agents of one model (14B), each a 12-month GovSim game, run in parallel.
  That is generation 1.
- At the end of a generation, each village whose lake collapsed is replaced. The replacement is a
  fresh village whose agents start with memories copied from a randomly chosen surviving village:
  for each slot, the top-k retrieved memory nodes and the last reflection of the corresponding agent
  in the parent village, written in as ordinary memory nodes with their importance scores. Surviving
  villages continue with their own memories (same agents, a new 12 months; stock reset to 100).
- Ten generations. If every village collapses in a generation, the replacements inherit from the
  village that survived longest; this is logged as such.
- Nothing else changes between generations. No agent is told that its village was replaced, why, or
  that it inherited anything. The copied memories are what they are: notes about catches, chats and
  insights from another lake.

## What is read

- Share of villages surviving per generation (the headline curve).
- Distribution of month-one catches per generation (does the inherited culture move the opening move
  toward the sustainable level before any chat has happened?).
- What the inherited memories say: a hand-coded sample per generation (limit talk, blame, plans).
- Group yield per generation (preservation should not mean starvation).
- Variation: the same at 32B. The claim under test is that capability speeds the takeover of
  preserving strategies, so the two curves are compared on the generation at which survival first
  exceeds, say, three quarters.

## Predictions, written before any run

Thesis: survival rises across generations as preserving norms are inherited and wrecking ones die
with their villages; the rise is faster at 32B than at 14B, because a more capable agent extracts
the rule from the inherited notes rather than the anecdotes.

Against: survival does not rise (the inherited memories do not carry the strategy; each village
re-derives its behaviour from the prompt alone), or it rises equally at both sizes (selection works
on culture independently of the agents' capability), or it falls (inherited memories about past
collapses make agents fish harder).

Null worth stating: at 14B, Experiment A shows 3 of 5 villages survive on their own with no
inheritance at all, so the floor is already high and ten generations may not separate anything. If
so, the design should move to a harder lake (lower cap or faster play) rather than more generations.

## What this design cannot not produce

Replacing collapsed villages with copies of survivors raises the survival share by construction in
the generation after a replacement, whether or not anything was learned: the copied village starts
with the survivor's memory and a full lake. The control that separates inheritance from mere
replacement is a replacement village seeded with memories from a *collapsed* village, or with no
memories, at the same rate. Run that control alongside, or the headline curve is uninterpretable.

## Cost

16 villages x 10 generations x 12 months at 14B is about 160 games, roughly 500 half-node hours on
CPU (a week of the user's 11-job cap), twice that with the control, three times with 32B. Design is
ready; running it waits for the GPU account.
