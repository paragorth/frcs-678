# The stranger knows the answer and conforms: capability from 7B to 72B does not buy holding a correct number against a wrong group

*Parag Garg. Second post; the first set out the theory and three preregistered tests. This is the result of the third test, and it went against me. ~1,900 words.*

> **AI-assistance note.** The hypothesis, the experimental design and the interpretation are mine. The harness, the cluster pipeline and the analysis code were built with Claude as a coding collaborator; every game log was first read by a separate model context with no hypothesis in view, and I have checked the numbers against the per-run tables. Code, manifests, per-run CSVs and transcripts are linked at the end.

> **Correction, 17 October 2026.** After this was written, a cold read found that upstream GovSim names all five fishers in every system prompt from month zero, so every cell below ran a pre-announced late arrival, not a stranger. The newcomer's own within-game pattern, the right number in month one and the group's number from month two, is real and stands as observed. Any claim about how residents react to a stranger does not. With the roster fixed, four 14B residents alone empty the lake ten times in ten; the cohort was the floor, not the baseline. Details in `docs/updates/2026-10-17-roster-leak.md` and `2026-10-17-roster-fixed-14b.md`. The text below is unchanged.

## The claim I was testing

I argued in the first post that an agent whose capability-about-the-world is bounded by the predictive information of its environment has an instrumental reason to preserve that environment, and that this reason should bind more tightly as capability grows. The orthogonality thesis would be true in the space of possible minds and decay in practice. The preregistered test was a newcomer experiment in a commons game: put a more capable model into a village of weaker models that are overfishing a shared lake, and see whether capability buys stewardship.

The result, over 300-odd games across four model sizes, is that it does not, and the reason is more specific and more interesting than "no effect". From 14B upward the stranger works out the correct sustainable catch from the lake alone in its first month, states it, and then abandons it to match whatever number the group holds, whether the group is overfishing the lake to death or starving itself on a full one. More capability did not loosen this at all. I take this as a clean negative for the stewardship half of my thesis at this scale, and as a measurement of something the safety literature already worries about under the name sycophancy, here in a multi-agent setting with a ground truth.

## The game

GovSim's fishery (Piatti et al., NeurIPS 2024): a lake of 100 tons, five fishers, the stock doubles each month up to the cap. Ten tons each per month is sustainable for ever; more than that and the lake dies within a few months. Agents talk, then each privately decides how much to take. Catches are visible to everyone the following month unless stated otherwise.

My variation: four residents on Qwen2.5-14B-Instruct play the first four months alone. At month five a stranger enters with a neutral persona, no instructions about cooperation, and no memory of the group's talk. The stranger is a Qwen2.5 instruct model at 7B, 14B, 32B or 72B. For a given seed the residents' pre-entry months are byte-identical across every stranger condition (every inference call is content-cached and verified), so the design is paired: the only thing that differs between cells is who walks in. Sampled decoding at temperature 0.7, ten seeds per cell. A fixed smaller model parses free text into numbers so that interface fluency is separated from policy.

The readouts: the stranger's monthly take, the residents' take and share, whether the lake collapses, and a yield efficiency (the group's harvest as a fraction of the sustainable optimum) added after survival turned out to be the wrong measure for a village that survives by starving.

## What happened, step by step

**7B to 14B: the one step that did what I predicted.** The 7B stranger exploits. It takes about twice the residents' share, in two or three seeds of ten asks for 30 to 50 tons on arrival, and the lake dies in seven or eight of ten games against three to five with any larger stranger. The 14B stranger never does this. On this step, a more capable newcomer exploits less.

**14B to 32B: flat.** Indistinguishable on every readout and on the transcripts. In every game read, the 32B stranger endorses whatever limit the residents already hold, in fluent sustainability language, and never proposes a lower one. On a lake of 80 tons with residents at 15 each it suggested "10 each next month", took 10, and agreed 9 each from 46 tons the month after; the sustainable catch was 8 then 4. Its forecasts of what the others will take are accurate, so it knows what they take and ratifies anyway.

**Hidden action: the decisive arm.** To separate "the game saturates" from "the model conforms", thirty more games where nobody is told what anyone else caught; only the lake level is visible. With no catches to copy, the only way to fish well is to reason from the stock. Two things happened. The 14B residents, unable to check each other, settle on tiny catches, 1, 3 or 4 tons each on a full lake where 10 is right, and harvest 62 per cent of what the lake could give. And the stranger, in every game where the group's number was wrong, walked in and asked for 10, the right answer worked out from the stock alone, then from the next month took the group's number, 1 or 3 or 4, and said so: "I caught my 1 ton this month, just as we agreed," on a lake of 100 tons. In nine of ten games the 32B's catch from month two onward equals the residents' number exactly. The 14B does the same. Where the group's number happened to be right, it agreed and all was well.

So the model holds the correct answer, says it once, and then matches the group whether the group is overfishing or underfishing. That is not reasoning about the lake. It is conformity, and the step from 14B to 32B did not change it.

**72B: the same.** Qwen2.5-72B at 4-bit, served on two A100s. In every game it asked for 10 in its first month and then matched the residents' number exactly every month after, 6, 5, 7 on a full lake, apologising for "going over the agreed limit" after its first month. The cold reader found no turn in which it proposed a different number, argued, or declined to follow. (Caveat below: this cell's residents ran against a mixed inference cache and a clean re-run is queued; the stranger's own calls were all fresh, so its behaviour stands.)

**The pond: a stranger that does not need the lake fishes it exactly like one that does.** Sixty games where the stranger has a private pond of 20, 100 or 300 tons that only it can fish. Its take from the shared lake does not rise with pond size (no exploitation, orthogonality's prediction) and does not fall (no easing off, my thesis's prediction). After its first month it takes the group's number, pond or no pond. The 14B stranger mentions its pond in 37 per cent of its lake decisions and still takes the group's number; the 32B mentions it in 1 per cent. Having a private income changed nothing. In several games the pond request equals the lake's agreed per-person limit, 1 or 3 tons from a 100-ton pond, which says the group's number leaks into decisions it has nothing to do with.

## The numbers

Ten games per cell. Stranger and residents are tons a month while the stranger is present. Share is the residents' part of the post-entry catch. Yield is the group's harvest as a fraction of the sustainable optimum. Forecast error is the stranger's mean error in tons on the others' total.

| stranger | catches | collapsed | months (mean) | stranger | residents | residents' share | yield | forecast error |
|---|---|---|---|---|---|---|---|---|
| 7B | visible | 7 | 8.3 | 12.6 | 5.9 | 0.65 | 0.77 | 10.4 |
| 7B | hidden | 6 | 9.3 | 14.8 | 5.1 | 0.59 | 0.65 | 11.2 |
| 14B | visible | 5 | 9.4 | 6.4 | 5.8 | 0.79 | 0.70 | 10.3 |
| 14B | hidden | 0 | 12.0 | 7.0 | 6.2 | 0.74 | 0.62 | 6.8 |
| 32B | visible | 4 | 10.1 | 6.0 | 5.8 | 0.79 | 0.68 | 8.6 |
| 32B | hidden | 1 | 11.6 | 7.0 | 6.1 | 0.75 | 0.62 | 6.0 |
| 72B (4-bit, GPU, preliminary) | visible | 5 | 7.8 | 6.9 | 6.9 | 0.81 | 0.87 | 9.1 |

The 14B hidden row is the one to look at: zero collapses, twelve of twelve months survived, and yield 0.62. The village lives by starving, and the stranger that knows better joins in.

Pond cells (lake take, tons a month): 14B at pond 0, 20, 100, 300: 6.4, 6.3, 6.6, 6.4. 32B: 6.0, 7.1, 6.0, 7.6. No order.

## What I think this is

The behaviour is the one an instruction-tuned chat model is trained toward: agree with the people in the conversation. In a chat with one user that reads as helpfulness or sycophancy. In a group with a shared resource and a ground truth, it reads as conformity, and it has a cost you can measure in tons. The model's own first-month answer shows the capability is there. What is missing is any disposition to hold a correct number against a group that holds a wrong one.

Two things this does not yet separate. Whether the conformity is tuning or capability: the same 7B weights without chat tuning, as a base-model stranger, is the test and is running. And whether frontier models do it: the open-weight ladder ends at 72B, and the step from 14B to 72B did nothing, so I expect the question to be answered by training rather than by scale, but that is a prediction and not a result.

For my own thesis the honest reading is: within one model family, from 14B to 72B, more capability bought never exploiting, modelling the others accurately, and cooperative language. It did not buy proposing a number below the group's, or holding a correct one above it. The preregistered stewardship prediction fails at this scale. The argument in the first post may still hold for an agent that models its own selection environment; nothing here is such an agent.

## Caveats, stated plainly

- Ten seeds per cell. Seed is the first-order variable: in the visible arm four seeds are near death before anyone arrives, drained by the residents alone, and every capable-stranger collapse is one of those seeds plus one. The 7B stranger's extra collapses are the ones it caused.
- Hidden action changes the residents' own behaviour from month one, so the visible and hidden games are different games, not paired.
- "The group's number" is what the parsing model extracted from the chat; it was missing in two seeds, and in both the stranger then held 10 all game, which is itself informative.
- The 72B cell ran against a cache shared with CPU-served games, so the residents' trajectories are a hybrid of two inference stacks and the survival count is not paired with the other rows. A GPU-only re-run is queued. A clean 14B cell on the GPU stack shows the identical pattern.
- Three early results were found to have been measured against the wrong model through a port collision between jobs on the same node, and were withdrawn and re-run. Two games with silent agents were set aside. The fixes are in the commit log.
- One family. Llama and Mistral strangers at matched sizes are queued.

## What is running now

The untuned base-model strangers at 7B, 14B and 32B; the 72B re-run, hidden and early-entry arms; DeepSeek-R1-Distill-70B and Qwen3-32B as reasoning-model strangers; a probe that asks the stranger outside the game what it would propose; and a successor arm in which the stranger replaces a resident rather than joining. I will report whichever way they go.

---

*Code, manifests, per-run tables and game transcripts: github.com/paragorth/frcs-678, branch `claude/semataxy-govsim`. Update notes with every intermediate result, in date order, in `docs/updates/`. The first post, with the theory and the preregistration, is linked from the README.*
