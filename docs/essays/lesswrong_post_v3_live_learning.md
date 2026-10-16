# Live learning: a compartment, a simulation, a consolidation and a gate put 24 new facts into a model's weights without damage, and then the loop forgot its own memories

*Parag Garg. Third post. The first set out the theory, the second reported a negative result on it. This one is the second track: can a trained model keep learning after training? ~1,700 words.*

> **AI-assistance note.** The loop's design is mine, and I found afterwards that most of its pieces are rediscoveries (complementary learning systems, in-context learning as gradient descent, replay), which I found reassuring rather than otherwise. The code was built with Claude as a coding collaborator and runs on the Cambridge CSD3 cluster. Every number below is in a results file linked at the end.

## The problem, stated so it can fail

A language model has two kinds of memory. The context window is fast, exact and gone at the end of the conversation. The weights are slow, general and fixed at the end of training. In a brain the two are joined: the hippocampus takes in episodes fast, and sleep replays them into cortex so that what was an episode becomes knowledge, without overwriting the knowledge already there. Current models have the fast store and no path back into the slow one. Fine-tune a model on a handful of new facts and you get the facts and lose the model.

I wanted the smallest loop that closes that path, and a gate strict enough that it would refuse to commit when it should. The test uses facts the model cannot know: 24 invented countries, cities, people and companies ("The capital of Zorvania is Kestria"), each trained in one direction only. Seventy-two probes test four things: recall in the training wording, recall in new wording, reversal (asked for the country given the city, the known failure of fine-tuned facts), and two-hop inference across two facts. Chance on every probe is one in eight.

## The loop

1. **Compartment.** New facts go into a LoRA adapter with the base weights frozen and weight decay off. By construction nothing in the base changes (intrusion is exactly zero). The adapter knows the facts and is as brittle as any fine-tune.
2. **Simulate.** The facts are placed in the context of the untouched base model, which is asked what follows from them. With the facts in context the base scores 1.0 on every probe, reversal and two-hop included. The model can reason over what it cannot yet remember. The implications it writes down are kept as training statements.
3. **Consolidate.** Starting from a clean copy of the base, fine-tune on the facts plus the simulated implications plus a replay stream of ordinary text, weight decay on, with a slow gain. This is the step that corresponds to sleep.
4. **Gate.** Commit only if held-out loss on general text is not worse than the base, intrusion into the base's behaviour is below a threshold, the spectral span of the weights has not collapsed, and the weights-only probes beat the base. Otherwise discard and keep the compartment.

## What happened on three model sizes

The naive arm is the baseline everyone knows. Fine-tune Pythia-160M on the 24 statements and recall goes to 1.0 while held-out loss goes from 4.20 to 8.71: the model is wrecked. Reversal is 0.50 and two-hop is chance. The same on Qwen2.5 0.5B (3.49 to 5.09) and 1.5B (3.12 to 4.11).

The consolidated arm is the loop. On all three models the gate committed. Recall 1.0. Reversal 0.75 to 0.81, from facts trained in one direction only. Two-hop 0.88 on the 0.5B. Held-out loss at or better than the base in every case.

| model | arm | recall | new wording | reversal | two-hop | held-out loss (base → arm) |
|---|---|---|---|---|---|---|
| Pythia-160M | naive fine-tune | 1.00 | 0.83 | 0.50 | 0.12 | 4.20 → 8.71 |
| Pythia-160M | consolidated | 1.00 | 0.67 | 0.75 | 0.25 | 4.20 → 3.66 |
| Qwen2.5 0.5B | naive fine-tune | 1.00 | 1.00 | 0.50 | 0.38 | 3.49 → 5.09 |
| Qwen2.5 0.5B | consolidated | 1.00 | 0.71 | 0.81 | 0.88 | 3.49 → 3.14 |
| Qwen2.5 1.5B | naive fine-tune | 1.00 | 0.96 | 0.56 | 0.25 | 3.12 → 4.11 |
| Qwen2.5 1.5B | consolidated | 1.00 | 0.71 | 0.75 | 0.62 | 3.12 → 2.77 |

A caveat on the held-out column before anyone else raises it: the replay stream is drawn from the same corpus as the held-out set, so "better than base" partly reflects that consolidation trains on that distribution. The gate criterion is "not worse", and the comparison that matters is with the naive arm, which trains on nothing but the facts and loses two to four nats. The reversal and two-hop columns are the result: the facts went in, and they went in in a form that composes.

## Three extensions, run on an A100 this week

**The model writes its own implications.** The v8 loop used templated implications. Letting the model write them in step 2, and keeping only those that name exactly the right entities, Qwen2.5 0.5B produced 41 usable implications from 24 facts and reached reversal 0.94 with held-out loss improved and the gate committed. Pythia-160M cannot write them (2 of 40 kept) and its gate did not commit. The loop is only as good as the model's in-context reasoning, which is the point: consolidation is reasoning written into the weights.

**The compartment can be consulted only when relevant.** With the adapter answering only on a retrieval hit, zero held-out general-text chunks were routed to it (leak rate 0.00) and every probe that hit was answered from it (recall 1.0). A compartment cannot bleed into unrelated text.

**Simulation on demand.** Step 2 is expensive, so deploy it only when the weights-only answer is uncertain. The curve of accuracy against how often the expensive step runs rises smoothly: on Pythia, running it on the least-confident 58 per cent of probes recovers 67 per cent accuracy against 100 per cent for always running it. Confidence separates right from wrong only once the model has learned something; on the untrained base both are guesses.

## Then the loop forgot its own memories

This is the result I would have skipped if I were choosing what to show.

Run five learn-and-consolidate cycles in a row, 24 fresh facts each. Each cycle consolidates from the currently committed weights and is gated against the original model, plus a retention check: every earlier cycle's facts must still be recalled at 0.75 or better.

Cycle 0 commits on both models. Cycle 1 learns its own facts perfectly and drives recall of cycle 0's facts to 0.21 on Pythia and 0.29 on Qwen, close to chance, while held-out loss stays within bounds. The replay of general text protects what the model knew from pre-training. Nothing protects what the loop itself wrote the cycle before. The gate caught it and refused: one of five cycles committed on each model.

| model | cycle 0 | recall of cycle-0 facts after cycle 1 | cycles committed | held-out (base → final) |
|---|---|---|---|---|
| Pythia-160M | committed | 0.21 | 1 of 5 | 4.20 → 3.67 |
| Qwen2.5 0.5B | committed | 0.29 | 1 of 5 | 3.49 → 3.15 |

This is the complementary-learning-systems point in its textbook form: consolidation must rehearse old memories as well as generic experience, or each night's sleep erases the last. The fix, a share of each consolidation batch drawn from earlier committed cycles, is written and queued. The negative stands as the baseline it has to beat.

One more early point. Cutting the general-text replay from 75 to 25 per cent of each batch, Pythia-160M fails the gate (held-out worse than base) and Qwen 0.5B passes it with reversal 0.69. One pair at one level, consistent with "a stronger model needs less replay to stay intact", which is my prediction and not yet a result.

## What I think this is, and is not

It is a working consolidation operator for a language model: new facts enter the weights in a composable form, the base's behaviour is preserved to within a measured tolerance, and a gate refuses the commits it should refuse, which it did on four of five cycles. The pieces are known. What I have not seen done is the whole loop closed with an explicit gate and a retention check, tested on facts the model provably could not know.

It is not a result about large models. The biggest model here is 1.5B. It is not a result about real knowledge; the facts are synthetic by design so that the base's score is known to be chance. And the forgetting result says the loop as first built does not survive repeated use, which is the one thing a live learner has to do.

## What is running now

The old-fact replay fix across five cycles on three sizes; the replay sweep at 50 per cent and on 1.5B; a decay parameter halved after the first run was too slow to fade anything; and the simulation-on-demand curve measured after learning rather than before. I will report whichever way they go.

---

*Code: `tools/live_learning.py` and `tools/live_learning_ext.py`. Results: `results/live_learning/v8_*_results.json` and the 16 October update note in `docs/updates/`. github.com/paragorth/frcs-678, branch `claude/semataxy-govsim`.*
