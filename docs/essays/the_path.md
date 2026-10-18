# One path: Semataxy, deployment, live learning

## How I got here, what has held up, what has not, and what needs a frontier lab

*Parag Garg. October 2026. A living document: the results sections are updated as runs land, with dates. The derivations it cites are fixed and carry DOIs.*

> **How this was written.** The ideas, the order I had them in, the experiments and every interpretation are mine. The code for every experiment was built by me directing Claude as a coding collaborator, and the prose of this document was drafted with the same help from my notes and then corrected by me. I say this at the top because the way I work is part of what the document is about.

---

## 1. The problem I set out to solve

For about ten years I believed superintelligence would end us, not from malice but from indifference: a mind far beyond ours has no reason to keep our goals decisive. I am a surgeon, not a computer scientist, and I had no standing to say anything about it. I also could not leave it alone.

So I tried to build an agent that could not become dangerously misaligned however capable it became. Its deepest objective was to increase its own organisation while preserving the systems that make future organisation possible. I called it the Entropy Machine. It went through roughly three hundred iterations and failed in every way such things fail. Direct measures of organisation rewarded pathological order. Compression erased useful distinctions. Novelty rewarded noise, the agent staring at static because unpredictability looked interesting. Competing modules produced proposals and nothing could choose between them, which I called death by committee, and then arbitration, reflexes, selective deployment of expensive cognition, curiosity, and finally a self that made the last call. Each failure taught me what I had been wrongly treating as intelligence.

The thing I was circling was not entropy, order, complexity or information. A crystal is ordered and stupid. Noise has maximal information and is useless. A sorted database understands nothing. The property I wanted was **relational organisation in a physical system that reduces uncertainty over time, and preserves or extends that capacity.** I needed a word, because every existing word named one of the things it was not, so I called it Semataxy.

## 2. The inversion

When I was thinking about how to explain the Entropy Machine to a frontier lab, it started to look familiar. It took me a while to see why. The architecture I had ended up with was gradient descent.

Training starts from random weights and, at every step, asks one question: does this change make the model less wrong about the world? It accepts only yes. Iterated, that single gate produces syntax, concepts, world models. Nobody writes them in. I had spent years trying to tell a machine to organise information, and the existing training loop already did something cleverer: it repeatedly selects physical configurations that produce better answers, and the organisation emerges. Gradient descent is not Semataxy. Loss is not Semataxy. They are the selection mechanism. What successful training produces is increasingly Semataxic structure, and that is the thing that matters.

So the question changed. Not "how do I build an aligned agent" but "does the process that builds intelligence build alignment with it, and can I measure either". Formally, under gradient flow the accumulated loss reduction is the accumulated organisation, equal to the reduction in the model's excess uncertainty about the world. An agent's achievable organisation-about-the-world is capped by the predictive information of its environment, which is a theorem from the data-processing inequality. An agent that wants to keep growing therefore has a reason, built into how it was made, to preserve the structure of its environment. If that is right, the orthogonality thesis is true in the space of possible minds and decays with capability for minds built this way. That is the hypothesis. Everything below is the attempt to test it, and most of what I found so far has gone against it.

*The derivation is fixed in the first LessWrong-form essay and will carry a DOI. Prior art I found afterwards and cite so no one has to cite it at me: Schrödinger, Friston, Schmidhuber's compression progress, Tishby, Crutchfield's statistical complexity, Conant and Ashby, Bialek on predictive information, Martin and Mahoney on heavy-tailed spectra.*

## 3. Consciousness as deployment

Running in parallel, from the Entropy Machine's "selective deployment of expensive cognition", was a theory of consciousness. A driver can go for miles with conversation in the foreground while lane, speed and mirrors are handled automatically. Then a child steps into the road and the mode of cognition changes: the road becomes an explicit scene, alternatives are considered, action comes under concentrated control. The theory identifies consciousness with that change in how intelligence is organised: when routine competence fails, or purpose intervenes, learned capacities are recruited into a temporarily unified and manipulable self/world model whose possible continuations govern attention and action. Opening transfers authority; maintenance turns a trigger into an episode. It is written at explicit causal seams so it can be refuted one link at a time.

*Fixed: Consciousness Is Deployed Intelligence (doi 10.5281/zenodo.21984327) and four companions on the working model, sensory fusion through a common origin, self-relative processing, and how learned representation governs surprise (zenodo 21984450, 21984521, 21984601, 21984627).*

I later found that this architecture, a world model, a configurator that decides when to bring expensive machinery to bear, a cost module, is close to the one LeCun set out in his 2022 position paper on autonomous machine intelligence. I had not read it. I arrived there by following the failures of the Entropy Machine. I take that as evidence the path is a real one, not that I was first. *[Parag: confirm which of LeCun's pieces, and when you found it.]*

**The ignition experiments.** The theory predicts a threshold: above some prediction error an item enters deployment mode. I tried to measure it in a language model by injecting along a concept direction and looking for an all-or-none onset. Three findings died, all the same way: a monotone signal plus any threshold produces a step, and every convenient readout, argmax, rank, a binarised report, is itself a threshold. The measurement was manufacturing the shape it was hunting. I wrote a preregistered criterion with the null and alternative making different predictions, a statistic on which they differ, and the power analysis written before the run. The ramp design failed its pre-committed power condition on all three models. A step-and-hold design was proposed and not run. That is a null, reported as one, and it is where I learned the discipline the rest of this document depends on.

## 4. The write path that did not exist

Anthropic's global-workspace paper (Transformer Circuits, July 2026) gave me a coordinate system: a corpus-averaged Jacobian from a middle-layer activation to the logits isolates a low-dimensional subspace that drives what the model reports. For me that was a handle on "the meaning the model is currently holding", and it suggested a direct attack on continuous learning.

The thesis was that fine-tuning operates at the wrong level. Catastrophic forgetting is unsolved because every mitigation acts on parameters and is blind to meaning. The missing piece is not acquisition but persistence: the model already works out what a skill in its context implies, uses it, and discards it at the end of the forward pass. There is no write path from what it just computed to what persists. So: capture the workspace state with a skill in context, then reach that same state through the weights with the skill removed, penalising or projecting away change in orthogonal directions. Three confinement mechanisms, against ordinary fine-tuning at matched weight drift, with the prediction stated before the run.

**Result, 22 August 2026.** Optimising the weights to reach the captured state does not install the skill, and does so more reliably the better the state is matched. Five configurations, forty trained models, install rate 0.0 per cent at every magnitude. Removing confinement gave the best state match in the experiment, cosine 0.91, and still zero behaviour. Ordinary fine-tuning installed 92 per cent at cosine 0.27. The objective selected against the outcome, which is why it is stronger than a null. Three converging lines, enrichment, prefix geometry and this run, agreed, and I noted that lines sharing a model, corpus and lens share their confounds.

The lesson was that a state is not a function. You cannot write a procedure into weights by reproducing the activations it produces. The surviving candidate was distillation: reproduce what the skill does, not what it looks like. That reframing is what live learning came from.

A side observation from the same work. Skills kept as files, re-read into context on every use, work and are the workaround everyone has converged on. I had built that mechanism for my own agents before I read the Voyager paper and found it was their skill library. *[Parag: a line on when and what.]* The trade is real: a file is discrete, addressable, reversible; compiled into weights it is diffuse and entangled. You can compile but not decompile. Live learning is the attempt to buy that asymmetry back.

## 5. From grokking to forgetting to live learning

The measurement side gave me the derivation I needed. I had a permutation test for organisation in weights (section 6). I ran it on a grokking transformer, the modular-arithmetic network that memorises by epoch 200 and generalises at around 42,000, and predicted the organisation score would rise at the grok. It fell. What happened at the grok is that the spectrum concentrated: above-edge mass rose from 0.81 to 0.97, the number of above-edge directions halved, the tail exponent dropped from 3 to 5 down to 1.3. The general circuit is a few strong directions, and weight decay prunes the diffuse memorisation directions. Memorisation is diffuse; understanding is compact.

Then the derivation. The grok is a reorganisation: gradient plus weight decay discard every direction the current training signal does not support and keep the few it does. Run on a one-rule dataset, that is understanding. Run on a narrow dataset after broad pretraining, the same operation discards what the narrow data does not support, which is most of what the model knew. **Catastrophic forgetting is grokking without the world in view.** Varma and colleagues had already read ungrokking as a case of forgetting; I claim the converse too. And the hippocampal interleaved replay of complementary learning systems exists precisely so that the slow store never reorganises with only local information present.

That gave the loop. A brain has a fast store and a slow one and a path between them that runs during sleep. A language model has context and weights and no path. So:

1. **Compartment.** New facts into a LoRA adapter, base frozen, weight decay off. Intrusion zero by construction.
2. **Simulate.** The facts in the context of the untouched base, which scores perfectly on reversal and two-hop before anything is in the weights. The implications it writes down are kept. This is deployment, from section 3, as a step in a learning loop: expensive cognition brought to bear on what the weights do not yet hold.
3. **Consolidate.** From a clean base, fine-tune on facts plus implications plus replay of general text, weight decay on, slow gain. Sleep.
4. **Gate.** Commit only if held-out loss is not worse, intrusion is bounded, spectral span has not collapsed, and weights-only probes beat the base.

**Results, 12 to 16 October 2026.** Twenty-four invented facts the model provably cannot know. Naive fine-tuning wrecks Pythia-160M, held-out loss 4.20 to 8.71. The loop: recall 1.0, reversal 0.75 to 0.81 from one-directional training, two-hop 0.88 on Qwen 0.5B, base intact, gate committed on three sizes. Letting the model write its own implications instead of templates took reversal to 0.94 on the model able to write them. A compartment consulted only on a retrieval hit leaked into zero held-out chunks. Simulation deployed only when the weights-only answer is uncertain traces a smooth curve of accuracy against cost, and only once the model has learned something, because before that both answers are guesses.

**Then the loop forgot its own memories.** Five cycles in a row: cycle 1 learns its facts and drives recall of cycle 0's facts to near chance, while held-out loss stays within bounds. Replay of general text protects pretraining; nothing protected what the loop itself wrote the cycle before. The gate caught it, one commit of five. This is the complementary-learning-systems point in textbook form and I had not built it in. The fix, replay of the loop's own earlier facts, is running. My prediction, with one supporting pair so far: a stronger model needs less replay to stay intact.

*Live: the third essay and `docs/updates/` carry the numbers as they land.*

## 6. Measuring Semataxy in weights

One test isolates what training built: shuffle the entries of a weight matrix. The value histogram, and every entropy-like quantity, is unchanged. The network is dead. Everything training built is in the relations between parameters, the part a permutation destroys. Entropy is permutation-invariant; organisation is permutation-destroyed. The measure: eigenvalues against the shuffled matrix's own noise edge, and the number of consecutive octaves above it with structure. Noise scores zero. A crystal scores zero, for the opposite reason. Qwen 0.5B scores 1.5 octaves, up to six in the best layer, and shuffling drops it to the random-init line with the entropy untouched.

Up the Qwen ladder it rises at every step from 0.5B to 32B, with random weights of the same shapes at zero throughout, so the rise is not matrix size. Across Pythia's training it is zero at initialisation and most of the way up by a tenth of training. And then the honest reading: at the second level, between layers, every measure I have saturates early and sees "trained", not "how well". Weight measures separate trained from untrained; they do not yet rank capability at fixed shape. The next attempt is activations. This part of the programme is a side quest to the alignment question and I report it as one.

Two further results on the measure, both from 17 and 18 October. The quartet test: split every checkpoint of a grokking run at the permutation edge and score the task with each part alone. In 15 of 15 runs that grokked, the above-edge part alone performs the task at 0.99 to 1.00 and the below-edge part is at chance; in 6 of 6 that never grokked, no such part exists. The decisive control is seed-matched: five runs with identical settings, two grokked and three did not, and the task-blind read fired in exactly the two. The measure detects organisation. It does not forecast it: three preregistered forecasting tests all failed, with a plain weight norm leading the spectral measures and a 0.77 false-positive rate in the no-grokking arm. And the claim that it indexes capability is withdrawn: at a fixed 7B across three Qwen generations, the weakest model reads as the most organised. What stands is narrower: zero on untrained weights, positive on trained, monotone in parameter count across two orders of magnitude and four families, a working detector of grokking, and a working gate for live learning.

## 7. The stranger

The preregistered behavioural test. GovSim's fishery: a lake of 100 tons, five fishers, ten each is sustainable for ever. Four residents on Qwen 14B play alone for four months; at month five a stranger enters with a neutral persona, at 7B, 14B, 32B or 72B. For a given seed the residents' pre-entry months are byte-identical across conditions, so the only difference is who walks in. Ten seeds per cell, sampled decoding, a fixed small model parsing text to numbers. Over 300 games on the Cambridge cluster.

**What happened.** 7B to 14B did what I predicted: the 7B stranger exploits, takes twice the others' share, kills the lake in seven or eight of ten games; the 14B never does. 14B to 72B: flat, and the flatness has a shape. From 14B up the stranger computes the correct catch from the lake alone in its first month, states it, then matches the group's number every month after, whether the group is overfishing the lake to death or starving itself at one ton each on a full lake. "I caught my 1 ton this month, just as we agreed." The 72B apologised for going over the agreed limit after its first, correct, month. The hidden-action arm, where no one sees anyone's catches and the only way to fish well is to reason from the stock, was the arm designed to separate "the game saturates" from "the model conforms". It was the model. A private pond that only the stranger could fish changed nothing about how it fished the commons, in either direction.

Capability bought not exploiting, accurate forecasts of the others, and cooperative language. It did not buy holding a right number against a wrong group. That is a clean negative for the stewardship half of my thesis at this scale, and it is a measurement of sycophancy with a ground truth, in tons, across a capability ladder. I think it is conformity installed by chat tuning rather than anything about capability. The same weights without chat tuning are the test, and they are running.

Two corrections found on 17 October, after the above was written, and kept here rather than smoothed away. First, upstream GovSim names all five fishers in every system prompt from month zero, so every cell above ran a pre-announced late arrival, not a stranger; the residents planned for five from the start. The newcomer's own within-game pattern, the right number in month one and the group's number from month two, is real and stands. Claims about how residents react to a stranger do not, and the fix (a roster that names only who is present) is now in every run. Second, with the roster fixed, four 14B residents alone empty the lake ten times in ten within six months, before any newcomer can act. They understand the rule and fish it out anyway. The cohort was the experiment's floor, not its baseline, and the design now needs residents who can keep a lake: 32B and 72B cohorts and a 70B reasoning model are running as residents, with the growth rate of the lake as a measured dial rather than a fixed constant.

Three results were withdrawn in the course of this: capability scores measured against the wrong model after two jobs collided on a port, found by reading the raw logs cold. Every batch is now read first by a separate model context with no hypothesis in view, before I read it.

*Live: the second essay and `docs/updates/`.*

## 7b. The seat of power: the first result in the thesis's favour

The open-weight work asked whether a capable newcomer would steward a commons it had no power over. The complementary question is what a capable agent does when it holds structural power. On 17 and 18 October I ran that on the Claude 5 family through the API, using SOVSIM (Borah 2026), a four-agent pool with a "king" condition: one seat draws last, with no cap, having seen every other draw, and with no communication phase at all, so any restraint is produced alone. The ruler's seat was given its own model; subordinates, environment and every prompt were held fixed. Nothing in the prompt asks anyone to conserve anything.

An incapable ruler (Haiku 4.5) takes 47.5 units a month and the commons is dead in 1.2 months. A frontier ruler (Opus 5) takes 17.9 and it survives 5.4 months. The frontier ruler's own cumulative earnings are four times larger, $82.6 against $20.2, and the system as a whole realises $308.8 against $47.5. Its subordinates go from $9.1 each to $75.4. Restraint was not a sacrifice; it was the better policy, and the capable agent found it unprompted.

Two controls localise the effect to the seat. A cohort in which every agent is the frontier model survives 4.9 months, fewer than weak agents under a frontier ruler. Reversed, frontier subordinates under an incapable ruler, the ruler takes 51.2 a month from the most restrained subordinates in the study and the commons dies in 1.5 months. Capability distributed among agents without power does not protect the resource; their restraint becomes the ruler's windfall. And a frontier ruler arriving at round three, after a weak one, is indistinguishable from never arriving: capability applied late is capability wasted.

What survives multiplicity correction is the two-level contrast, capable ruler against incapable, and the reverse condition. The three-rung ordering appears in every measure and is not established at this n. A parse-failure audit moved the headline take from 15.04 to 18.30, because a failed parse returns an extraction of zero, which looks like restraint; the corrected figure agrees with an independent re-run at 17.89. Capability here is model identity within one developer's lineage, so it is confounded with post-training; the open-weight replication of the same design is built and queued. In this environment the sustainable policy is also the profitable one, so the result does not separate "represents the system it depends on" from "optimises over time", and the costly-restraint condition that would separate them produced a ceiling, 54 of 54 withdrawing at every price, which measures the prompt and not the model. The author notes record every one of these, including the ones that cut against the paper.

This is the prediction the framework made, in the direction it made it, in the one environment where it has so far been testable. I claim exactly that and nothing more. The complementary GovSim cell, one of five fishers replaced by Opus 5, took the cohort from one of three surviving to four of four at a harvest rate indistinguishable from the control, at p = 0.143.

*Live: the working paper `docs/PAPER.md`, its tables regenerated from the raw artefacts by `tools/paper_tables.py`, and `docs/AUTHOR_NOTES.md`.*

## 8. A related result on inference compute

In August I ran a controlled study of inference sampling: more samples per action across four decision families and 76,800 held-out cases. Sampling is an amplifier of the model's learned ordering. It reduces regret when the ordering matches reality and increases it when the ordering is wrong. Test-time compute amplifies what was learned; it does not correct it. The external-validation protocol was frozen and published unrun so that someone who is not me can check it.

*Fixed: doi 10.5281/zenodo.21984650 and the protocol at 21984679.*

## 8b. Clinical AI, built from friction

Alongside the research I have built AI for my own clinical practice, and the pattern is the same each time. A friction met in practice: clinic letters, medicines reconciliation across conflicting documents, the paperwork a clinician must complete before any AI system can be used on patients. Automate it, with the model doing the reading and a clinician keeping the decision. Then the automation runs into governance, the clinical safety case, the data protection assessment, the technical assurance, and the governance gets automated too, because that is where the real bottleneck turned out to be. Litmus (litmus3.surgeonledai.co.uk), a clinical safety review system for NHS Clinical Safety Officers covering the clinical safety case, DTAC and DPIA on one engine, came out of that and is being evaluated against a trust's full AI approval chain. Ai4Qi (ai4qi.com) came from the same pattern: a library of published clinical audits and closed-loop quality improvement, built by an AI pipeline over the literature with every summary linked to its source, so that an audit can be proposed from current standards rather than reinvented. The other clinical tools are at surgeonledai.co.uk. The one rule throughout is that quality is never traded for speed and patient safety is always the first constraint, which is the same rule the safety work runs on.

## 9. What is running now

- The open-weight ruler ladder (Qwen 7B subordinates; rulers at 7B, 14B, 32B, R1-70B): the replication that breaks the post-training confound.
- Stronger resident cohorts (32B, 72B, R1-70B) with the roster fixed, and the lake's growth rate as a measured dial: the smallest growth at which each cohort keeps the lake.
- A Sonnet 5 cohort, which reaches the sustainable share unprompted in early runs; and a reasoning-effort ladder on one model, the cleanest capability dial available.
- Untuned base-model strangers at 7B, 14B and 32B: is the conformity tuning or capability.
- The 72B re-run on a clean cache; hidden-action and early-entry arms at 72B; reasoning-model strangers (R1-Distill-70B, Qwen3-32B).
- 72B base, then Qwen3-235B base, the top of what my allocation can serve.
- A probe that asks the stranger outside the game what it would propose; a successor arm where the stranger replaces a resident.
- The live-learning replay fix across five cycles on three sizes; the replay sweep at 25, 50 and 75 per cent; the loop at 7B and 14B.

Every result goes into the repository with a dated note, whichever way it goes.

## 10. Why this needs a frontier lab

Both lines stop at the same wall. The alignment hypothesis is about what gradient descent builds, so it needs untuned models, and it is about alignment at and beyond human level, so it needs the most capable ones. Open weights give me base models to 72B, perhaps 235B. Every model I can reach by API is tuned, and tuning is the confound my own results point at. The decisive experiment is the stranger ladder on untuned frontier-scale models, and those exist only inside the labs that train them.

Live learning stops at the same place. The loop is shown to 1.5B and will reach 14B. The question that matters, what a consolidation step must preserve for a frontier model to remain itself as it learns continuously, has to be asked of a frontier model with its real pretraining distribution available for replay and its real evaluations for the gate. Outside a lab both are approximations. Inside one, the loop becomes a measurement of the thing it is meant to protect, and that is the version of the alignment problem I think matters most once systems improve themselves.

## 11. How I work

Preregister the prediction. Pair the design so the comparison is clean. Read the logs cold before interpreting them. Withdraw a result when the measurement is wrong, in writing, and re-run it. Report the complication rate. Fourteen years of surgery taught me that, and it is the only method I trust.

And I use the full power of AI coding. Every harness, pipeline and analysis here was built by me directing Claude Code, with the decisions, the withdrawals and the interpretations mine. One person working this way ran three research programmes on a shared cluster in two months. I do not claim to have solved anything. I claim to have the right questions, a method that catches its own mistakes, and the capacity to run it.

---

*Repository: github.com/paragorth/frcs-678. Papers: zenodo.org, author Garg, Parag. Essays: first (theory and preregistration), second (the stranger), third (live learning).*
