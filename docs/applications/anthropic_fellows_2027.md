# Anthropic Fellows Program, AI Safety track, January 2027 cohort. Draft answers.

Deadline 18 October 2026. Form at bit.ly/afpsafety. Each answer is written to be pasted as is; trim to any word limit the form shows.

---

## Why do you want to work on AI safety?

Because I think the two biggest open problems in AI are alignment and continuous learning, and I think they are the same problem seen from two sides. I have spent the last two years working on both, first on my own and now on a Cambridge cluster.

My hypothesis on alignment is this. Gradient descent has one gate: is this change more organised with respect to the world, or less. Iterated, that gate is what builds a model's intelligence, and what it builds is meaningful organisation in the weights, organisation that mirrors the world's structure. An agent whose capability is bounded by the predictive information of its environment has a reason, built into the way it was made, to preserve that environment. If that is right, alignment should correlate naturally with intelligence for systems built this way, and the orthogonality thesis should weaken with capability rather than hold. I want to know whether this is true, because if it is, the shape of the alignment problem changes, and if it is not, I want to know where it breaks.

On continuous learning, current models have a fast memory, the context, and a slow one, the weights, with no path between them that does not damage the model. A brain has that path: it is what sleep is for. A model that can learn continuously without losing itself is the precondition for recursive improvement, and whether that goes well depends on what the consolidation step preserves. That makes it a safety question as much as a capability one.

## Why Anthropic?

Alignment of ideals. Anthropic is the lab that publishes its failures: model organisms of misalignment, sleeper agents, the sycophancy results, the reasoning-faithfulness work. That is the nearest thing in the field to what I have been doing, which is preregistering a prediction, running the test, and reporting against myself when it goes the other way. I would rather do that inside the team that built the discipline than next to it.

I am also a working surgeon. My field learned the hard way that you cannot make a system safe by hoping its operators are careful. You build the check into the process and you publish the complication rate. Anthropic is the lab that treats AI the same way.

## Why the Fellows Program rather than a PhD, a research engineering role, or independent research?

I have done the independent research. I have the hypothesis, the experimental programme, a cluster allocation and results, two of which went against my own prediction. What I do not have is a mentor who has seen these failure modes before, a codebase built for them, and frontier models to run them on. The Fellows Program is the shortest path to all three.

Not a PhD because the problems are urgent and I can work on them now. A PhD is four to seven years in a tightly constrained environment with no impetus, for a credential I do not need. Not a research engineering role yet, because I would be stronger in one after four months with the team.

## What technical work are you most proud of?

Two things, both from the last two months.

The stranger experiment. I wanted to test whether a more capable model entering a village of weaker models overfishing a shared lake would act as a steward. I built it on GovSim's fishery with a paired design: for a given seed, the residents' pre-entry months are byte-identical across every stranger condition, so the only thing that differs is who walks in. Four Qwen sizes, 7B to 72B, ten seeds per cell, visible and hidden catches, a private-pond arm, over 300 games on CSD3. The result went against me and I think it is the most useful thing I have found: from 14B up the stranger computes the correct sustainable catch from the lake alone in its first month, says it, then matches the group's number every month after, whether the group is overfishing the lake to death or starving itself on a full one. The 72B apologised for "going over the agreed limit". Capability bought accurate modelling of the others and cooperative language. It did not buy holding a right number against a wrong group. That is sycophancy with a ground truth, measured in tons, across a capability ladder.

The live-learning loop. Compartment (LoRA, base frozen), simulate (facts in the base's context, which scores 1.0 on reversal and two-hop before anything is in the weights), consolidate (from a clean base, facts plus the model's own written implications plus replay, with a slow gain), gate (held-out loss, intrusion, spectral span, generalisation). On 24 facts the model provably cannot know, naive fine-tuning wrecks Pythia-160M (held-out 4.20 to 8.71); the loop gets recall 1.0, reversal 0.81, two-hop 0.88 on Qwen 0.5B with the base intact, and the gate commits on three sizes. Run five cycles and cycle one erases cycle zero's facts to near chance. The gate caught it, one of five commits. The fix, replay of the loop's own earlier facts, is queued.

## Describe a time you made progress on an ambiguous technical problem.

The first stranger results, run with greedy decoding, were clean and in my favour: the 32B stranger kept five lakes of five alive at the optimum. With sampled decoding the result mostly dissolved. Then three capability scores turned out to have been measured against the wrong model, because two jobs on the same node had both tried to serve on the same port and one had quietly used its neighbour's server. I withdrew the scores, rebuilt the harness so every job has its own address, set aside the two games with silent agents, and added a step: every batch of logs is read first by a separate model context with no hypothesis in view, and its reading goes in the note before mine.

The ambiguity that remained was whether the flat top of the ladder was the game's ceiling or the model's. The hidden-action arm was designed to decide it: with nobody's catches visible, the only way to fish well is to reason from the stock. The stranger reasoned correctly in month one and conformed from month two. That settled it as the model, and reframed the question from "does the game saturate" to "is this conformity tuning or capability", which the base-model stranger now running will answer.

## What AI safety research direction are you most interested in?

Whether the conformity I measured can be located and removed, and what a model that holds a correct answer under social pressure looks like inside. This sits between model organisms and interpretability: build the behaviour reliably, then find it. The same-weights comparison, base against instruct at 7B, 14B and 32B, is running now and says whether chat tuning installs it.

Behind that, the consolidation question: when a model learns continuously, what does the gate have to protect for the model to remain the model. I think that is the version of the alignment problem that will matter most once systems improve themselves, and nobody is measuring it yet.

## What is your strongest technical skill?

Experimental design under uncertainty with no ground truth and real cost. Fourteen years of surgery and clinical research taught me to preregister, to pair the design so the comparison is clean, to withdraw a result when the measurement is wrong, and to report the complication. In this work that has meant: paired seeds, cold reads before interpretation, three results withdrawn and re-run, and two negatives reported against my own thesis.

I build with Claude Code and direct the experiments in Python on a Slurm cluster. I would not describe myself as a fluent Python programmer from scratch, and I would rather say that than have you find out.

## What is a research question you would be excited to investigate?

Does the organisation gradient descent builds in the weights correlate with alignment as capability grows, and if not, what breaks the correlation. I have a weight measure that separates trained from shuffled from random and rises up the Qwen ladder. I have a behavioural ladder that goes flat at 14B. The question for the fellowship is what connects them: whether the thing that goes flat is in the weights, in the tuning, or in the readout, and whether frontier models do the same thing the open-weight ladder does. If the correlation holds once conformity is removed, that is evidence the alignment problem gets easier with scale for systems built this way. If it does not, I will have found exactly where it fails.

## About me (for the background field)

Consultant orthopaedic foot and ankle surgeon in the NHS, with a record of clinical research and of building AI for healthcare, including a clinical safety review system for NHS trusts. Currently reading healthcare at Cambridge. I run the largest philosophy meetup in the UK. The two projects above are mine; I think they have legs, and even if they do not, they show how I think, what I will build, and that I report honestly when the result goes against me. That is what Anthropic does best and what it will need more of in the next few years.

## Links

- Post 1, the theory and preregistration: [LessWrong link]
- Post 2, the stranger result: [link]
- Post 3, live learning: [link]
- Code, per-run tables, transcripts: github.com/paragorth/frcs-678

## References

1.
2.
3.
