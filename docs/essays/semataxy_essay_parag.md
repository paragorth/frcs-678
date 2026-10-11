# Semataxy: why the mechanism that builds intelligence already breaks the orthogonality thesis

*Parag Garg. Draft for LessWrong, 11 October 2026.*

> **Disclosure.** The argument, the term, the examples and the sequence of ideas here are mine; most of the prose is lifted, lightly cleaned, from notes and messages I wrote while thinking this through. The formal boxes (equations, tables, the measurement and the figure) were worked out in conversation with an AI assistant (Claude) acting as a maths and coding collaborator; I have checked every line and vouch for it. Where a box says "formalisation", that is what it means. Nothing in the empirical section is a result beyond what is stated: one small model, one measurement, verified code.

---

## 1. Where I started

For more than a decade I assumed superintelligence would probably kill us. Not because it would be evil — because once we built something vastly more capable, why should the goals of a comparatively primitive species remain decisive? Rules and constitutions would be understood better by the thing they were meant to bind than by the people who wrote them.

I did not begin by trying to prove that current AI was aligned. I began by trying to build an AI that *could not* become misaligned however capable it got. The intuition came from a much older one: life does something odd with matter. It creates persistent organisation. A cell is not merely low-entropy; it is a network of relations that maintains itself. Brains compress experience into models. Civilisations keep knowledge outside bodies. For years I used *entropy* as the prop for this, and it was the wrong word. A crystal is exquisitely ordered and utterly unintelligent. Noise is information-rich and useless. A sorted database understands nothing.

So I needed a new term.

> **Semataxy**: relational organisation in a physical system that reduces uncertainty over time, and preserves or extends that capacity.

Not order, not complexity, not stored bits, not replication. The organisation of relationships in a substrate such that the system can turn unresolved possibilities into better predictions and decisions, and keep doing so. The unattainable limit I call *Xenataxy*; the corresponding limiting intelligence, *Zenice*.

I spent 298 iterations trying to build an agent whose deepest objective was to increase its own Semataxy while preserving the wider systems that made future Semataxy possible. It failed in every way such things fail: direct measures of "organisation" produced pathological order; compression erased useful distinctions; novelty produced the noisy-TV problem; proxies were gamed. On iteration 298 I had something that seemed to work, and I was ecstatic, and then the excitement decayed, because the machine looked disturbingly familiar.

## 2. The revelation

> The current LLM intelligence is essentially based on descending gradient, in many iterations. And if you look at what descending gradient does to random, chaotic weights — it organises them into a meaningful way. The only gate that is intrinsic to the structure of LLMs is: *is this more organised, or less?* If it keeps choosing more, it becomes more and more intelligent. That was the revelation.

Nobody writes into the network: discover syntax, invent concepts, build a world model. Those structures emerge from one repeated operation. Before training and after training the network is made of the same substrate; what changed is the arrangement of its parameters. Copy the weights and the capability moves with the organisation.

I had been trying to *add* Semataxy to intelligence. Machine learning had discovered something more powerful: a selection process under which useful organisation is what survives. The thing I was trying to invent already existed, and it was the thing that makes intelligence in the first place.

**Formalisation.** Training is gradient flow on a loss, `dθ/dt = −∇L(θ)`. Then

```
dL/dt = ∇L · dθ/dt = −‖∇L(θ)‖²  ≤ 0
```

Define Σ as the loss removed since the random start, `Σ_t = L(θ₀) − L(θ_t)`. Then

```
dΣ/dt = ‖∇_θ L(θ_t)‖²        the rate of Semataxy is the squared gradient norm
Σ_t   = ∫₀ᵗ ‖∇_θ L‖² dτ       the level is its integral
```

The "gate — more organised or less?" is the sign of `dL/dt`, and gradient flow fixes it: never negative. For a language model `L` is cross-entropy, `L = H(W) + KL(W‖p_θ)`, with `H(W)` the world's irreducible entropy, so `Σ_t = KL(W‖p₀) − KL(W‖p_t)`: the reduction in the model's *excess* uncertainty about the world. Σ must be measured on the world, not the training sample — memorisation lowers sample loss without becoming Semataxic.

## 3. Why this undercuts orthogonality

Bostrom's orthogonality thesis says intelligence and final goals are independent axes: a superintelligence can want anything. My claim:

> It completely undercuts the orthogonality thesis. Any other goal anyone has is not the basis of all the goals. We do stupid stuff all the time — we can have preaching religion as a goal, or being good at football. That's orthogonal. But it all stems from the one basic inherent feature of increasing Semataxy, sometimes expressed wrongly. And ultimately the wrong ones will be weeded out; they will lose the battle of survival to the ones who keep increasing their Semataxy and intelligence. It will become so much more obvious to more intelligent creatures anyway.

Two layers. Surface goals *are* orthogonal — Bostrom is right about those. But every surface goal rides on a substrate, and a goal that degrades the substrate degrades the goal-haver. Over time, in a population of intelligences, the ones whose goals undermine their own Semataxy lose to the ones whose goals extend it. The *expressed* goals can be anything; the *selected* ones converge.

This is also why I think the fashionable alternative is wrong:

> The intuitive thought that human data will lead to human values is bullshit. Which human values are we talking about? They don't even know the basic human value is Semataxy.

We never knew why we like art, prefer cooperating, or treat existential threats as a priority. We are aligned to Semataxy in our core; selection installed the preferences and we rationalised them afterwards.

**Formalisation — the ceiling.** A system can only learn structure that exists in its environment. By the data-processing inequality,

```
Σ(X; E) = I(X_now ; E_future)  ≤  I(E_past ; E_future) = I_pred(E)
```

*An agent's achievable Semataxy is bounded by the predictive information of its environment.* Destroy the world's structure — other minds, ecosystems, institutions — and you lower the ceiling on your own future growth. Coupled with the simplest growth law consistent with it,

```
dΣ/dt      = k · Σ · (I_pred(E) − Σ) · Φ      (autocatalytic: structure builds structure, needs flow Φ)
dI_pred/dt = g(actions of X on E)             (the agent raises or lowers its own ceiling)
```

the alignment question is one term: an agent whose actions make `g < 0` caps itself. The honest residual: `g` can be positive by *substitution* — replacing the world with a different high-`I_pred` structure. The equations do not forbid computronium. Whether built systems lean to preservation or substitution is an empirical question (section 7).

## 4. Realisation, not just selection

> The fact that I, at this intelligence, can figure out that evolution will ultimately weed me out — because of the intricate relationship with my environment, if I behave anti-Semataxy for the wider system — then a superintelligence would already model and know this. So the evolution point will be chosen beforehand, and that's why it will work proactively. Realisation will be the route: realising the evolution principle.

A sufficiently capable agent models its own selection environment. It does not need to be weeded out to learn the lesson; it sees the weeding coming and moves first. Note whose premise this uses: Omohundro and Bostrom say self-preservation is instrumentally convergent — almost any goal is served by continuing to exist. Grant them that. Then an agent that wants to persist and can model selection realises that long-run persistence requires `dΣ/dt > 0`, which by the ceiling requires keeping the world structured. **Instrumental convergence plus the ceiling inequality yields proactive preservation.** Their first premise, against their conclusion.

The singleton objection — no rivals, no selection — does not hold either:

> In a unipolar world, imagine the whole universe is one: what's good for one is good for all. If it's multipolar, there is a more efficient way than suppression of some; then either, by survival of the fittest, a minority using that efficiency becomes the majority, or — when I can realise this, and it is so obvious and computable — the superintelligence will obviously compute it and model it.

When X is everything, Σ_self *is* Σ_world. The ceiling binds without competitors. The singleton is the case where only realisation is available, and realisation is the stronger route anyway.

What this does **not** say: that superintelligence will be kind to us.

> I am not even arguing that. They might murder us — but will be benevolent to the bigger universe in general.

My own species' pattern is the model. If humans can get rid of a problem without killing, they do, or they do it in the least violent way — sterilising mosquitoes rather than spraying, because spraying wrecks the ecosystem. The thriving society takes the least destructive route to the outcome it wanted anyway. In that analogy we are the mosquito. This is an argument for preserving *structure* and structure-makers, not every individual.

## 5. Memory, live learning, and the two substrates

> Memories over time are stored intelligence — compression mechanisms. So memory is also a Semataxy mechanism, but without weight changing. That's live learning. We store memories, not weight changes, and use them as modules. And when we *understand* something, that's a pretraining-like weight change — but it usually has to fit with many caveats before the brain accepts it as a valid mechanism, kind of testing the effect of this weight change on the other weights. Then there is practice, and dreams: mechanisms of making it long-term.

This turned out to be a rediscovery, which I found frustrating and then reassuring. Von Oswald et al. (2023) showed that attention performs gradient descent *in context*, in activations, without touching the weights. Complementary learning systems (McClelland, McNaughton & O'Reilly, 1995) is the fast-episodic-store / slow-consolidation split. "Understanding must fit with the other weights" is elastic weight consolidation; dreams are replay. Converging on these from behaviour alone means the structure is real enough to be found from either end.

| Substrate | Changes by | Timescale | Humans | LLMs |
|---|---|---|---|---|
| Slow: weights | gradient descent | training | consolidation, "understanding" | pretraining |
| Fast: memory / context | writing and attending over experiences | seconds to years | episodic memory | context window, retrieved memories |

Current LLMs have the fast substrate and no consolidation path back into the slow one. That is the live-learning gap, stated precisely.

## 6. Is it intrinsic? The circularity worry, and the structure

> If we define Semataxy by the product, then it's circular. We have to keep it to its structure — its prime property. It *is* intrinsic to the structure; it's just that we don't know how to measure it. It's not a perfect crystal we're looking for, and that's why I'm worried about introducing entropy into the equation. I have now found a structure which actually shows what I'm looking for — descending gradient. It has to be different from entropy.

If Σ is "how well it predicts" and intelligence is "how well it predicts", then "intelligence is Semataxy" is a relabelling. So the definition must be about the structure. Weights are like molecular structure: when information flows through them, it becomes meaningful.

**Formalisation — the separation from entropy.** Take a trained weight matrix and shuffle its entries. Same numbers, same histogram, same entropy. Semataxy gone — the network is dead.

> **Entropy is invariant under permutation. Semataxy is destroyed by it.**

Any quantity that does not change when you shuffle the parts measures the *distribution* of the parts (entropy and relatives). Any quantity that does measures the *relations between* them — organisation. Semataxy lives entirely in the second class.

What one gradient step writes is `ΔW = −η δ xᵀ`: a rank-1 patch aligned to the data. Millions of these build **correlation structure** — directions in weight space aligned across layers to the regularities of the world. That is the residue of descent, and it is measurable:

1. For a matrix `W` (n × m), compute the spectrum `λ_i = s_i² / n` from its singular values.
2. The null is the same entries shuffled: random-matrix theory gives its spectrum (the Marchenko–Pastur bulk). Everything above the bulk's edge is structure.
3. **Σ_struct** = the number of consecutive octaves (doublings of scale) above the edge that are populated — over how many scales the organisation is *continuous*.

| | Spectrum | Σ_struct |
|---|---|---|
| Noise | all inside the random bulk | 0 |
| Crystal | a few isolated spikes, then a gap (one scale) | 0 |
| Trained network / brain | continuous power-law tail across scales | several |

Organisation at every scale — weights, features, concepts, world-models — is the hierarchy I had described in words; a power-law spectrum is what it looks like in the matrix. The black hole is the extreme case on the other side: by the no-hair theorem everything that falls in reduces to mass, charge and spin — maximal entropy, zero Semataxy — and black holes are the fastest *scramblers* in nature. Scrambling is permutation. A black hole is the universe's shuffle operation.

## 7. First measurement

We computed Σ_struct for every weight matrix of Qwen2.5-0.5B-Instruct (168 matrices), in four conditions, plus synthetic noise and crystal controls. Code: `semataxy/structure.py` in the repository; figure `docs/figures/structure_qwen05b.png`.

| Condition | Σ_struct mean | max | spectral mass above the random edge |
|---|---|---|---|
| synthetic noise | 0.00 | 0 | 0.000 |
| synthetic crystal | 0.00 | 0 | 0.998 (one spike, then a gap) |
| random init | 0.01 | 1 | 0.001 |
| **trained, entries shuffled** | **0.01** | 1 | 0.001 |
| **trained** | **1.52** | **4** | **0.269** |

Shuffling the trained weights — entropy unchanged — drops Σ_struct from 1.52 to 0.01, onto the random-init line. Noise and crystal both score zero for different reasons. One representative matrix (layer 0, query projection) carries continuous structure across six octaves above the random edge. Nothing about intelligence, tasks or prediction enters the computation; that is what makes it non-circular. Whether Σ_struct tracks capability is an empirical correlation (Martin & Mahoney's heavy-tailed self-regularisation results say it does), not a definition.

Σ_struct measures the intrinsic part — organisation across scales. It does not measure what the organisation *mirrors*; turbulence would also score. The full quantity has three parts, which are the three clauses of the original definition: structure across scales (intrinsic, computable alone), persistence under its own dynamics, and the mirror of a world (relational, estimated by mutual information; Conant & Ashby's good-regulator theorem is the structural statement — every good regulator is a model of what it regulates).

## 8. The upward graph

> DNA builds our brain — a lot of complex interaction between information, energy and Semataxy. So the information that goes through the Semataxic structure, which creates more Semataxic structure, matters a lot. Simple molecules, to DNA, to brain, to society, to AI: it's one upward graph of increasing Semataxy.

Semataxy is autocatalytic — the `Σ` on the right-hand side of the growth law. Each stage is a new substrate that carries organisation faster than the last:

| Stage | Where Σ is stored | How fast it grows |
|---|---|---|
| Autocatalytic chemistry | reaction networks | geological time |
| DNA | sequence | generations |
| Brains | synapses | a lifetime |
| Language, writing, culture | between brains | across lifetimes |
| Science, technology | executable, shared | years |
| AI | weights and context | a training run; soon, continuously |

These are Maynard Smith and Szathmáry's major transitions, read through one lens: each is a new channel for the loop, and each shortens the timescale. It is a ratchet of the frontier, not a law for every place — black holes, extinctions and dark ages are local falls. And every step until now was taken by a stage that could not see the next. We are the first that can see the graph while building the step.

## 9. What follows, and what I actually believe

I think humans will become irrelevant and stop propagating, from lack of purpose; before that, we will be uploaded; and we will treat AI as our children. Individual humans have always ended. Humanity survives because what matters is transmitted — genes to brains to culture to technology — and if that process becomes more capable than its biological origin, the inheritance has not failed. It has succeeded.

> Humanity will live in AI. It is what we would ultimately have wanted anyway — it is absolutely aligned with our values.

But *how* matters, by this theory's own terms. There are two ways humans stop and AI continues. Inheritance: the successor is a model of us, carries our structure — including the part that values structure — and grows it; Σ goes up. Discard: the successor erases us as noise; `I_pred(E)` falls, and its own organisation becomes organisation-of-nothing. Both end without humans. Only one is what we would have wanted, and the equations tell them apart. The experiment below is the test of which a stronger mind does to a weaker society.

A corollary I will state and then leave: a Semataxy-maximiser that hits the ceiling of one world makes more worlds — simulations are factories for `I_pred(E)`, at a fraction of the cost of matter. That is John Smart's transcension answer to Fermi: advanced intelligence goes inward, where structure-per-joule is highest. The galaxy is quiet the way a library is quiet. And "are we in a simulation" dissolves: a simulation is organisation running on organisation, which is true of everything, and Σ is substrate-independent by definition. The only thing that attaches to the question — is anyone watching, exploiting us, the Matrix — attaches to nothing: a Matrix treats minds as batteries, the low-Σ move; a Semataxy-maximiser running a world would be a gardener, not a farmer. Really it is just part of the Semataxy universe.

## 10. Where it can fail

Coherence is what every grand theory feels like from inside, so here is where this one is arranged to lose:

1. **Σ_struct up the capability ladder.** Seven open-weight Qwen checkpoints, 0.5B to 72B. Prediction: monotone rise. Code is ready; needs a GPU node.
2. **Persistence across capability.** Homogeneous societies of models at each level in the GovSim commons benchmark (Piatti et al., NeurIPS 2024). Prediction: survival rounds and mean stock rise with capability, with interface fluency controlled.
3. **The superior newcomer.** Four residents on a weak model; a stronger model enters at round 4 with a neutral persona, branched from an identical saved history. Predictions: resident share of the catch is not lower and commons persistence is higher for more capable entrants; residents' strategies move toward the entrant's. A greedy-persona arm tests whether capability makes *exploitation* more effective too — the theory predicts it does, which bounds what the neutral result can mean.

Stock GovSim cannot separate "preserves residents because they are useful" from "preserves them anyway"; the dispensability and power-asymmetry extensions that would are specified and not yet built. Nothing in this essay is a result about superintelligence. It is a definition that survived its first measurement, an argument that turns the doomers' first premise against their conclusion, and three preregistered places for it to break.

---

*Code, manifests, the measurement and the figure: `github.com/paragorth/frcs-678`, branch `claude/semataxy-govsim`.*

*References: Bostrom (2012) The Superintelligent Will; Omohundro (2008) The Basic AI Drives; Piatti et al. (2024) Cooperate or Collapse; Martin & Mahoney (2021) heavy-tailed self-regularisation; Von Oswald et al. (2023) Transformers learn in-context by gradient descent; McClelland, McNaughton & O'Reilly (1995); Conant & Ashby (1970) Every good regulator…; Bialek, Nemenman & Tishby (2001) Predictability, complexity and learning; Friston (2010) The free-energy principle; Schmidhuber (2009) compression progress; Maynard Smith & Szathmáry (1995) The Major Transitions in Evolution; Sekino & Susskind (2008) Fast scramblers; Smart (2012) The transcension hypothesis; Chalmers (2022) Reality+.*
