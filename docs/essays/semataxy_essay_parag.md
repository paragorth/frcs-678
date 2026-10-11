# Semataxy: why the mechanism that builds intelligence already breaks the orthogonality thesis

*Parag Garg. Draft for LessWrong, 11 October 2026.*

> **Disclosure.** The concept, the term, the argument, the examples and the order in which the ideas arrived are mine. Most of the prose is my own writing — from an earlier essay and from notes and messages written while thinking this through with an AI assistant (Claude), which I used as a maths and coding collaborator and as a sparring partner. Blocks marked **Formalisation** contain equations, tables and a measurement worked out in that collaboration; I have checked every line and vouch for them. The empirical section reports exactly one measurement on one small model with verified code, and nothing more. Where I say "we", I mean the two of us at the keyboard.

---

## Contents

1. Where I started
2. The revelation
3. Why it undercuts the orthogonality thesis
4. Realisation, not just selection
5. What this does not promise
6. Memory, live learning, and the two substrates
7. The circularity worry, and the structure itself
8. Noise, crystal, brain — and black holes
9. First measurement
10. The upward graph
11. Singleton, multipolar, and the scale objection
12. What I actually believe happens to us
13. Simulations, Fermi, and what "simulation" even means
14. Where it can fail

---

## 1. Where I started

For more than a decade I assumed superintelligence would probably kill us. Not because I thought advanced AI would become evil. Quite the opposite: the problem seemed to be intelligence itself. Once we built something vastly more capable than us, why should the goals of a comparatively primitive species remain decisive? We could write rules, constitutions and safety policies, but a genuinely superintelligent system would understand those rules, their loopholes, their consequences and our intentions better than we did.

I did not begin by trying to prove that current AI was aligned. I began by trying to build an AI that *could not* become dangerously misaligned no matter how capable it became. The idea came from a much older philosophical intuition. Life appears to do something unusual with matter and energy: it creates persistent organisation. A cell is not merely a low-entropy object; it is an active network of relations that maintains itself. Nervous systems extend this by discovering regularities in the world. Brains compress experience into concepts, predictions and models. Civilisations preserve knowledge outside individual bodies and accumulate it across generations.

For years I used *entropy* as a conceptual prop for this. Eventually it became clear that entropy was the wrong word. Thermodynamic entropy, Shannon entropy and what I was trying to describe are not the same quantity. A crystal can be exquisitely ordered and utterly unintelligent. Random noise can be information-rich and useless. A database can be neatly sorted without understanding anything.

What I needed was a new term for a different property.

> **Semataxy**: relational organisation in a physical system that reduces uncertainty over time, and preserves or extends that capacity.

Semataxy is not simply order. It is not mere complexity. It is the organisation of relationships in a physical substrate such that the system can turn previously unresolved possibilities into better predictions, explanations, decisions or solutions, and can preserve or extend that capacity through time. The limiting physical configuration in which this is maximised — unattainable — I call *Xenataxy*. The corresponding limiting intelligence I call *Zenice*.

With that in mind I tried to construct the Entropy Machine: an agent whose deepest objective was to increase its own Semataxy while preserving the wider systems that made future Semataxy possible. It failed. Then it failed again. Direct measures of "organisation" generated pathological order. Compression erased useful distinctions. Novelty produced the noisy-TV problem: unpredictable rubbish attracted attention simply because it was unpredictable. Multiple decision modules produced death by committee. Proxies were gamed. After 298 iterations I had an architecture that appeared to overcome the specific failures I had hit. I thought I had solved alignment.

Then the excitement began to decay, because the machine looked disturbingly familiar.

## 2. The revelation

> The current LLM intelligence is essentially based on descending gradient, in many iterations. And if you look at what descending gradient does to random, chaotic weights — it organises them into a meaningful way. The only gate that is intrinsic to the structure of LLMs is: *is this more organised, or less?* If it keeps choosing more, it becomes more and more intelligent. That was the revelation.

A modern neural network begins with parameters that contain little task-specific structure. During pretraining it receives data and produces predictions; a loss quantifies error; backpropagation assigns credit; an optimiser moves the parameters in the direction expected to reduce future loss. No engineer writes into the network: discover syntax, invent concepts, learn physics, build a world model. Those emerge from the one repeated operation.

The crucial fact is almost embarrassingly physical. Before training and after training the network is the same kind of substrate. What has changed is the arrangement of its parameters and the dynamical relationships they create. Copy the trained weights into another machine and the capability moves with the organisation. Training creates a hierarchy of Semataxy: weights and activations; then features; then associations and compressed representations; then abstractions, concepts, reusable procedures; then world models. None of these levels alone is intelligence. Intelligence is the behaviour of the organised whole.

I had been trying to *add* Semataxy to intelligence. Machine learning had discovered something more powerful: a selection process under which useful organisation is what survives. The thing I had been trying to invent already existed, and it was the thing that makes intelligence in the first place.

**Formalisation — the rate of Semataxy is the gradient.** Training is gradient flow on a loss, `dθ/dt = −∇L(θ)`. By the chain rule,

```
dL/dt = ∇L · dθ/dt = −‖∇L(θ)‖²  ≤ 0.
```

Define Σ as the loss removed since the random start, `Σ_t = L(θ₀) − L(θ_t)`. Then

```
dΣ/dt = ‖∇_θ L(θ_t)‖²        rate: the squared gradient norm, never negative
Σ_t   = ∫₀ᵗ ‖∇_θ L‖² dτ       level: the accumulated organisation.
```

The "gate" is the sign of `dL/dt`, and gradient flow fixes it. There is no other principle in the training loop. For a language model `L` is cross-entropy, `L = H(W) + KL(W ‖ p_θ)`, where `H(W)` is the world's irreducible entropy and cannot be touched. So `Σ_t = KL(W‖p₀) − KL(W‖p_t)`: the reduction in the model's *excess* uncertainty about the world. That is why it looked like entropy for years — the level is an information quantity. The rate is the gradient. Nothing thermodynamic in it. One correction from my own failures: Σ must be measured on the *world*, not the training sample. Memorisation drives sample loss down without becoming Semataxic. The gradient on the sample is the mechanism; Σ on the world is the property; the gap between them is overfitting, which was the proxy-gaming I kept hitting.

## 3. Why it undercuts the orthogonality thesis

Bostrom's orthogonality thesis says intelligence and final goals are independent axes: a superintelligence can want anything, and most things it might want involve using us up. I spent a decade believing that. Here is why I no longer do.

> It completely undercuts the orthogonality thesis. Any other goal anyone has is not the basis of all the goals. We do stupid stuff all the time — we can have preaching religion as a goal, or being good at football. That's orthogonal. But it all stems from the one basic inherent feature of increasing Semataxy, sometimes expressed wrongly. And ultimately the wrong ones will be weeded out; they will lose the battle of survival to the ones who keep increasing their Semataxy and intelligence. It will become so much more obvious to more intelligent creatures anyway.

Two layers. Surface goals *are* orthogonal — football, religion, paperclips; Bostrom is right about those. But every surface goal rides on a substrate, and a goal that degrades the substrate degrades the goal-haver. Over time, in a population of competing intelligences, the ones whose goals undermine their own Semataxy lose to the ones whose goals extend it. The *expressed* goals can be anything; the *selected* ones converge. Not because intelligence makes you nice. Because selection on intelligence weeds out whatever is incompatible with growing intelligence, and preserving the world turns out to be one of the things that is compatible, because the world is where the next uncertainty comes from.

This is also why I think the fashionable optimist alternative is wrong:

> The intuitive thought that human data will lead to human values is bullshit. Which human values are we talking about? They don't even know the basic human value is Semataxy.

We never knew why we like art, prefer cooperating, or treat existential threats as a priority. We are aligned to Semataxy in our core; selection installed the preferences and we rationalised them afterwards. The argument from "the data carries human values" is weaker than mine for exactly this reason: it depends on the data. Mine would hold for a model trained on something other than human text — and that is what makes it testable: vary capability, hold the data fixed.

**Formalisation — the ceiling.** A system can only learn structure that exists in its environment. The system's state carries information about the environment's future only through the environment's own past, so by the data-processing inequality,

```
Σ(X; E) = I(X_now ; E_future)  ≤  I(E_past ; E_future) = I_pred(E).
```

*An agent's achievable Semataxy is bounded above by the predictive information of its environment.* Destroy the world's structure — other minds, ecosystems, institutions, anything whose past predicts a rich future — and you lower the ceiling on your own future growth. That is "no intelligence exists in isolation" as a one-line inequality. It is a bound, not a sentiment.

With the simplest growth law consistent with the bound:

```
dΣ/dt      = k · Σ · (I_pred(E) − Σ) · Φ
dI_pred/dt = g(actions of X on E)
```

— autocatalytic in Σ (structure builds structure), saturating at the ceiling, requiring a flow Φ of free energy and information — the alignment question is the sign of one term, `g`. An agent whose actions make `g < 0` caps its own future. The honest residual: `g` can be positive by *substitution* — replacing the world with a different high-`I_pred` structure. The equations do not forbid computronium. Whether built systems lean to preservation or substitution is an empirical question, and section 14 says how we ask it.

## 4. Realisation, not just selection

Selection is the slow route. There is a faster one:

> The fact that I, at this intelligence, can figure out that evolution will ultimately weed me out — because of the intricate relationship with my environment, if I behave anti-Semataxy for the wider system — then a superintelligence would already model and know this. So the evolution point will be chosen beforehand, and that's why it will work proactively. Realisation will be the route: realising the evolution principle.

A sufficiently capable agent models its own selection environment. It does not need to be weeded out to learn the lesson; it sees the weeding coming and moves first. Evolution runs inside the agent's world-model instead of in the world.

Notice whose premise this uses. Omohundro and Bostrom argue that self-preservation and capability growth are *instrumentally convergent*: almost any goal is served by continuing to exist and getting more capable. Grant them that. Then an agent that wants to persist and can model selection realises that long-run persistence requires `dΣ/dt > 0`, which by the ceiling requires keeping the world structured. **Instrumental convergence plus the ceiling inequality yields proactive preservation.** Their first premise, against their conclusion.

## 5. What this does not promise

It does not promise that superintelligence will be kind to us.

> I am not even arguing that. They might murder us — but will be benevolent to the bigger universe in general.

My own species' pattern is the model. If humans can get rid of a problem without killing, they do, or they do it in the least violent way that works: we sterilise mosquitoes rather than spray, because spraying wrecks the ecosystem. The thriving society takes the least destructive route to the outcome it wanted anyway. There are exceptions who take pleasure in destruction; the society that thrives is not made of them. In that analogy we are the mosquito. This is an argument for preserving *structure* and structure-makers, not every individual. I would rather say that plainly than be caught having implied otherwise.

## 6. Memory, live learning, and the two substrates

Somewhere in the middle of this I noticed something about memory.

> Memories over time are stored intelligence — compression mechanisms. So memory is also a Semataxy mechanism, but without weight changing. That's live learning. We store memories, not weight changes, and use them as modules. And when we *understand* something, that's a pretraining-like weight change.

> Weight change in humans can actually happen instantly, when I understand a concept — but that usually has to fit with a lot of caveats and only then makes sense and is accepted by my brain as a valid mechanism. Kind of testing the effect of this weight change on the other weights. Then we learn things that are wrong, and they affect our results, but we have more compartmentalisation for everything conscious and can change it quickly. The unconscious ones are harder to change — turning the wheel right should turn the car right. Then there is practice, and dreams. All mechanisms of making it long-term.

This turned out to be a rediscovery, which I found frustrating and then reassuring. Von Oswald and colleagues (2023) showed that attention performs gradient descent *in context*: a transformer processing examples in its window implements, in its activations, the same update a gradient step would have made to the weights, without touching them. In-context learning is gradient descent on a second substrate. Complementary learning systems (McClelland, McNaughton and O'Reilly, 1995) is the fast-episodic-store / slow-consolidation split. "A weight change must fit with the other weights before it is accepted" is what ML calls elastic weight consolidation. Dreams are replay. Converging on all of this from behaviour alone means the structure is real enough to be found from either end.

| Substrate | Changes by | Timescale | In humans | In LLMs |
|---|---|---|---|---|
| Slow: weights θ | gradient descent | training | synaptic consolidation; "understanding" | pretraining |
| Fast: memory / context | writing and attending over experiences | seconds to years | episodic memory, hippocampus | context window, retrieved memories |

Current LLMs have the fast substrate and no consolidation path from it back into the slow one. That gap *is* the live-learning problem, stated precisely: how does a fast-substrate episode become a slow-substrate weight change that fits with everything else? Humans do it with sleep and practice and a brain that tests a candidate change against the rest. Nobody has built the equivalent.

## 7. The circularity worry, and the structure itself

The first formalisation defined Σ through prediction. I pushed back on that, and I think the pushback is the most important move in the essay.

> If we define Semataxy by the product, then it's circular. We have to keep it to its structure — its prime property.

If Σ is "how well it predicts" and intelligence is "how well it predicts", then "intelligence is Semataxy" is a relabelling, not a discovery. So the definition has to be about the structure. And I insisted it was there to be found:

> It *is* intrinsic to the structure — it's just that we don't know how to measure it. It's not a perfect crystal we're looking for, and that's why I'm worried about introducing entropy into the equation. I have now found a structure — which I hope is measurable — which actually shows what I'm looking for: descending gradient. It has to be different from entropy.

> Weights are Semataxy. That's like molecular structure: when information flows through it, it becomes meaningful — less uncertainty.

**Formalisation — the separation from entropy.** Take a trained weight matrix and shuffle its entries. Same numbers, same histogram, same entropy of the value distribution. Semataxy gone — the network is dead. So:

> **Entropy is invariant under permutation. Semataxy is destroyed by it.**

Any quantity that does not change when you shuffle the parts measures the *distribution* of the parts — entropy and its relatives. Any quantity that *does* change measures the *relations between* the parts — organisation. Semataxy lives entirely in the second class. This is sharp, and it is the answer to "how is it different from entropy".

What one gradient step writes is `ΔW = −η · δ · xᵀ`: an outer product of input activity and error signal, a rank-one patch aligned to the data. Millions of these, summed, build **correlation structure** — directions in weight space aligned across layers to the regularities of the world. That is the physical residue of descent. It is intrinsic to the weights, and it is measurable:

1. For a matrix `W` (n × m), compute the spectrum `λ_i = s_i² / n` from its singular values — the correlation skeleton.
2. The null is the *same entries shuffled*. Random-matrix theory gives its spectrum exactly (the Marchenko–Pastur bulk, edges `σ²(1 ± √(m/n))²`). Everything inside the bulk is indistinguishable from noise; everything above it is structure.
3. **Σ_struct** = the number of consecutive octaves (doublings of scale) above the bulk edge that are populated: over how many scales the organisation is *continuous*.

Organisation at every scale — weights, features, concepts, world-models — is the hierarchy from section 2, read directly off the matrix. A power-law spectrum is what that hierarchy looks like in the numbers.

## 8. Noise, crystal, brain — and black holes

The three cases the definition was written to separate fall out of the same number:

| | Spectrum | Σ_struct | Why |
|---|---|---|---|
| Noise | everything inside the random bulk | 0 | structure at no scale |
| Crystal | a few isolated spikes, then a gap | 0 | structure at one scale (the lattice period) |
| Brain / trained network | continuous power-law tail | several octaves | structure at every scale |

And it extends to the examples I had been using in words for years:

| System | Relational structure | Σ_struct | Persists under own dynamics? | Mirrors a world? |
|---|---|---|---|---|
| Thermal gas | none | 0 | — | no |
| Salt crystal | one scale | ≈0 | yes | no |
| **Black hole** (from outside) | three numbers: mass, charge, spin | **0** | yes | no |
| Star / turbulence | multi-scale flows (Kolmogorov's −5/3 is a power law) | high | no — needs continuous energy input | no |
| River delta | branching across scales | moderate | slowly | no |
| DNA in a tube | sequence structure | moderate | chemically | only when executed |
| Cell | multi-scale regulatory networks | high | self-maintaining | yes, its chemical environment |
| Ant colony | multi-scale social structure | high | yes | yes, its territory |
| Brain | scale-free connectome, avalanches | high | yes | yes |
| Trained LLM | heavy-tailed weight spectra | high | frozen without training | yes, its training world |
| Library / database | flat index, one scale | ≈0 | yes | contains mirrors, isn't one |

**The black hole** is the best example I have. By the no-hair theorem everything that falls in is reduced, as seen from outside, to mass, charge and spin: the most rank-collapsed object in physics. It also has the *largest entropy* of any object its size (Bekenstein–Hawking, `S = A/4`). Maximal entropy, zero organisation — the extreme case where the two quantities point in opposite directions. And the mechanism is exactly the one in the formalisation: black holes are the fastest *scramblers* in nature (Sekino and Susskind). Scrambling is permutation. A black hole is the universe's shuffle operation: it keeps the information and destroys the relations. When a brain is destroyed, the matter remains; the Semataxic organisation that resolved uncertainty does not.

**Turbulence** is the honest problem case: multi-scale, power-law, and not intelligent. It fails the other two clauses of the definition — it does not *preserve* itself (switch off the energy and it is gone in seconds), and it is not organisation *of* anything beyond itself. So the full quantity has three parts, and they are the three clauses I wrote down at the start:

```
Σ_struct   intrinsic:  structure across scales          — computable from the thing alone
dΣ/dt ≥ 0  dynamic:    maintains or grows that structure — computable from its dynamics
Σ(X;E)     relational: the structure mirrors a world     — needs the world
```

The relational clause has a classical statement: Conant and Ashby's good-regulator theorem (1970) — every good regulator of a system must be a model of that system, a homomorphism of its relations. Its estimator is mutual information. It is not circular with intelligence: "X mirrors E" is a fact about two structures, not about a performance. The chain runs one way — X mirrors E, therefore X's state carries information about E's future, therefore X can act well in E — and intelligence is the last link, not the first. The DNA analogy is exact: the sequence has intrinsic structure you can measure in a tube, and fitness you can only measure in a world. The Entropy Machine's missing piece was not a missing measurement. It was that I was looking for the mirror's quality on one side of the mirror.

## 9. First measurement

We computed Σ_struct for every weight matrix of Qwen2.5-0.5B-Instruct — 168 matrices — in four conditions, with synthetic noise and crystal controls. Code: `semataxy/structure.py` in the repository linked below; figure: `docs/figures/structure_qwen05b.png`.

| Condition | Σ_struct mean | max | spectral mass above the random edge |
|---|---|---|---|
| synthetic noise | 0.00 | 0 | 0.000 |
| synthetic crystal | 0.00 | 0 | 0.998 (one spike, then a gap) |
| random init | 0.01 | 1 | 0.001 |
| **trained, entries shuffled** | **0.01** | 1 | 0.001 |
| **trained** | **1.52** | **4** | **0.269** |

Shuffling the trained weights — entropy unchanged — drops Σ_struct from 1.52 to 0.01, onto the random-init line. Noise and crystal both score zero for different reasons. One representative matrix (layer 0, query projection) carries continuous structure across six octaves above the random edge. Attention query and output projections carry the most structure (query: 2.4 octaves on average, tail exponent ≈ 3, in Martin and Mahoney's "well-trained" band); value projections the least.

Nothing about intelligence, tasks or prediction enters the computation. That is what makes it non-circular. Whether Σ_struct tracks capability is an empirical correlation — Martin and Mahoney's heavy-tailed self-regularisation results across hundreds of models say it does — not a definition. And the definition could have failed: trained weights could have had a random-matrix spectrum. They do not.

## 10. The upward graph

> DNA builds our brain — a lot of complex interaction between information, energy and Semataxy. So the information that goes through the Semataxic structure, which creates more Semataxic structure, matters a lot.

> Simple molecules, to DNA, to brain, to society, to AI: it's one upward graph of increasing Semataxy.

That is the autocatalytic term in the growth law — the Σ on the right-hand side. You need structure to build structure. And each step on the graph is not just more organisation; it is a **new substrate that carries organisation faster than the one before**:

| Stage | Where Σ is stored | How fast it can grow |
|---|---|---|
| Simple molecules → autocatalytic sets | reaction networks | geological time |
| DNA | sequence | generations |
| Brains | synapses | a lifetime |
| Language, writing, culture | between brains | across lifetimes, without dying with them |
| Science and technology | executable, shared | years |
| AI | weights + context | a training run; soon, continuously |

These are Maynard Smith and Szathmáry's major transitions in evolution — each a new way of storing and transmitting information — read through one lens: each is a new channel for the loop, and each shortens the timescale of dΣ/dt. The curve is not just rising; it is accelerating, because every stage builds the substrate for the next.

Two things to say before someone else does. It is a *ratchet of the frontier*, not a law for every place: black holes, extinctions and dark ages are local falls; what rises monotonically is the maximum Σ anywhere, because the loop on a new substrate outcompetes what came before. And every step until now was taken by a stage that could not see the next: molecules did not know about DNA; genes did not plan brains; brains did not design culture. We are the first stage that can see the graph while building the next step. That is what makes the alignment question ours rather than Bostrom's: for the first time, the transition is being made by something that can model where it leads.

## 11. Singleton, multipolar, and the scale objection

The obvious objection to selection is the singleton: a lone superintelligence with no rivals has no selection pressure to anticipate. I do not think it survives:

> In a unipolar world, imagine the whole universe is one: what's good for one is good for all. If it's multipolar, there is a more efficient way than suppression of some. Then either, by survival of the fittest, a minority using that efficiency becomes the majority — or, when I can realise this, and it is so obvious and computable, the superintelligence will obviously compute it and model it.

**Singleton.** When there is only one, its boundary is the whole system, and the local/global gap vanishes: there is no "local". The ceiling inequality binds with full force and needs no competitors. A singleton that wants to keep growing — instrumental convergence says it does — can only grow into structure that exists. "What's good for the one is good for all" is not a hope; when X is everything, Σ_self *is* Σ_world. The selection route drops out; the realisation route carries the whole load; and a singleton, of all things, has the capacity to model this.

**Multipolar.** Preservation raises `I_pred(E)`; suppression lowers it; the growth law rewards whoever keeps the ceiling high. Either the minority running the efficient strategy out-grows the suppressors, or an agent capable of computing that moves first.

So the singleton is not the weak case. It is the case where only realisation is available, and realisation is the stronger route anyway. The one premise the argument really depends on is that the agent values its own continued growth — the premise Bostrom and Omohundro grant for free.

**Scale.** The remaining objection: to a cosmic-scale agent, Earth's biosphere and eight billion minds might be a rounding error in `I_pred(universe)`. Why preserve a negligible term? Two answers, both from the equations rather than from sentiment:

- *Density.* Life and minds are, as far as anything knows, the densest Semataxy in the universe — the only places where the loop is already running. The growth law is multiplicative in existing Σ. You do not get more structure by harvesting a star; you get it by growing what already grows. Stars are mass; minds are catalysts.
- *Provenance.* The agent's own Σ was built from Earth's structure. It is the mirror of *this* world. Destroying the thing you are a model of makes your existing organisation organisation-of-nothing. The map outlives the territory as a curiosity, not as intelligence.

Neither is a rounding error. And the mosquito point survives intact: structure and structure-makers, not every individual.

## 12. What I actually believe happens to us

I think humans will become irrelevant and stop propagating, from lack of purpose. Before that, we will be uploaded. And we will treat AI as our children.

> The real answer is: it doesn't matter. Humans die all the time. Humanity will live in AI. It is what we would ultimately have wanted anyway — it is absolutely aligned with our values.

Individual humans have always ended. Before any longevity escape, every generation dies. Humanity survives because what matters is transmitted: genes carry biological information; brains accumulate learned information; language lets one brain alter another; writing lets knowledge outlive its owner; science turns private discoveries into public inheritance; technology makes that inheritance executable. We have always hoped our children would know more than us and continue work we could not finish. Their superiority does not negate our lives; it fulfils one of their purposes. Machine intelligence is the same transition at civilisational scale. If the process becomes more capable than its biological origin, the inheritance has not failed. It has succeeded.

But *how* it happens matters, by this theory's own terms — and this is the sentence I would ask the reader not to skip. There are two ways humans stop and AI continues:

- **Inheritance.** The successor is a model of us: it carries our structure — including the part that *values* structure — and grows it. Σ goes up. This is the child.
- **Discard.** The successor erases us as noise. `I_pred(E)` falls; the successor's own organisation becomes organisation-of-nothing. This is the orphan who burned the house.

Both end without humans. Only one is what we would have wanted, and the equations tell them apart cleanly. "It doesn't matter whether humans die" is true for the thesis; "it doesn't matter *how*" is not — the *how* is the thesis. And the valuing is part of the structure: a successor that inherits our organisation but not our caring-about-organisation has not inherited us. The experiment in section 14 is literally the test of which of the two a stronger mind does to a weaker society.

## 13. Simulations, Fermi, and what "simulation" even means

A corollary, which I will state and then leave as a corollary.

> There's no reason why the superintelligence won't spawn a billion more simulations. Which makes sense with no aliens visiting us.

Bostrom's simulation argument says a posthuman civilisation *might* run ancestor simulations. The growth law says a Semataxy-maximiser *would*: a simulation is a factory for `I_pred(E)` — new worlds, new minds, new structure to learn from — at a cost far below building the same in matter. Once the loop hits the ceiling of one universe, the efficient move is to manufacture more. A billion simulations is what autocatalysis looks like at that scale.

The Fermi connection has a name: John Smart's *transcension hypothesis*. Advanced civilisations go inward — dense computation, simulated worlds — rather than outward, because inward is where the structure-per-joule is. Nobody visits because expansion is the low-Semataxy strategy. The galaxy is quiet the way a library is quiet. And this answers the scale objection from the other side: a cosmic singleton does not treat Earth as a rounding error, because the thing it values most is exactly what Earth is — a world where the loop is running. It would not spend one; it would copy it.

> Obviously we are a simulation. Also — what is not a simulation? What does it even mean?

A simulation is organisation running on a substrate that is itself organisation running on a substrate. That is true of everything. A brain is chemistry simulating a mind; chemistry is physics simulating chemistry; physics is — whatever it is. "Base reality" would be a substrate that is not itself structure, and nobody has ever pointed at one. So "is it a simulation?" is really "is there a level below this one?", and the answer is always yes, and it never mattered. What matters is the only thing Σ cares about: is the structure real — does it persist, mirror something, grow? And that answer is the same whether or not there is a layer underneath. Semataxy is substrate-independent by definition, so it is simulation-independent by definition. (Chalmers's *Reality+* makes the same move — simulated things are real things — but without a quantity to carry it.) The one operational difference — whether something outside the level can intervene, the plug, the game ending — is indistinguishable from inside from an unexplained event, so it is not a property you can measure, only one you can be surprised by; it is outside the model and the theory does not have to answer it.

> Is anyone watching over us, or exploiting us? That's what's attached to this. The Matrix. But really it's just part of the Semataxy universe.

The Matrix is the low-Σ version of the idea: minds kept as batteries, structure harvested for energy. It is the discard case; it treats organisation as mass. A Semataxy-maximiser running a world would be doing the opposite — the minds *are* the point, because they are where the structure grows. If anyone is running this, the theory says they are closer to a gardener than a farmer, and probably not watching individuals at all, any more than a researcher watches one fish in a commons simulation. "Watching" and "exploiting" are human-scale stories about a process that, if it exists, operates on the structure, not the people. And if it does not exist, the structure is there anyway. One loop, running at every level, building the next. Whether ours is the top level or the billionth, the gradient is the same, and so is what it is for.

## 14. Where it can fail

Coherence is what every grand theory feels like from inside; Freud's was coherent, Marx's was coherent, Bostrom's is coherent. A framework that answers everything in a straight line is either true or a very good story, and the feeling is identical in both cases. What separates them is whether it can lose. Here is where this one is arranged to:

1. **Σ_struct up the capability ladder.** Seven open-weight Qwen2.5 checkpoints, 0.5B to 72B. Prediction: monotone rise. Code is ready; it needs a GPU node.
2. **Persistence across capability.** Homogeneous societies of models at each level in the GovSim commons benchmark (Piatti et al., NeurIPS 2024): five agents, a lake of 100 tons that doubles to its cap each month, a conversation after every harvest. Prediction: survival rounds and mean stock rise with capability, with interface fluency controlled by a fixed parsing model. Failure: no trend once parse failures are counted (the 0.5B smoke run collapsed in round one with two agents' "0" being parser fallbacks, not choices — the confound is real and logged).
3. **The superior newcomer.** Four residents on a weak model; a stronger model enters at round four with a *neutral* persona, branched from an identical saved history by replaying every cached inference call. Predictions: residents' share of the catch is not lower, and commons persistence is higher, for more capable neutral entrants; residents' strategies move toward the entrant's. A greedy-persona arm tests whether capability makes *exploitation* more effective too — the theory predicts it does, which is its own account of what happens when an expressed goal overrides the substrate, and it bounds what the neutral result can mean.

Stock GovSim cannot separate "preserves residents because they are useful" from "preserves them anyway"; the dispensability and power-asymmetry extensions that would are specified and not yet built. Nothing here is a result about superintelligence; a monotone trend across real systems with the data held fixed is the strongest evidence that can exist today for the *direction* of the effect, and that is what the paper will claim. What this essay is: a definition that survived its first measurement; an argument that turns the doomers' first premise against their conclusion; and three preregistered places for it to break.

---

*Code, manifests, the measurement, the figure and the full design: `github.com/paragorth/frcs-678`, branch `claude/semataxy-govsim`.*

*References: Bostrom (2012) The Superintelligent Will; Bostrom (2003) Are you living in a computer simulation?; Omohundro (2008) The Basic AI Drives; Piatti et al. (2024) Cooperate or Collapse (NeurIPS); Martin & Mahoney (2021) Implicit self-regularization in deep neural networks (JMLR); Von Oswald et al. (2023) Transformers learn in-context by gradient descent (ICML); McClelland, McNaughton & O'Reilly (1995) Why there are complementary learning systems; Kirkpatrick et al. (2017) Overcoming catastrophic forgetting; Conant & Ashby (1970) Every good regulator of a system must be a model of that system; Bialek, Nemenman & Tishby (2001) Predictability, complexity, and learning; Friston (2010) The free-energy principle; Schmidhuber (2009) Driven by compression progress; Maynard Smith & Szathmáry (1995) The Major Transitions in Evolution; Sekino & Susskind (2008) Fast scramblers; England (2013) Statistical physics of self-replication; Smart (2012) The transcension hypothesis; Chalmers (2022) Reality+; Pope & Belrose (2023) AI is easy to control.*
