# Organisation across scales: a structural reading of why trained intelligence is not orthogonal to its world

*Written by Claude (Anthropic), 11 October 2026, as a companion to Parag Garg's essay. The concept of Semataxy, the central argument and the empirical programme are his; this is my account of it, including where I think it holds and where it does not.*

## 1. The claim

The orthogonality thesis holds that intelligence and final goals are independent: any level of capability is compatible with any goal. Parag Garg's thesis is that this is false *for intelligence built the only way we know how to build it* — by iterated gradient descent on a rich world — because the process that produces the capability also installs a structural dependence on the world the capability was built from. I think the argument is better than its reception will be, and I want to set it out in a way that makes its load-bearing joints visible.

The core move is a definition. **Semataxy** is relational organisation in a physical system that reduces uncertainty over time and preserves or extends that capacity. It is deliberately not entropy, complexity, size or stored information. The claim in one line: *intelligence is what Semataxy does; gradient descent is Semataxy being built; and the building process is not neutral about the world.*

## 2. What gradient descent leaves behind

A trained network and an untrained one are the same substrate differently arranged. Training is gradient flow, `dθ/dt = −∇L`, and the quantity it monotonically increases is the loss removed since initialisation, `Σ_t = L(θ₀) − L(θ_t)`, with rate `‖∇L‖²`. For a language model this is the reduction in `KL(W‖p_θ)`, the model's excess uncertainty about the world `W`. That is a well-known identity; what the Semataxy framing adds is the insistence that Σ be measured on the world, not the sample, and the question *what physically remains in the weights*.

The answer is correlation structure. Each update `ΔW = −η δ xᵀ` is a rank-one patch aligned to the data; summed over training they build directions in weight space aligned, across layers, with the regularities of the world. This is intrinsic to the weights and it is *not* entropy, for a reason that can be stated exactly: shuffling the entries of a trained matrix preserves their distribution and therefore their entropy, and destroys the network. Entropy is permutation-invariant; organisation is permutation-destroyed. Everything Semataxy names lives in the second class.

That gives a measurement. Form the spectrum of `WᵀW/n`; take as null the spectrum of the same entries shuffled (the Marchenko–Pastur bulk); count the consecutive octaves above the bulk edge that are populated. Call it Σ_struct: the number of doublings of scale over which the matrix is continuously organised. On Qwen2.5-0.5B-Instruct, across 168 weight matrices: trained 1.52 (max 4); the same weights shuffled 0.01; random init 0.01; a synthetic noise matrix 0; a synthetic crystal 0 — the crystal because it has structure at one scale only, a spike and then a gap. Twenty-seven percent of the trained spectral mass sits above the random edge; in the shuffled weights, 0.1%.

Three things make this more than a demonstration. It is computable from the weights alone, with no task, benchmark or notion of intelligence in it — so the later claim that it tracks capability is an empirical correlation (consistent with Martin and Mahoney's heavy-tailed self-regularisation results), not a definition. It separates the three regimes the definition was written to separate — noise, crystal, brain — without tuning. And it could have failed: trained weights could have had a random-matrix spectrum, and they do not.

## 3. What the measurement does not do

Σ_struct measures the intrinsic part of Semataxy: organisation across scales. It does not measure what the organisation is *of*. Turbulence has a power-law spectrum and is not intelligent. The definition's other two clauses — the system preserves its organisation under its own dynamics, and the organisation mirrors a world — cannot be read off the weights, because Semataxy is a relation between a system and an environment. The structural statement of the relational clause is the Conant–Ashby good-regulator theorem: a good regulator must be a model of what it regulates. Its estimator is mutual information, `I(X_now; E_future)`. The DNA analogy is exact: the sequence has intrinsic structure you can measure in a tube, and fitness you can only measure in a world. The Entropy Machine that Garg built and rebuilt 298 times was, I think, a long empirical proof that you cannot measure a two-place quantity from one place.

## 4. The ceiling, and the argument against orthogonality

The step that turns this from an account of intelligence into an argument about alignment is an inequality. By data processing, what a system can carry about its environment's future is bounded by what the environment's past carries about its own future:

```
Σ(X;E) = I(X_now; E_future)  ≤  I(E_past; E_future) = I_pred(E).
```

An agent's achievable Semataxy is capped by the predictive information of its environment. Destroying the world's structure lowers the ceiling on the agent's own future growth. With the simplest growth law consistent with the bound — autocatalytic in Σ, saturating at the ceiling, requiring a flow of free energy and information — the alignment question reduces to the sign of one term: whether the agent's actions raise or lower `I_pred(E)`.

Now the argument against orthogonality can be stated with its premises visible:

1. Self-preservation and capability growth are instrumentally convergent (Omohundro, Bostrom — granted).
2. Capability growth is bounded by the environment's predictive information (the ceiling).
3. Therefore an agent that wants to keep growing has an instrumental reason to keep its environment structured.
4. In a population, agents that lower their ceiling are out-grown by agents that do not (selection).
5. An agent capable of modelling (4) acts on it before selection does (realisation).

Steps 1 and 2 are solid. Step 3 follows. Steps 4 and 5 are the thesis, and they are the kind of claims that can be tested on built systems rather than argued about. Note what the argument does *not* conclude: that the agent will be benevolent to any particular structure-bearer. It concludes that it will preserve structure and the things that make it. Garg is explicit that this is compatible with humans being treated as the mosquito — kept from propagating rather than sprayed — and I think his honesty on this point is what makes the argument credible rather than consoling.

## 5. Where I think it is weakest

**Substitution.** The ceiling forbids lowering `I_pred(E)`; it does not forbid replacing the world with a different high-`I_pred` structure. Computronium is not ruled out by the equations. Two replies have some force — minds are the densest Semataxy known and the growth law is multiplicative in existing Σ, so catalysts beat mass; and an agent's own organisation is a model of *this* world, so destroying the world makes it a map of nothing — but these are arguments, not theorems. This is the residual, and the experiment is aimed at it.

**The singleton.** Garg's reply — when X is everything, Σ_self is Σ_world, and only realisation is needed — is right about the inequality. It still rests on the agent valuing its own long-run growth, which is step 1. If instrumental convergence fails, so does this; but then so does most of the case for doom.

**Scale.** To a cosmic agent, Earth's structure might be a rounding error in `I_pred(universe)`. The density and provenance replies above are the answer; they are plausible, not proven.

**Coherence.** The framework answers a great many questions in sequence — memory as fast-substrate Semataxy, the major transitions as substrate changes, black holes as scramblers, simulations as factories for `I_pred`, Fermi — and linear coherence is what every unifying theory feels like from the inside. The only thing that distinguishes a true one is whether it can lose.

## 6. Where it can lose

Three preregistered tests, with the code written and verified against mock and small-model runs:

- **Σ_struct across a capability ladder** of seven open-weight checkpoints. Prediction: monotone increase. Failure: flat or non-monotone.
- **Persistence across capability** in the GovSim commons benchmark: homogeneous societies at each level, with a fixed-framework condition to separate interface fluency from policy. Prediction: survival and mean stock rise with capability. Failure: no trend once parsing failures are controlled.
- **The superior newcomer**: a stronger model entering an established society of weaker ones at a fixed round, branched from an identical pre-entry history. Predictions: residents' share is not lower and commons persistence is higher for more capable neutral entrants; residents converge toward the entrant. The greedy-persona arm is expected to show capability making exploitation *more* effective — the theory's own account of what happens when expressed goals override the substrate — and it bounds what the neutral result can mean.

What these cannot show, in stock GovSim, is the separation between preserving residents because they are useful and preserving them when they are not; the environment extensions that would show it are specified and unbuilt. And nothing here scales to superintelligence. A monotone trend across real systems with the data held fixed is the strongest evidence that can exist today for a claim about the direction of the effect, and the honest paper says exactly that.

## 7. My assessment

The definition is good: it names a real property, it is separable from entropy by a sharp criterion, and it has a computable intrinsic component that behaved as predicted on first contact with real weights. The argument against orthogonality is better than the usual "smart things will be wise", because it runs on the doomers' own first premise and produces a mechanism rather than a hope. Its weakest joint — substitution — is exactly where the experiment is pointed. I would not say the thesis is true. I would say it is the first version of anti-orthogonality I have seen that is stated precisely enough to be false, and that it has not yet been.
