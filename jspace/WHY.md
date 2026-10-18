# WHY — the reasoning behind the constraints

`DECISIONS.md` says what to do. This says why. Read it when a rule seems
arbitrary or when you need to extend one to a case it does not cover.

## Why persistence, not acquisition
Weights are the function; activations are the output. Weights are fixed at
inference, held in memory, read every pass. Activations are recomputed per input
and freed at the end of the forward pass. **Nothing computed in a forward pass
survives it.** The model already does the inference — works out what the skill
implies, uses it, discards it. The gap is not that it cannot learn; there is no
**write path** from what it just computed to what persists.

Context is not memory. "Remembering" earlier in a conversation is re-reading text
still in the input, attended from scratch every token. The KV cache is a saved
intermediate, not a store. Anything worked out in the workspace and not written
into the text is unrecoverable one token later. (Also a real argument that
chain-of-thought is more than a prompting trick.) Long-context work — flash
attention, sparse/learned block attention, GQA, RoPE rescaling — makes a longer
window cheaper. None of it addresses persistence.

## Why consolidation does not already work
1. Fast and slow stores are the same substrate — no hippocampal analogue, so any
   write contaminates everything.
2. Replay needs to know what to replay; with no salience or prediction-error
   tagging the field replays random old data and it half-works.
3. Biological consolidation is reorganisation, not storage. Gradient descent
   moves weights until the new loss is low and treats everything else as
   collateral.

## Why skills-as-files is a workaround
It works, but the learning lives in text outside the weights: every skill must be
verbalisable and re-read through context on every use, paying attention cost
forever for what a brain would consolidate. The trade cuts both ways — a file is
discrete, addressable, reversible; compiled into weights it is diffuse and
entangled. **You can compile but you cannot decompile.** That asymmetry is what
the confinement mechanisms try to buy back.

## Why there is a penalty at all — orthogonality does not transfer
In activation space orthogonality is exact by construction: subtract the
component along one direction, add another, remainder untouched. **In weight
space there is no such operation.** One matrix computes all inputs; no update
touches one meaning and nothing else. Hence a soft penalty or hard projection
rather than a guarantee — the mechanism exists because the clean version is
unavailable. Persistence and collateral damage are two faces of the same fact.

## Why the target is the raw J-space vector
Not a token readout. The lens basis is model-specific; a target captured in one
model's coordinates means nothing in another's. Within-model only. This is why
teacher and student being the same model — with and without the skill in context
— is convenient rather than a compromise: the bases match by construction.

## Why the three mechanisms
- **Soft penalty**: `lambda_orth` in the loss. Trades off. Cheapest, weakest.
- **Gradient projection**: removes the orthogonal component from the update
  rather than trading against it. GEM/OGD lineage. Hard constraint.
- **Verify gate**: take the step, measure orthogonal drift against J-space
  progress, revert if collateral exceeds tolerance. Certify-before-commit at the
  weight level. **The rejection count is itself a result** — it says how often a
  useful update is inseparable from a damaging one.

**Ablating all three is how you find out whether confinement is doing work or
whether any constraint at all would have done.**

## Why matched drift, bisected — not matched steps
Matched steps compares arms that moved different distances, confounding any
difference with how far the weights travelled. Matching `||delta_theta||` by
bisection holds the budget fixed and asks what each arm bought with it. **Both
arms must be bisected with the same effort** — eight iterations for the baseline
and one hand-computed LR for the experimental arm is exactly what killed Job 4 v1.

## Why no LoRA
Low-rank adaptation is itself a forgetting mitigation. Running inside it measures
confinement on top of confinement and the result cannot be attributed.

## Why artificial skills in Phase 3, medical in Phase 5
Phase 3 asks whether the write path works at all, which needs unambiguous
scoring. Phase 5 asks about forgetting across domains, where near/far transfer
matters and content must be real. Different question, different corpus. Do not
mix them.

## Why the first skill cohort failed
Calibrated for unknowability but not executability. Three cipher skills scored 0%
in **both** conditions — a 7B cannot run a position-dependent cipher even with
the rules present. A skill that cannot be executed installs nothing, exactly as
uselessly as one the model already passes; the discard rule only caught the
latter. **Both ends need checking.**

## Why collateral is measured at three levels
Behavioural, representational (J-space drift on unrelated held-out concepts),
distributional (KL). They dissociate — an update can leave behaviour intact while
moving representations, or shift the output distribution without changing any
scored answer. A single number hides that.

## Why ignition is in this repo at all
Dehaene & Naccache called the workspace paper a landmark but flagged no ignition
signature. Independently, Parag's deployment model contains a prediction-error
threshold above which an item enters deployment mode. Same structural claim,
arrived at separately — so the lens is a cheap test of his own prediction inside
someone else's model. Advances the write path not at all.

Technical note if unparked: **action potentials do not get you ignition for
free.** A population of thresholded units with jittered thresholds sums to a
smooth sigmoid, so ignition is an additional claim about *synchrony*, not just
thresholds. This makes a language model the better instrument — no spikes, so a
step cannot be inherited from the units.

## Why the measurement rules exist
Every retraction is one error in different clothes: measuring something that had
to come out that way.

- `R^2 = 0.9997` — inject `alpha*v`, measure the projection onto `v`. That is
  `<h0,u> + alpha<v,u>`: linear by construction.
- "Degraded readout produces a threshold" — true by definition; steepness is set
  by trial variance. Signal detection theory, not a finding.
- "Text transitions while projection stays linear" — argmax is a threshold, and
  the text change sat exactly where the logit derivative peaked.
- Pooling — stepped units with jittered thresholds pool into a smooth unimodal
  distribution. Pooling destroys the signal being looked for.
- Un-gated alpha window — three findings computed over degenerate repetition.

**General form: a monotonic signal plus any threshold produces a step, and every
convenient measure — argmax, rank, binarised report — is itself a threshold. The
null and the alternative make the same prediction unless what distinguishes them
is specified in advance.** Hence the design cold read's question: *what result is
this design incapable of not producing.* It would have caught all five.

## Why the cold read is a separate context
Not a capability gap — the same model, differently positioned. Every significant
correction came from data already in hand that had not been interrogated. Hours
of debugging buys speed and buys a frame, and the frame is what makes "the null
is stronger now" feel true when two independent defects have been conflated.
**The specific thing an invested context does not do reliably is stop.** The pull
is toward the next experiment rather than "this instrument was never validated,
do that first." A context with no investment has no such pull. That is the whole
mechanism, and why the cold read must be fresh rather than a prompt to the same
session.
