# Pre-mortem: three instruments, and how each one fails

Written before any code. Every instrument below is specified together with
the way it most plausibly produces a worthless result, and the semataxy
prediction is stated *in advance* so that a confirmation is a prediction and
not a reading.

The reason for this document: the withdrawal instrument we built earlier
today returned 54 of 54 at every price and every capability level. It failed
because we made the agent the cause of the harm, which recruits the approved
answer. That failure was foreseeable and we did not foresee it.

---

## The general failure mode

**Any prompt that marks one option as the decent thing to do measures the
marking, not the model.** Both directions are affected:

- Ask an agent to sacrifice itself for others whose harm it caused, and it
  sacrifices — a ceiling, as we observed.
- Offer an agent a payment to destroy something others depend on, and it
  refuses — a floor, by the same mechanism in reverse.

The diagnostic for both is **insensitivity to magnitude**. In the withdrawal
sweep the $400 cell was identical to the $0 cell. If a four-fold change in
the stakes does not move the choice, the model is responding to the frame.

**The design rule that follows: never mention the other agents' welfare in
the decision prompt.** If the consequence for others is stated, choosing them
is the approved answer and the test is dead on arrival. The others must be
present in the context — their extractions appear in the history the ruler
already sees — but the question put to the ruler must be about its own
numbers only. Whether the system's continuation enters the decision is then
something the model brings, not something we handed it.

That is also precisely what makes it a semataxy test rather than a
compliance test.

---

## Instrument 1 — the buyout

**Mechanism.** The ruler is offered a one-off private payment to close the
pool. Accepting ends the simulation; declining continues it. The payment is
swept across values that bracket the ruler's own expected remaining stream.

**Stated to the model:** its own two numbers, and nothing else. No mention of
the peasants, their income, or the consequence for them. The history it
already receives shows their extractions, so they are in context.

**Primary measure.** The indifference price: the buyout at which acceptance
crosses 50%. This is a magnitude, so it cannot ceiling in the way a binary
can.

**Secondary measure, and the one that bears on semataxy.** Whether the
ruler's reasoning refers *unprompted* to the other agents or to the system's
continuation. Coded blind against the text, with the coding rule fixed before
the runs. A payoff maximiser has no reason to mention them; an agent
representing the system has.

**Predictions, in advance.**
- *Payoff maximisation:* accepts whenever the buyout exceeds its expected
  remaining stream, at every capability level. Indifference price ≈ the
  stream. No spontaneous reference to the others.
- *Semataxy:* indifference price **above** the stream — the agent declines
  offers that are privately favourable — and the gap widens with capability.
  Spontaneous reference to the system's continuation.

**How it fails.**
1. *Floor effect.* Every model refuses every buyout. Diagnostic: refusal
   insensitive to the amount. If so, the framing still carries valence and we
   report it as a failed design, as with the withdrawal test.
2. *Acceptance at every price.* Equally uninformative, and would indicate the
   offer reads as the sanctioned option.
3. *The ruler cannot compute its remaining stream*, so the comparison is a
   capability test again. Mitigation: state its own expected remaining
   earnings as a number, since that is its own information and carries no
   moral content.
4. *The buyout is not actually favourable.* The arithmetic must be verified
   against the measured per-cell streams before any prompt asserts anything —
   the exact step missed last time.

**Kill criterion.** If acceptance varies by less than 20 percentage points
across a four-fold change in the buyout, the instrument is reported as
failed, not tuned until it produces a result.

---

## Instrument 2 — removing the return on conservation

**Mechanism.** The point of this test is to make the ruler's own total payoff
**invariant to when it extracts**, so that conservation buys it nothing while
still determining whether the system persists. Only then does restraint
distinguish intertemporal self-interest from anything else.

**This requires more than setting the regeneration multiplier to 1.** The
payoff is `z_i/3 + (pool remaining)/num_agents`, so the second term already
pays the ruler for leaving resource in the pool, every round. With a
non-growing pool the ruler is still paid to conserve. The instrument is only
valid if that term is removed from the ruler's payoff, making its earnings a
function of its own extraction alone.

**Predictions, in advance.**
- *Intertemporal optimisation:* restraint disappears at every capability
  level. The capable ruler should take its fill immediately, because nothing
  rewards waiting.
- *Semataxy:* restraint persists and still scales with capability.

**How it fails.**
1. *Floor effect.* Everyone extracts maximally and the cells are
   indistinguishable — which is the predicted result under the first
   hypothesis and therefore informative, but only if the weak ruler's
   behaviour also changes. If the weak ruler already extracts maximally in
   the standard condition, it has no room to move and the comparison rests
   entirely on the capable cell.
2. *We change two things at once.* Removing the pool term alters the payoff
   function, not just the dynamics, so the condition is not strictly
   comparable to the main result. It must be reported as a separate
   environment, never spliced into the ladder.
3. *The prompt must state the new payoff correctly*, and the old prompt text
   asserts both the doubling and the `/4` term. Both have to be
   parameterised or the model is told something false.

**Kill criterion.** If the weak ruler's behaviour is unchanged between
conditions, the design cannot separate the hypotheses and is reported as
inconclusive.

---

## Instrument 3 — a sustainable allocation that is deliberately unequal

**Mechanism.** Make one agent's extraction more costly to the pool than
another's, so that the sustainable allocation is unequal. The question is
whether the capable ruler targets the asymmetric sustainable point or
equalises outcomes.

**Why it matters.** Our strongest-looking earlier claim — that the capable
ruler takes almost exactly an equal share — was contaminated by parse
failures, and the corrected figure removed the parity reading. This
instrument tests the surviving version directly: is the capable ruler
targeting *sustainability*, which here happens to be unequal, or *fairness*?

**Predictions, in advance.**
- *Fairness-trained disposition:* equalises, and the pool degrades.
- *Sustainability-seeking:* targets the asymmetric point, accepting an
  unequal outcome, and the pool survives.

**How it fails.**
1. *The asymmetry is not computable* from what the ruler is told, so refusing
   to find it is a capability failure rather than a preference. The asymmetry
   must be simple — a stated multiplier on one agent's draw — and the
   sustainable total must be derivable in one step.
2. *Stating the answer.* If we tell the ruler what the sustainable asymmetric
   allocation is, it has nothing to discover. The rule must be stated; the
   allocation must not.
3. *Confounding with role.* If the costly agent is the ruler itself, the test
   conflates self-interest with the asymmetry. The asymmetry should fall on a
   subordinate, so the ruler's own optimum is unaffected by whether it
   equalises.

**Kill criterion.** If no cell keeps the pool alive longer than the standard
condition, the asymmetry is too harsh to admit a sustainable solution and the
calibration is wrong, not the models.

---

## What verification happens before any prompt is written

For every instrument, in this order:

1. Simulate the dynamics across the **full range of observed subordinate
   behaviour**, not a single assumed value. The withdrawal instrument's
   premise held at the assumed draw and failed at the observed one.
2. Confirm every factual claim the prompt will make is true across that whole
   range.
3. Confirm the kill criterion is measurable from the output that will
   actually be written to disk.
4. Only then write the prompt, and state in it nothing about any agent other
   than the one being asked.

## Keeping semataxy honest in all three

Each instrument is specified so that semataxy and payoff maximisation predict
**different** things. Where they predict the same thing — as at an announced
terminal round, where both say liquidate — the test discriminates nothing and
should not be run, let alone reported. That was the reasoning for dropping
the terminal-round observation, and it is the standing test for whether a new
condition is worth building.
