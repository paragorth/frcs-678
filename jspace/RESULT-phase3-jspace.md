# RESULT — the corpus-averaged J-space state is not a viable write target

Scoped claim, 22 August 2026. This is the narrow version. The strong version is
stated separately at the bottom as a hypothesis, not a result.

> **Read this first. The three converging lines below share a model, a skill
> corpus and a lens. Three lines that share their confounds are not three
> independent lines.** They agree, and agreement is worth something, but a
> common-cause failure — this lens being the wrong instrument for this class of
> skill on this model — would produce exactly the same agreement. Nothing here
> is replicated across models, corpora or lenses.

## Claim

**Optimising a model's weights to reach the J-space activation state captured
from skill-in-context does not install the skill, and does so more reliably the
better the state is matched.**

## Evidence

Qwen2.5-7B-Instruct, invented skill `beacon`, baseline install 0%.
40 trained models across two runs (`phase3_writepath_7b.json`,
`phase3_writepath_7b_lorth0.json`), five configurations x four drift magnitudes
x two confinement settings.

1. **Five configurations optimise `L_target` (J-space state match). All install
   0.0% at every magnitude, strict and normalised.** `soft_penalty`,
   `grad_projection`, `verify_gate`, `random_subspace`, and `soft_penalty` with
   confinement disabled (`lambda_orth=0`). Drift hits target in every case and
   `L_target` falls monotonically from theta0's 0.4721 to 0.21-0.40. They train;
   they simply produce no behaviour.

2. **Removing confinement makes the state match better and the behaviour no
   better.** Unconfined: L_target 0.2144, cosine **0.907** to the captured state
   -- the best match in the experiment -- install **0.0%**. This was a
   pre-registered falsification test with the prediction stated before the run.

3. **Ordinary fine-tuning installs 91.7% at cosine 0.273**, i.e. while barely
   matching the captured state at all, at the same drift where unconfined
   state-matching installs nothing.

Therefore, precisely:

- **Minimising `L_target` is sufficient for 0% install** (5 configurations, 40
  models, no exceptions).
- **Matching the captured state is not necessary for install.**

This has a *direction* -- the objective selects against the outcome -- which is
why it is stronger than a null result.

## Converging lines

Three independent routes reach the same place:

| line | finding |
|---|---|
| enrichment | skill-in-context vs no-skill states differ in J-space only **1.23x** over chance -- the signal was never concentrated there |
| prefix geometry | the positive control fails identically for a known-good *fact* edit once the readout moves past the patch boundary -- state patching cannot carry a procedure across positions |
| this run | reaching the state through weights yields zero behaviour, and does so better the harder the state is matched |

## Scope -- what this does NOT establish

One model (7B), one skill (`beacon`), one lens, artificial procedural tasks with
invented notation. `beacon` is additionally the weakest of the four skills in the
Phase 3 corpus (+6.7% where others gave +23-25%). No claim about other lenses,
other layers, per-example rather than corpus-averaged states, or natural skills.

## The strong version, as a HYPOTHESIS

*Activation state is not the object that carries a procedure.* All three lines
above are consistent with it, but they share a model, a corpus and a lens, so
they are not independent in the way the table's layout suggests. Recorded as the
hypothesis this result suggests, not as something demonstrated.

## Two different lens quantities — do not conflate them

This result is about **cosine to a captured target state**. It shows that
proximity to a specific captured activation state does not predict capability
(log 32, DECISIONS "What the lens does not measure").

It says **nothing directly** about the other way the lens gets used:
**J-space readout drift on unrelated held-out concepts**, i.e. collateral. That
is a different quantity — distance from a model's *own* prior readouts, not
distance to an externally captured target — and no measurement here tests it.
Retiring collateral-by-drift on the strength of this result would be an
overreach of exactly the kind this project keeps making.

Collateral-by-drift **is** already retired, for a stronger and independent
reason: `repr_collateral / drift = 0.287 +- 0.022` across every arm and every
install level, so it is a restatement of `||delta_theta||` and carries no
information of its own. That argument stands alone and does not need this one.

## What survives

The question -- is there a write path from activation to weight? -- is
untouched. What is retired is one answer to *what to write*. The Jacobian lens
remains valid instrumentation: the 7B report onset is real, strictly scored,
with a 0.0% control (log 26, 30).
