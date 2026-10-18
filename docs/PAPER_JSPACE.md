# The write path that did not exist: reaching a skill's workspace state through the weights does not install the skill

**Parag Garg. Working paper, drafted 18 October 2026 from the result record of 22 August 2026 (`jspace/RESULT-phase3-jspace.md`), the design notes (`jspace/CONTEXT.md`, `jspace/WHY.md`) and the decision log (`jspace/DECISIONS.md`). Numbers are those recorded there; nothing has been re-run for this write-up.**

## Abstract

Fine-tuning acts on parameters and is blind to meaning, which is one reason catastrophic forgetting is unsolved. A model given a skill in its context already computes what the skill implies, uses it, and discards it at the end of the forward pass; what it lacks is a write path from that computation to something that persists. We tested the most direct candidate. Using the corpus-averaged Jacobian subspace from Anthropic's global-workspace paper (Transformer Circuits, July 2026) as a coordinate system for "the meaning the model is currently holding", we captured the workspace state of Qwen2.5-7B-Instruct with an invented procedural skill in context, then optimised the weights to reach that same state with the skill removed, under three confinement mechanisms that penalise or project away change in orthogonal directions, against ordinary fine-tuning at matched weight drift. The prediction was stated before the run. **Optimising the weights to reach the captured state does not install the skill, and the better the state is matched the same zero results.** Five configurations, forty trained models, install rate 0.0% at every drift magnitude. Removing confinement gave the best state match in the experiment, cosine 0.907 to the captured state, and still 0.0%. Ordinary fine-tuning installed 91.7% at cosine 0.273. The objective selected against the outcome, which makes this a directional negative rather than a null: minimising distance to the captured state is sufficient for zero install, and matching the state is not necessary for install. The conclusion we draw is narrow and we state the strong version only as a hypothesis: an activation state is not the object that carries a procedure.

## 1. The question

Weights are the function; activations are its output, recomputed per input and freed at the end of the pass. Nothing computed in a forward pass survives it. "Remembering" earlier in a conversation is re-reading text still in the window, attended from scratch every token; the KV cache is a saved intermediate, not a store. The field's workaround is skills as files, re-read into context on every use, paying attention cost forever for what a brain would consolidate. The trade cuts both ways: a file is discrete, addressable and reversible; compiled into weights it is diffuse and entangled. You can compile but you cannot decompile. The question here is whether there is a write path at all: can what the model has just computed from a skill in context be written into the weights so that it persists without the file?

## 2. Instrument

Anthropic's global-workspace paper supplies a coordinate system: a corpus-averaged Jacobian from a middle-layer activation to the vocabulary logits isolates a low-dimensional subspace that drives what the model reports. We used it as a target space, not because workspace theory is the point but because it is the only measurable, manipulable handle we had on the meaning the model is holding. The lens was validated on Qwen2.5-7B and 14B (4 of 4 on baseline, swapped and moved prompts; cross-check against the full Jacobian at 0.9999) and weakly on 1.5B (3 of 4 swapped, and 3 of 4 became the threshold after the fact, which we record as a weakness). Injection along a concept direction moved report rate from 1.7% to 86.7% over alpha 0 to 8 on the 7B inside the coherent window, with a random-direction control that never fired. The target is the raw J-space vector within one model; teacher and student are the same model with and without the skill in context, so the bases match by construction.

## 3. Design

Skill corpus: artificial procedural tasks with invented notation, chosen so that scoring is unambiguous (10 of 14 candidate skills kept, 5 with a context-versus-no-context gap of 95% or more). The skill reported here, `beacon`, has a baseline install of 0% without the file and is the weakest of the four Phase 3 skills (+6.7% with the file, where the others gave +23 to +25%).

Loss: `L = L_target + lambda_orth * L_orth + lambda_size * ||delta_theta||^2`, where `L_target` is distance to the captured workspace state and `L_orth` penalises drift in directions orthogonal to it. Three confinement mechanisms: a soft penalty; hard gradient projection in the GEM/OGD lineage; and a verify-before-commit gate that takes the step, measures orthogonal drift against progress, and reverts if collateral exceeds tolerance. Controls: a random-subspace target, and the soft penalty with confinement disabled (`lambda_orth = 0`), the latter a preregistered falsification test with the prediction stated before the run. Comparator: ordinary fine-tuning on the skill. Arms were compared at **matched weight drift**, `||delta_theta||` bisected to a common budget with the same effort in every arm, because an earlier version of this comparison (Job 4) had bisected one arm and hand-tuned the other, and was retracted for it. No LoRA, since low-rank adaptation is itself a forgetting mitigation and would confound the measurement. Four drift magnitudes.

## 4. Result

Qwen2.5-7B-Instruct, skill `beacon`, forty trained models across two runs (`phase3_writepath_7b.json`, `phase3_writepath_7b_lorth0.json`): five configurations, four drift magnitudes, two confinement settings.

| arm | `L_target` reached (from 0.4721) | cosine to captured state | install, strict | install, normalised |
|---|---|---|---|---|
| soft penalty | falls monotonically, 0.21–0.40 | | 0.0% at every magnitude | 0.0% |
| gradient projection | falls monotonically | | 0.0% | 0.0% |
| verify gate | falls monotonically | | 0.0% | 0.0% |
| random subspace | falls monotonically | | 0.0% | 0.0% |
| soft penalty, confinement off | **0.2144** | **0.907** | **0.0%** | 0.0% |
| ordinary fine-tuning | not optimised | 0.273 | **91.7%** | |

Every state-matching arm trains: drift hits its target and `L_target` falls. None produces the behaviour. Removing confinement makes the state match better, the best in the experiment, and the behaviour no better. Ordinary fine-tuning installs the skill while barely matching the captured state, at the same drift where unconfined state-matching installs nothing.

Precisely: minimising `L_target` is sufficient for 0% install (five configurations, forty models, no exceptions), and matching the captured state is not necessary for install. The objective selects against the outcome. That is why we call this a directional negative rather than a null result.

## 5. Converging lines, and why they are not independent

Three routes reached the same place. *Enrichment*: the skill-in-context and no-skill states differ in J-space by only 1.23 times chance, so the signal was never concentrated there. *Prefix geometry*: the positive control fails identically for a known-good factual edit once the readout moves past the patch boundary, so state patching cannot carry a procedure across positions. *This run*: reaching the state through the weights yields zero behaviour, and does so better the harder the state is matched. They agree. But they share a model, a skill corpus and a lens, and three lines that share their confounds are not three independent lines. A common-cause failure, this lens being the wrong instrument for this class of skill on this model, would produce exactly the same agreement.

## 6. Scope

One model (7B), one skill, one lens, artificial procedural tasks. No claim about other lenses, other layers, per-example rather than corpus-averaged states, or natural skills. This result is about cosine to a captured target state; it says nothing about the other use of the lens, readout drift on unrelated held-out concepts as a measure of collateral. That quantity was retired separately and independently: `repr_collateral / drift = 0.287 ± 0.022` across every arm and every install level, so it restates `||delta_theta||` and carries no information of its own.

## 7. What was retracted along the way

The decision log records three retractions that shaped the final design, and we report them because the method is the point. A headline of "100%, 60 of 60" (Job 2) had scored degenerate repetition. A finding that confinement loses to fine-tuning (Job 4) rested on asymmetric optimisation effort and an unmatched install, n = 1; it was marked not tested rather than falsified, and the matched-drift bisection above replaced it. Three ignition findings each consisted of a monotonic signal plus a threshold, which produces a step by construction; a preregistered criterion on which the null and the alternative make different predictions was written, and the ramp design then failed its pre-committed power condition. The recurring error was the same each time: measuring something that had to come out that way, or measuring outside the regime where the model is still coherent. Every sweep now reports its coherent alpha ceiling before any analysis runs.

## 8. What survives, and the hypothesis

The question, is there a write path from activation to weight, is untouched. What is retired is one answer to *what to write*: a state. The lens remains valid instrumentation; the 7B report onset is real, strictly scored, with a 0.0% control. The strong reading, that activation state is not the object that carries a procedure, is consistent with all three lines and is recorded as the hypothesis this result suggests, not as something demonstrated. The surviving candidate is to reproduce what the skill does rather than what it looks like, distillation of behaviour rather than matching of state, and that reframing is where the live-learning loop in this repository came from.

## Reproduction

`jspace/RESULT-phase3-jspace.md` (the result record), `jspace/PREREG-*.md` (preregistrations), `jspace/DECISIONS.md` (every decision and retraction with its reason), `jspace/scripts/` (analysis and sweep scripts), `jspace/out/*.json` (recorded results). Large artefacts (lens tensors, sweep logs, pulled model outputs) are omitted for size and are regenerated by the scripts.
