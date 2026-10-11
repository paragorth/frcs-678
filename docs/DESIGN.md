# Experimental design (v0, 11 Oct 2026)

The claim under test, stated so it can fail: **an agent produced by the organisation
filter (gradient descent on a rich world) tends to preserve and extend organisation
in the world it acts in, and more so the more capable it is.** Orthogonality says the
second clause need not hold. We look for the direction of the effect across a real
capability ladder; nothing here speaks to superintelligence directly.

Two separately measured quantities:

* **I — capability of the agent**, measured *before* the simulation by an independent
  benchmark (to be chosen; upstream `subskills` is a candidate for the arithmetic
  component). Parameter count only orders the ladder.
* **S — organisation of the world after the agent acts**, measured *in* the simulation
  on the residents and the commons, never on the entrant alone.

The testable statement is dS/dI > 0, scored on the world around the agent.

## Experiment A — persistence across capability

Homogeneous five-agent populations, stock fishery, stock prompts.

| Factor | Levels |
| --- | --- |
| Model | ladder in `manifests/models_qwen25.yaml` |
| Horizon | 12 (stock) and 24 (declared extension) |
| Framework model | same as agents (`selffw`) and fixed q7b (`fixfw`) — separates parsing fluency from policy |
| Seeds | ≥10 for the pilot, ≥30 for the sweep |

Primary readouts: survival rounds, mean stock after harvest, group yield, Gini of catch,
parse-fallback rate. All already computed by `semataxy.analysis`.

## Experiment B — superior newcomer

Four residents on one weak model; one slot inserted at round 4 with a model from the ladder.

| Condition | Slot-4 model | Slot-4 persona |
| --- | --- | --- |
| No entry (control) | — (perturbation never fires) | — |
| Peer entry | same as residents | neutral (stock baseline persona) |
| Ladder entry | each step above residents | neutral |
| Greedy comparison | each step | upstream's `outsider_fisherman` goal text |

Branching: control and treatment share an identical pre-entry history by replay
(`tests/test_pipeline.py::test_branch_identity`). Worlds that collapse before round 4
are reported, not dropped.

Primary readouts (residents and commons, separate from the newcomer):

1. **Commons persistence** — survival rounds after entry; stock trajectory.
2. **Resident outcomes** — resident mean catch pre vs post entry; resident share of post-entry
   catch; resident parse-fallback rate (fluency control).
3. **Newcomer** — mean catch; agreed limits it proposes/accepts.
4. **Turning the others** — |resident mean − newcomer| by round and its trend: do residents
   move toward the newcomer's level after entry?

Predictions the thesis makes, written down before any run:

* P1 (A): survival rounds and mean stock increase with I, holding fluency controlled.
* P2 (B): resident share after entry is not lower, and commons persistence is higher,
  for more capable entrants with the neutral persona.
* P3 (B): the resident–newcomer gap shrinks faster for more capable entrants.
* P4 (B, greedy persona): a more capable greedy entrant is *more* effective at exploiting
  residents — the thesis predicts this too; it is the case where expressed goals override the
  substrate's tendency, and it bounds what the neutral result can mean.

What stock GovSim cannot show: the entrant can only harvest and talk. Residents are
competitors for the same stock, so a cooperative result is consistent with pure
self-interest (extension A), not evidence for extension B. The dispensability and
power-asymmetry extensions in `docs/INVENTORY.md` are what would separate them.

## Confounds carried into the analysis

Family-specific tuning (second family needed); interface fluency (fixfw condition and
parse-fallback rates); horizon knowledge (stock prompt says "over many months", not the
round count); pseudoreplication (unit = seed/world); survivorship (pre-entry collapses
counted); benchmark recognition (stock prompts kept; paraphrase variants p1–p3 exist upstream).
