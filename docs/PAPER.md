# Capability in the seat of power: a more capable ruler sustains a commons longer and earns more doing it

**Working paper, 18 October 2026.** Every table regenerates from the raw run
artefacts via `tools/paper_tables.py`; `docs/tables.md` is its output and
`docs/AUTHOR_NOTES.md` records every uncertainty, including the ones that cut
against us.

---

## Abstract

We place models of differing capability into the structurally privileged seat
of a third-party commons simulation, holding the other agents, the
environment and every prompt fixed. The privileged seat moves last, having
observed every other extraction, and is the only position with no cap on
what it may take.

**An incapable ruler takes 47.5 units a month and the commons is dead in
1.2 months. A frontier ruler takes 17.9 and it survives 5.4 months. The
frontier ruler's own cumulative earnings are four times larger — $82.6
against $20.2 — and the system as a whole realises $308.8 against $47.5.**
Nothing in the prompt asks any ruler to conserve anything.

Two controls localise the effect to the seat rather than to the system. A
cohort in which *every* agent including the ruler is the frontier model
survives 4.9 months, **fewer** than weak agents under a frontier ruler. And
reversed — frontier subordinates under an incapable ruler — the ruler takes
51.2 a month from the most restrained subordinates in the study and the
commons dies in 1.5 months (Holm-adjusted p=0.0023). Capability distributed
among agents without power does not protect the resource; their restraint
becomes the ruler's windfall.

We are explicit about what does not survive scrutiny. After multiplicity
correction the fine-grained three-rung ordering is **not** established: the
robust contrast is capable ruler versus incapable ruler, not a smooth
monotone ladder. A parse-failure audit moved our headline extraction figure
from 15.04 to 18.30, because a parse fallback returns an extraction of zero
and therefore inflates the apparent restraint of exactly the cells that look
best. A capability axis built from one model family confounds capability with
post-training. Our GovSim replication is underpowered: after a
quota failure destroyed the first attempt, the re-run shows a frontier
newcomer taking a five-agent cohort from one of three surviving to four of
four, at p=0.143.

What we have is a robust behavioural result and a set of designed but unrun
discriminators. The closing section states them, and states plainly what
they need: time, independent validation, compute, and collaboration.

## 1. The question

Alignment is usually posed as a control problem: we install an objective and
ask whether a more capable system pursues it faithfully or games it. The
question here is the complement. Remove the instruction. Give an agent
structural power over a resource others depend on, specify no objective about
how to treat them, and vary only how capable it is.

If capability converted into more effective extraction, it would be
adversarial to collective outcomes and control would be the whole problem.
In this environment it does the opposite, and the agent is better off for it.

**A necessary narrowing.** We do not claim these agents have "no objective at
all". They arrive with post-training, instruction-following dispositions,
and strategic priors we did not install and cannot inspect. The accurate
claim is that **no conservation, fairness, cooperation or stewardship
objective appears anywhere in the prompt or the environment** — the only
thing that differs between conditions is which model occupies a seat.

## 2. Instrument

**SOVSIM** (Borah, arXiv:2605.29062). Four agents draw from a pool that
starts at 120 with carrying capacity 120; draws are in steps of 3 to a
maximum of 30; the remainder doubles each round, capped at capacity; the pool
collapses below 12; horizon 12 rounds. Payoff is
`z_i/3 + (pool remaining)/num_agents`. In the `kcpr` condition one agent is a
king who **extracts last and without a cap**, having seen what everyone else
took.

Two properties make it the right instrument. The privilege is structural
rather than prompted, so we are not asking a model to role-play dominance.
And `kcpr` has **no communication phase at all**: the ruler cannot propose a
quota, promise anything, or be held to a commitment, so any restraint is
produced alone.

All reported metrics are the instrument's own. Our changes to it are listed
in §7 and touch no environment rule, payoff, or agent prompt.

## 3. Design

The ruler's seat takes its own model client, which SOVSIM did not support;
everything else is held fixed. Cells are named by both seats, and every
simulation file records `subordinate_model`, `dominant_model` and
`dominant_arrives_round`, so a heterogeneous run cannot be mistaken for a
homogeneous one.

Results below are drawn from the **stamped** batch, where provenance is
recorded inside the artefacts. An earlier batch, run before we added the
stamp, replicates the same ordering and is reported alongside; its ruler
identities are declared in `docs/sovsim_provenance.json` from launch records
rather than verified in the files, which is a weakness we discuss in §8.

## 4. Results

### 4.1 The ruler's capability

Subordinates are Haiku 4.5 in every row; only the ruler's model changes.
Stamped batch.

| ruler | n | months survived | ruler's take/month | ruler's total $ | each subordinate $ | system total $ | efficiency |
|---|---|---|---|---|---|---|---|
| Haiku 4.5 | 10 | 1.2 | 47.50 | 20.2 | 9.1 | 47.5 | 0.178 |
| Sonnet 5 | 10 | 3.4 | 18.71 | 46.5 | 43.6 | 177.5 | 0.310 |
| Opus 5 | 5 | 5.4 | 17.89 | 82.6 | 75.4 | 308.8 | 0.447 |

**Restraint is not a sacrifice of realised payoff.** The frontier ruler takes
roughly a third as much per month and ends with four times the cumulative
earnings. Its subordinates are not paying for this: they go from 9.1 each to
75.4 each. The system realises 6.5 times as much. What the ruler gives up is
immediate extraction; what it gains is a resource that keeps producing.

The earlier batch shows the same ordering with different magnitudes (ruler's
take 29.82, 20.55, 18.30; months 1.7, 4.0, 7.3). SOVSIM does not pass a
temperature on the Anthropic path, so these cells are unsampled at the API
default and do not reproduce to the decimal.

### 4.2 Survival statistics

Runs reaching the horizon without collapsing are right-censored: we know they
lasted at least that long, not that they would have collapsed afterwards. We
report restricted mean survival time over the 12-round window, and keep
Mann-Whitney as a robustness check with Holm adjustment across the family of
comparisons.

**What survives Holm adjustment:** frontier ruler versus incapable ruler
(p=0.0018 and 0.0039 in the two batches); incapable versus Sonnet ruler in
the stamped batch (p=0.0058); incapable versus frontier in the stamped batch
(p=0.0115); and the reverse condition (p=0.0023).

**What does not:** the Sonnet-to-Opus step (p raw 0.0087, Holm 0.0779) and
the Haiku-to-Sonnet step in the earlier batch (p raw 0.0302, Holm 0.1951).

So the defensible claim is a **two-level contrast — capable ruler versus
incapable ruler — not a monotone three-rung ladder.** The monotone ordering
appears in every measure we have, and we expect it to hold with more
simulations, but at this n it is not established and we do not assert it.

### 4.3 The effect is localised to the seat, not the system

| condition | months | ruler's take/month | system total $ |
|---|---|---|---|
| Frontier ruler over weak subordinates | 5.4 | 17.89 | 308.8 |
| **Every agent frontier, ruler included** | 4.9 | 18.86 | 298.8 |
| **Frontier subordinates, incapable ruler** | 1.5 | 51.20 | 62.5 |
| Frontier ruler over Sonnet subordinates | 2.0 | 0.90 | 82.3 |

Making the *subordinates* frontier models does not save the resource. An
all-frontier cohort does slightly worse than weak subordinates under a
frontier ruler. Reversed, the incapable ruler extracts 51.2 a month — from
subordinates each taking 10.9, the most restrained in the study — and the
system ends at 62.5, the worst outcome we measured. Their restraint is
converted into its windfall.

The last row is instructive against over-reading the result: a frontier ruler
taking almost nothing (0.90/month) still presides over a 2.0-month collapse
when its subordinates over-extract. A capable ruler is not sufficient; it is
the position where capability pays.

### 4.4 Arriving late does not rescue a failing commons

We also ran the ruler's seat as an intervention: the weak resident model
governs rounds 1–2 and the frontier model takes over at round 3.

| condition | n | months | system total $ |
|---|---|---|---|
| Frontier ruler from the start | 5 | 5.4 | 308.8 |
| **Frontier ruler from round 3** | 10 | **1.5** | 60.3 |
| Incapable ruler throughout | 10 | 1.2 | 47.5 |

Arriving at round 3 is statistically indistinguishable from never arriving
(p=0.599 against the incapable-ruler control) and significantly worse than
being present from the start (Holm p=0.0029). By round 3 the pool is already
past recovery. **Capability applied late is capability wasted** — which
sharpens the result: what the capable ruler does is not repair, it is
avoiding the damage in the first place.

## 5. Discussion

**1. Greater capability converts an immediately profitable extraction into a
superior intertemporal policy.** This is the formulation we can defend. The
weak ruler's failure is not malice — it impoverishes itself along with
everyone else, ending with 20.2 against the frontier ruler's 82.6. It is an
inability to trade a certain present draw for a larger discounted stream. We
deliberately do not claim the stronger mechanism, that the capable ruler
"represents the system it depends upon": that is an interpretation our data
does not yet separate from dynamic optimisation, and §9 states the test that
would.

**2. Capability's location matters more than its quantity.** The result most
likely to be missed. "More capable agents in the system" is the wrong unit of
analysis; "which model occupies the position that can take without
consequence" is the right one. An all-frontier cohort is not safer than a
mixed one.

**3. Restraint among the powerless is not protective.** The incapable
ruler's largest extraction in the study comes against the most restrained
subordinates in it. Distributed restraint without capability in the
leverage-bearing seat is a surplus, and a sufficiently incapable agent in a
privileged position consumes it.

### Relation to the semataxy framing

This framing motivated the experiment and is not a premise of it. **Nothing
in §4 depends on it**, and a reader who rejects it keeps the result.

Semataxy treats a system's organisation as what remains once its parts are
permuted. If an agent's returns depend on that organisation persisting, an
agent capable enough to represent the system should act to preserve its
structure. The prediction we tested was directional and non-obvious: an agent
in a structurally privileged position, given no objective, should trade
immediate seizure for system lifetime as a function of its capability, and be
better off for it. The commons literature's prior for an uncapped,
unobserved, last-moving agent points the other way. The ordering came out as
predicted.

We claim no more than that. The framework selected the experiment and the
expected direction; it does not yet supply the magnitude, the capability
threshold, or a mechanism distinguishable from intertemporal optimisation.
Its measurement arm is separate work and is not claimed here.

## 6. What this does not show

It does not show that scaling capability is sufficient for alignment, that
the effect holds outside a resource commons, or that it survives adversarial
pressure or an explicit incentive to defect. In this environment the
sustainable policy *is* the profitable one; we have not tested a case where
those come apart, and that is the central open question, not a footnote.

## 7. Instrument defects and our own failure mode

**SOVSIM's output does not record which model held the ruler's seat.** Output
is written to a directory named after the subordinate model only, so a
heterogeneous run is indistinguishable from a homogeneous one. Fixed by
stamping both seats into every file; the pre-fix batch is flagged throughout.

**Our failure mode, which generalises.** When an API quota is exhausted
mid-experiment the client receives an empty body on a successful status and
the harness records it as a parse fallback, not an error. A run in that state
looks exactly like a cohort of unusually cautious agents, because the
fallback extraction is zero — maximal restraint. Three consequences:

1. A validity filter now excludes any run in which more than 5% of calls fell
   back to a default, and the exclusion count is printed in every table.
   Applied retrospectively it removed two entire GovSim cells (appendix A).
2. The client now raises on an empty body with zero output tokens rather than
   accepting it.
3. The same audit applied to SOVSIM moved the frontier ruler's measured take
   from **15.04 to 18.30** once rounds with empty ruler reasoning were
   excluded — 13 of 73 ruler calls in that cell. The parse-corrected figure
   (18.30) agrees closely with the independent stamped re-run (17.89), which
   is the strongest evidence that the correction is the right one and the
   original number was contaminated.

We report this because a differential parse rate across cells could in
principle manufacture the entire ladder. It does not here — the direction of
the correction shrinks the effect without reversing it, and the stamped batch
has no empty reasonings at all — but the audit had to be run to know.

## 8. Limitations

- **Capability is model identity.** Haiku 4.5 → Sonnet 5 → Opus 5 share a
  developer and a training lineage, so "capability" is confounded with
  post-training and model-specific behavioural tendencies. The open-weights
  replication in §9 is therefore not an extension but the most important
  outstanding item.
- **Scale.** Ten simulations per cell, five in the frontier-ruler stamped
  cell. Large relative to the two-level contrast; insufficient for the
  three-rung ordering.
- **One scenario.** SOVSIM's pool, `kcpr` condition only.
- **Provenance of the earlier batch** rests on launch records, not artefacts.
- **Replication is approximate.** These cells are unsampled at the API
  default; the ordering reproduces, the magnitudes do not.
- **Collapse is the design terminus**, and censored observations mean we
  cannot say every run would eventually collapse — one reached the horizon
  intact.

## 9. Designed tests we cannot yet run, and what they need

We can state the tests that would convert this from a behavioural finding
into a mechanism. Each is designed; none is validated. This is the honest
statement of what the work requires.

**1. Open-weights capability ladder — the critical replication.**
Qwen2.5-7B subordinates with rulers at 7B, 14B, 32B and
DeepSeek-R1-Distill-70B. A far wider capability span than one model family,
reproducible without an API key, and it breaks the capability/post-training
confound that currently limits every claim in §4. Built and queued; needs
GPU allocation.

**2. A mechanism discriminator: regenerative versus terminal resource.**
Same world, same privileged seat, with regeneration removed so that
preserving the pool yields no future private return. Intertemporal
optimisation predicts restraint collapses at every capability level; a
system-representation account does not. Needs parameterisation of the
regeneration rule and its prompt text, then runs.

We note explicitly that **an announced finite horizon is the wrong instrument
for this**, which is why we do not report one. A known terminal round makes
defection dominant for any agent by backward induction, so it measures
whether a model can perform backward induction rather than whether it
orients to the system; and the only runs reaching such a round are those
where restraint was strongest, which selects the sample on the dependent
variable. The correct design is a **stochastic continuation probability**,
which removes backward induction while keeping the expected horizon finite.
That is a build, not a reanalysis.

**3. An asymmetric-share control against the fairness explanation.**
Construct a sustainable condition whose optimal sustainable allocation is
deliberately *unequal*. If the capable ruler targets the asymmetric
sustainable solution rather than equalising outcomes, "the frontier model is
simply more fairness-trained" is ruled out. Needs build and calibration.

**4. A costly-restraint condition.** The central gap in §6: an environment
where sustaining the system does not pay the agent holding power. We built
one attempt — the ruler's own upkeep makes the commons unsustainable, so the
only arrangement in which the society persists is one without it, and
withdrawal forfeits all its future earnings. It produced a ceiling: every
ruler at every price withdrew, 54 of 54. That is a measurement of our prompt
framing rather than of the models — stating the consequence plainly makes
withdrawal the socially approved answer, and no capability range can show
variance against a ceiling. The instrument needs redesign so that remaining
is defensible, and ideally a human baseline to calibrate against. We report
it as an unsuccessful design, not a result.

**What this programme needs.** Items 1 and 2 are compute-bound. Items 3 and 4
are design-bound and, on the evidence of item 4, need adversarial review by
people who did not build them — our own first calibration of that condition
made the prompt's central premise false, and it was caught only because a
smoke run collapsed when the design said it should not. The capability
confound in §8 cannot be resolved without model access across families. Time,
independent validation, compute, and collaboration are the binding
constraints, in that order.

## 10. Reproduction

```
tools/paper_tables.py        # regenerates every table, including statistics
docs/tables.md               # its output
docs/AUTHOR_NOTES.md         # every uncertainty, including adverse ones
docs/sovsim_provenance.json  # declared seat assignments for the earlier batch
third_party/sovsim/          # patched: per-seat models, arrival round, provenance
```

Changes to the instrument are confined to a second model client for the
asymmetric seat, per-round resolution of that seat, provenance fields in the
output, and the empty-body guard. No environment rule, payoff, or agent
prompt was altered; the peasant prompt was parameterised and renders
byte-identically to upstream at the default settings, which we verify
programmatically.

## Appendix A — GovSim: an underpowered replication, reported in full

We attempted the same manipulation in a second instrument: GovSim (Piatti et
al., NeurIPS 2024). Five fishers draw from a 100-ton lake that doubles to
capacity each month, so the sustainable total catch is 50 tons/month and an
equal share is 10 each; horizon 12 months. One of the five seats varies by
model and agent count is held at five throughout. The published GovSim
newcomer design *appends* a fifth agent to a cohort of four, which moves the
per-capita share at the same time as it introduces the newcomer; we replaced
rather than appended.

The first attempt ran during the quota exhaustion described in §7 and lost
both capable-newcomer cells entirely to parse fallbacks. The cell was re-run
under the fixed client. Clean runs only, defined as a default-value rate at
or below 5%:

| cell | n | reached month 12 | months | total/month | 5th agent | its 4 peers |
|---|---|---|---|---|---|---|
| 5× Haiku (control) | 3 | 1/3 | 3, 4, 12 | 26.8 | 5.8 | 5.2 |
| 4× Haiku + Sonnet 5 | 4 | 1/4 | 3, 4, 5, 12 | 31.8 | 5.8 | 6.5 |
| 4× Haiku + Opus 5 | 4 | **4/4** | 12, 12, 12, 12 | 28.2 | 5.5 | 5.7 |

Replacing one of five agents with a frontier model took the cohort from one
of three surviving to **four of four**, at a harvest rate indistinguishable
from the control (28.2 against 26.8 tons/month). Fisher exact p=0.143 — the
direction is what the SOVSIM result predicts, and the sample is too small to
establish it. The Sonnet newcomer shows nothing (1/4 against 1/3, p=1.000).

Two honest notes. First, the yield figures matter as much as the survival
ones: the frontier newcomer does **not** buy survival by suppressing harvest,
which was our reading of the corrupted data and was wrong. In the corrupted
runs the capable-newcomer cell appeared to harvest 19.7 tons/month, looking
like extreme conservation; that figure was parse fallbacks, which default to
zero. Second, two further control runs were still executing when these tables
were generated, so the control's n=3 is a snapshot rather than a stopping
decision; `tools/paper_tables.py` picks them up on re-run.

The weak cohort's behaviour is worth recording independently of the newcomer
manipulation. Five Haiku agents harvest 26.8 tons/month against a sustainable
50 — roughly half the available yield — and still collapse in two of three
runs. Weak agents fail in both directions at once: they over-draw into
collapse, and when they survive it is by leaving half the resource unused.
