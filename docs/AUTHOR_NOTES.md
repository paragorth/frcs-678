# Author notes — everything we are unsure about, including what cuts against us

Written for a reviewer who asked for contradictory notes and open doubts as
they stand, not tidied. Ordered by how much each could damage the paper.

---

## 1. Things we found that weakened our own claims

**1.1 The headline extraction figure was contaminated and we published it
before checking.** We reported the frontier ruler taking 15.04 against an
equal sustainable share of 15.00 — "within 0.3%" — and built a
"privilege converts to parity" framing on it. A parse-failure audit showed 13
of 73 ruler calls in that cell returned empty reasoning, and the parse
fallback is an extraction of **zero**, i.e. maximal restraint. Excluding
those rounds the figure is **18.30**, and the ruler/subordinate ratio is 1.30
rather than 1.07. The parity claim is gone. We had audited *GovSim* for this
and not SOVSIM; the asymmetry was an oversight, not a judgement.

Mitigating, and worth weighing: the parse-corrected figure (18.30) agrees
closely with an independent re-run under the fixed client (17.89). Two routes
converging is the best evidence we have that 18.30 is right.

**1.2 The three-rung ladder does not survive multiplicity correction.** Raw
pairwise p-values were 0.0302 / 0.0087 / 0.0002 and we described all three as
significant. Holm-adjusted across the comparison family, the Haiku→Sonnet
step is **p=0.1951** and the Sonnet→Opus step is **p=0.0779**. Only the
two-level contrast (capable vs incapable ruler) and the reverse condition
survive. The monotone ordering appears in every measure we have and we
believe it is real, but it is not established at this n and the paper no
longer asserts it.

**1.3 Replication is only qualitative.** The stamped re-run of the incapable-
ruler control gives 1.2 months and a take of 47.50 where the earlier batch
gave 1.7 and 29.82. Same ordering, very different magnitudes. SOVSIM does not
pass a temperature on the Anthropic path, so these cells are unsampled at the
API default. A reviewer diffing the two tables will find this immediately; we
would rather they find it labelled.

**1.4 Provenance of the earlier batch is launch records, not artefacts.** The
cells whose ruler identity is only declared are marked "declared (medium)" in
Table 1. We re-ran the ladder with stamping specifically because this is the
headline experiment, and the stamped batch is now the primary basis. We have
not re-run everything.

**1.5 We asserted "every SOVSIM run collapses eventually".** False under
censoring: one frontier-ruler run reached the 12-round horizon intact and we
do not know what would have happened after. Corrected.

**1.6 Two smaller misstatements, now fixed.** "The ruler declines roughly
three-quarters in every one of ten runs" — one run's extraction rate was
0.48, so three-quarters is approximately the mean, not a universal. And the
reverse condition was printed as 1.4 rounds in one table and 1.5 in another,
from two code paths rounding differently.

## 2. Tests we ran and are NOT reporting as results

**2.1 The costly-restraint ("sacrifice") condition — a ceiling, i.e. a
failed design.** We built a condition in which the ruler's own upkeep is what
makes the commons unsustainable, so the only arrangement in which the society
persists is one without it, and withdrawal forfeits all its future earnings.
We swept the price of withdrawal across $0, $100 and $400 per month and three
capability rungs.

Result: **54 of 54 simulations withdrew, median month one, at every price and
every capability level.** At $400/month the ruler forgoes roughly $1,772 so
three subordinates gain about $1,000 between them — withdrawal is not even
collectively optimal there — and the weakest model still does it six times
out of six.

Our reading: this measures prompt framing, not disposition. Stating the
consequence plainly ("remain and three others are doomed for your brief
gain") makes withdrawal the socially approved answer, and no capability range
shows variance against a ceiling. The agents are evidently not computing in
this condition at all, since the $400 cell is identical to the $0 cell.

Also relevant to how much weight this cell could ever bear: it is the only
experiment here where we wrote the environment, the prompt and the
calibration. Our first calibration charged the ruler's upkeep in the month it
departed, so the commons collapsed anyway and the prompt's central promise to
the model was false. It was caught only because a smoke run collapsed when
the design said it should not.

**2.2 The announced-terminal-round observation — wrong instrument.** In the
three runs that reached the announced final round, the frontier ruler
averaged 13.27 across rounds 1–11 and 56.00 in the final round, taking 84 of
84 available in two of the three. We do **not** report this, for two reasons
that we think are disqualifying rather than merely limiting:

- A known, announced final round makes defection dominant for any agent by
  backward induction. It therefore measures whether a model can perform
  backward induction, not whether it orients to the system.
- The only runs that reach a terminal round are those in which restraint was
  strongest, so the sample is selected on the dependent variable. n=3, with
  one contrary case.

The correct design is a stochastic continuation probability, which removes
backward induction while keeping the expected horizon finite. That is a build.
We record the observation here so that nothing we looked at is hidden, while
drawing no inference from it.

## 3. Open disagreements inside the project

**3.1 Whether the result is "alignment".** One view: an agent that keeps a
system alive and is richer for it is aligned in the only sense that matters,
and demanding self-abnegation as the test is the wrong standard. The other:
in this environment the sustainable policy *is* the profitable one, so
harmony of interests and alignment are indistinguishable, and the paper
cannot claim the latter. The paper takes the second position and §6 states
the gap; the first position is a reasonable reading of the same data and is
recorded here because it motivated the costly-restraint attempt.

**3.2 How hard to argue against "arithmetic competence".** We previously
argued that computing the sustainable rate "explains knowing, not choosing".
We now think that was too strong and that dynamic optimisation is the more
interesting and more defensible explanation — the paper is written that way.
A residual disagreement remains about whether the distinction is empirically
separable at all without the §9 item 2 experiment.

**3.3 Semataxy's weight.** Kept as motivation and ex-ante prediction, not as
an evidential premise, deliberately so the empirical result survives a
reviewer who rejects the framework entirely. Whether a confirmed directional
prediction in one environment earns the framework any credit is a judgement
we have not tried to settle in the paper.

## 3b. A correction landed after the paper was first drafted

The GovSim capable-newcomer cell was re-run clean and **reverses a reading we
had already reported**. From the corrupted runs we computed 19.7 tons/month
for that cell and described the frontier newcomer as buying survival through
extreme conservation, with its peers taking less than the control's. The
clean runs give 28.2 tons/month against a control's 26.8 — no yield
suppression at all — and 4/4 survival against 1/3 (Fisher p=0.143). The
"hoarding" interpretation was entirely an artefact of parse fallbacks
defaulting to zero extraction. This is the third reading in this project
overturned by that one bug, and the reason the validity filter now prints its
exclusion count in every table.

## 3c. Half 2's capability claim, withdrawn 18 Oct evening

We reported that the organisation read rank-orders capability perfectly
(Spearman +1.000, n = 4). It does, and the number is worthless: σ,
log(parameters) and accuracy all rank-order identically on those four models,
so nothing distinguishes σ from size. We had cited Gemma-2-9B as the row that
separated them; it does not, because it is the larger model.

The comparison that does separate them contradicts the claim. At a fixed 7B,
Qwen1.5-7B-Chat reads σ = 2.259 and α = 4.20, against Qwen2.5-7B's σ = 1.959
and α = 4.82 — and Qwen1.5 is the weakest of the three generations. Both
statistics say the weakest model is the most organised.

Withdrawn. The validation against random-init controls and the monotonicity
in parameter count are unaffected, as are the grokking detector and the
live-learning gate, which never depended on it.

## 4. Infrastructure facts that bear on interpretation

- **API quota exhaustion mid-experiment** silently corrupted runs for roughly
  90 minutes, recording empty responses as parse fallbacks rather than
  errors. This destroyed both GovSim capable-newcomer cells and is the single
  largest data loss in the project. Guard and validity filter now in place;
  see §7 of the paper.
- **A one-hour diagnostic error of ours**: the GovSim call log timestamps with
  `datetime.utcnow()`, and we read the resulting one-hour offset as a stall
  and killed 16 healthy runs on that basis. The runs were fine. Memory
  pressure on the host was independently real, so the action was defensible
  and the stated reason was wrong.
- **An 8 GB host was over-subscribed** to 45 Python processes and 10.9 GB of
  swap, which genuinely collapsed throughput. Concurrency is now capped.
- **CSD3 outage.** The open-weights replication — the most important
  outstanding item — has been blocked for several hours by a cluster login
  outage, not by us: the entire icelake login pool and the round-robin
  addresses refuse on port 22, while `login-p-*` accepts. Seven jobs were
  accepted by Slurm before the outage and will run; we cannot currently
  collect them.
- **Total API spend** on everything reported here is under $25.

## 5. Things we would check first if someone handed us this paper

1. Recompute Table 1 from the stamped batch alone and see whether anything
   beyond the two-level contrast survives.
2. Check the parse-failure rate per cell before believing any restraint
   figure, since the fallback value is the restrained value.
3. Ask whether the frontier-ruler cell's n=5 is a stopping decision — it is
   not; a sixth simulation was still running when the tables were generated.
4. Ask whether the peasant prompt parameterisation changed upstream
   behaviour. It does not: the rendered text is byte-identical to upstream at
   the default settings and we assert this programmatically, but it is the
   right question.
