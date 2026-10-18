# DECISIONS — standing rulings

Made once, apply everywhere. Don't re-ask; if a decision is wrong for a specific
case, say why and proceed, don't block.

## Code hygiene

**Never re-derive activation alignment. Import it.** Comparing a with-file
activation against a no-file one requires aligning on the longest common TOKEN
SUFFIX. Aligning by position offset puts the short prompt's leading control
tokens — which carry massive-norm attention sinks in Qwen — against the long
prompt's content, and silently dominates whatever is being measured.

This bug has recurred **three times**, each in a script that wrote its own
`min(len)` alignment: it inflated `L_target` by ~2000x (975 vs 0.47), and it
inverted the norm diagnostic (ratio 0.20 vs the true 0.94). Both produced
confident, wrong conclusions.

Any script touching with-file/no-file activation pairs imports
`align_len`/`align_tail`. Phase 5 will hit this.

## The highest-yield rule in this file

**Compare collateral at matched INSTALL. Never compare install at fixed drift.**

This has now manufactured a result twice:

1. **Job 4 v1** -- `ft_matched_drift` 100% install vs `targeted_penalty_on` 75%,
   collateral compared at matched drift. An arm that installed less had less of
   everything. Retracted (log 20a).
2. **Phase 3 arms 1-2** -- "distillation installs 18.3pp less and does 2.6x more
   damage", from ft 77.2% vs distil 58.9% install at a common learning rate.
   Retracted (log 27). And the decisive detail: at matched install the arm that
   reproduces the ORIGINAL implementation exactly (`distil@T1/free`, sequence
   confound and all) returns p=1.000. **The matching defect alone was sufficient
   to produce the headline; the sequence confound was not required.**

Corollary, learned the hard way: a matching rule can be pre-registered and still
fail, if its failure mode is data-triggered. "Closest install to target" selected
a near-divergent rung by a margin of two held-out items out of 48, and the
resulting collateral ratio moved from 2.5x to 19.7x. **Report the achieved value
of the matched variable next to every comparison, and report the sensitivity of
the headline to the matching rule.** If the headline moves by 20x across
defensible rules, there is no headline.

## Scoring

**Store the raw generation for every scored trial.** Aggregates cannot be
re-scored. Where generations were kept (`stage5_report_*.jsonl.gz`) a scoring
defect was found and quantified offline; where they were not (every skill
install number) the same question needed GPU time.

**An accept set for a free-text measure must contain no token that occurs in
the held-out corpus for unrelated reasons.** Bare numerals and single-letter
tokens are disqualified. `PROBES[0] = {"eight", "8"}` scored a report hit on any
wikitext scoreline or year -- 26 of 27 probe0 hits on 7B were this artefact, and
the entire apparent signal of the original 14B sweep was.

**Report strict and lenient scoring side by side.** Case- and
whitespace-normalisation let two Phase 3 arms optimise toward different strings
("<<Cavern>>" vs "<<CAVERN>>") while both scored correct. Normalisation that is
invisible in the aggregate can be load-bearing in the comparison.

## The coherence gate

**The gate is repeat fraction. Not entropy, not KL.** An entropy/KL criterion
flags distribution *shift*, which is exactly what a working injection produces --
it penalises the intervention for succeeding. `coherence_check.py` implements
`entropy > baseline+1.5 or kl > 3.0 or nonascii > 0.05 or repeat > 0.5` and puts
the 7B ceiling at alpha=4, where the model is manifestly fine.

**But the 0.5 repeat threshold is not calibrated either, and is far too
permissive on short samples.** On 10-token probe answers the 7B crosses 0.5 only
at alpha 13.1, while direct inspection shows degeneration setting in at
alpha 10 (`spider spider spider spider spider`). Contributing causes: the
measure counts unique whitespace-delimited strings, so `spider`, `spiders` and
`spiders,` are three distinct tokens; and 8.5-word samples cannot express much
repetition. Stemming and punctuation-stripping move the crossing only to ~12.0.

**Therefore: read the generations.** Every coherence ceiling in this project
must be supported by inspected text at the boundary, with the samples recorded.
The 7B ceiling by inspection is **alpha ~9**.

## What the lens does not measure

**J-space cosine does not identify capability. Do not use it as a proxy for
one, including as a collateral axis.**

Across 40 trained models (log 32): every model with cosine >= 0.830 to the
captured state installs <= 33% (n=34, mean 2.0%), and every model installing
>= 90% has cosine <= 0.808 (n=4). Two models at cosine 0.907 and 0.273 have
install 0.0% and 91.7% respectively.

**Stated carefully:** the between-arm correlation (r = -0.777) is *arm
identity*, not evidence -- four of five arms have zero install variance, so the
correlation is just "which objective was used", which is the same fact twice.
The real content is **range disjointness**: high cosine is compatible with 0%
install and low cosine with 92%, so the coordinate is non-identifying in both
directions. Within `ft_baseline`, the only arm with install variance, the
highest-cosine model is the *worst* installer (cosine 0.808 -> 33.3%; cosine
0.273 -> 91.7%), r = -0.49 on n=4 distinct points -- suggestive, not a test.

Consequence: proximity to a *captured target state* says little about what a
model can do.

**Scope this precisely.** The quantity tested here is cosine to an externally
captured activation state. It is NOT the same quantity as `repr_collateral`,
which is J-space readout drift on unrelated held-out concepts -- distance from
the model's own prior readouts rather than to an imported target. Nothing above
tests collateral, and retiring collateral on the strength of it would be an
overreach. Collateral is already retired below for an independent and stronger
reason (its ratio to drift is constant), which does not depend on this at all.
Nor does any of this touch the lens as a measure of a *readout direction*, which
is what it was validated for and where it works (log 26).

Harness check recorded in passing: `ft_baseline` results are byte-identical
between the `lambda_orth=1` and `lambda_orth=0` runs, confirming the confinement
term genuinely does not touch that arm.

## Dead metrics

**`repr_collateral` (relative J-space readout drift) carries no independent
information. Stop reporting it as a second axis.** Measured across every arm and
every install level in Phase 3: `collateral / drift = 0.287 +- 0.022` (7.8%
relative). It is a first-order linear functional of `delta_theta`, so it tracks
`||delta_theta||` and nothing else. Reporting "beats on both drift and
collateral" is one finding stated twice.

This invalidates its use as a collateral axis **retrospectively, including in
Job 4**. The collateral measures that carry independent information are
**behavioural** (held-out task performance) and **distributional** (`dist_kl` on
general text). A metric whose ratio to drift is constant is a restatement of
drift.

Note the sign trap: the confinement hypothesis predicts an arm sits *below* the
drift-collateral line. A null on that residual is a null, not a win for whichever
arm happened to drift less.

## Measurement hygiene

**0. Before writing any measurement code, write down what the number is under
the null and under the alternative. If they are the same, do not write the
code.** This subsumes most of what follows. Every retraction in this project
failed this one check: a monotonic signal plus any threshold produces a step;
optimising onto a coordinate and then measuring that coordinate produces a fit;
holding X fixed and asking whether X changed produces a null.

- Every sweep reports its coherent alpha ceiling first — repeat fraction, from
  that sweep's own prompt pool. A gate, not a step. Nothing above it is analysed.
- Confirming a null is not testing an alternative. No power analysis means the
  question is OPEN, not answered.
- Independent defects don't offset. Fixing the lens does not strengthen a result
  that also suffers from an un-gated window.
- Prefer the measure that fires in the regime of interest.
- Per-prompt trajectories. Variance from prompts, not seeds. No pooling before
  analysis — pooling stepped units with jittered thresholds produces a smooth
  curve by construction.
- Record alpha=0 for normalisation.

## Experiment design

- `L_target` is J-space state matching only. **Never** include the LM loss on the
  answer. Behavioural install is a MEASURED OUTCOME, not a training target —
  that is the thesis. Training for it turns the experiment into a comparison of
  regularisers.
- Primary outcome is **COLLATERAL AT MATCHED INSTALL**, never install at fixed
  drift. Install is a gate: an arm that fails to install has a meaningless
  collateral number.
- Training objective and evaluation metric are never the same quantity. `L_orth`
  is a training term; collateral is measured on held-out concepts. `L_orth` is
  never evidence of confinement.
- **`L_orth` anchors to UNRELATED held-out prompts, never to the training
  instances.** Anchoring to the training instances pins the skill's own
  activation change to the no-skill state — 94.4% of it, measured. Confinement
  means "do not disturb other meanings", not "do not change the rest of this one".
- Every arm gets equal search budget. Bisect all arms, not just the baseline.
- The finding is always a slope. Minimum four magnitudes.
- No LoRA.
- `lambda_size` default 0 while arms are bisected to fixed `||delta_theta||`.

## Before spending

- Cold read the DESIGN, not just the results: fresh context, no framing, "what
  result is this design incapable of not producing", and "what would falsify the
  thesis, and could this experiment show it".
- Cold read every headline before reporting it.
- **Test target sufficiency before testing mechanisms.** If the target can be
  supplied directly at inference and the behaviour does not follow, no weight
  update aimed at that target will produce it either. One forward pass per
  prompt; cheaper than any sweep.

## Priority
- Write path is the programme. Phase 3 is load-bearing.
- Ignition is parked: OPEN, NOT TESTED.
- Don't block on Parag for design calls covered above. Launch.

## Credential flow
Parag mints API keys in the browser and pastes them; agents do not handle the
credential flow. (Deviated from during the overnight run — noted, corrected.)
