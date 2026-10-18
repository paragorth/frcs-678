# PRE-REGISTRATION — Sparse Memory Finetuning on procedural skills

Qwen-2.5-0.5B-Instruct. Committed before any training run.

## Question

Lin et al. 2025 (arXiv 2510.15103) test SMF on **factual recall**. Our skills are
**procedural** (invented formal systems: beacon, crate, hollow, stamp). Whether a
procedure installs into a key-value memory layer is untested.

## Scoring rule — strict AND normalised, side by side (DECISIONS "Scoring")

`install` = held-out accuracy, no skill file in context, from a baseline of 0%.
- **normalised**: `re.sub(r"\s+"," ",s.strip()).upper()` equality
- **strict**: exact string equality
Both reported at every point. Raw generation stored for every scored trial.

## Install-matching rule — DEVIATION FROM THE BRIEF, and why

The brief's alternative reads "install within 20pp of `ft_hard` **at matched
drift**". **`DECISIONS.md` forbids this**: "Compare collateral at matched
INSTALL. Never compare install at fixed drift." That rule exists because
comparing at fixed drift manufactured a false result twice in this project
(Job 4 v1; Phase 3 arms 1-2).

**Resolution, applied and flagged:** each arm is titrated along its own ladder to
a matched INSTALL target, and collateral is compared there. Drift is reported as
an outcome, never used as the matching variable. The brief's 20pp criterion is
retained but applied to install-at-matched-install-target, i.e. it becomes a
check that both arms actually reached the target.

Note the arms are not commensurable in `||delta_theta||` anyway: SMF updates a
few thousand memory-value entries, `ft_hard` updates all 0.5B parameters. Matched
drift between them would be meaningless as well as forbidden.

## Collateral instruments (DECISIONS "Dead metrics")

`repr_collateral` is retired and is NOT used. Instruments:
1. **`dist_kl`** — KL from the untrained model's next-token distribution on 12
   held-out general-text prompts. Distributional.
2. **Cross-skill retention** — install on the OTHER three skills' held-out items,
   with their spec in context. Behavioural.
3. **Repeat fraction** on the general-text continuations, as a degeneracy guard
   (DECISIONS: the gate is repeat fraction, not entropy/KL), with generations
   read at the boundary rather than trusted to a threshold.

## Outcomes, fixed now

- **NULL**: install < 20% at every magnitude, OR collateral no better than
  `ft_hard` at matched install.
- **ALTERNATIVE**: install within 20pp of `ft_hard` at the matched-install
  target, AND collateral within 2pp of base on held-out probes.
- **AMBIGUOUS**: anything else. Resolved in neither direction.

## Stop conditions (from the brief)

1. Pipeline does not run clean within 30 min of effort. **-> PASSED, see report.**
2. `ft_hard` on the 0.5B installs < 20% on one skill/one seed: the model is too
   small to adjudicate. Stop.
3. Watcher stall.

## Also recorded

Rows updated per skill per step is logged (the count of memory value rows with
nonzero gradient), since sparsity is the mechanism under test.

---

## RUN LOG

**Step 1 — pipeline check: PASS** (CPU, well under the 30-min limit).
`memory_layers.HashingMemory` inserted at `model.layers.6.mlp`, `values.weight`
(256, 896); forward loss 3.7025; backward OK; **28 of 256 value rows carried
nonzero gradient** on one short prompt — the sparse read is real. One fix:
`load_qwen_with_memory` requires an explicit `dtype`.

**Step 3 — comparison arm: PASS, but only after extending the ladder.**

| lr | drift | install (norm) | install (strict) |
|---|---|---|---|
| 1e-5 | 0.0006 | 0.0% | 0.0% |
| 3e-5 | 0.0017 | 0.0% | 0.0% |
| 1e-4 | 0.0049 | 0.0% | 0.0% |
| **3e-4** | **0.597** | **33.3%** | **16.7%** |
| 1e-3 | 1.93 | 0.0% | 0.0% |
| 3e-3 | 9.61 | 0.0% | 0.0% |
| 1e-2 | 29.5 | 0.0% | 0.0% |

**The first ladder (<=1e-4) returned 0.0% everywhere and would have triggered
STOP_MODEL_TOO_SMALL. That would have been a false stop.** Drift was
0.0006-0.0049, roughly 20-100x below the 7B runs that installed (0.05-0.28), and
the generations were prose that ignored the `ANSWER:` format entirely: the arm
was never trained, not incapable. Two causes: an lr ladder carried over from the
7B without rescaling, and `clip_grad_norm_(..., 1.0)` where the 7B protocol clips
at 1e9. Both corrected; ladder extended.

This is the same error class as the disqualifying objection to the Phase 3 report
(concluding from an under-trained arm), caught here by checking drift against a
known-good reference before applying a stop condition.

**Usable window is narrow:** exactly one rung installs. 3e-4 works; 1e-3 emits
`<<VELLUM>>` for everything; 3e-3 and 1e-2 are degenerate
(`::::::::::::::::`). Any SMF comparison must be titrated inside that window.

**Strict/normalised gap is large:** 33.3% vs 16.7%, a 16.6pp gap against ~4pp on
the 7B. Half the 0.5B's "successes" are case- or spacing-variants.

**Step 4 — NOT RUN. Blocked, not skipped.** See report.

## Deviations forced by the $8 budget (recorded before results)

- **Retrofit reduced**: `--sample-size 10000 --num-train-epochs 1` against the
  paper's 50000 x 2 (`slurm/train_med_replacement_dense.sh`). 8x less retrofit
  compute. **This is an under-training risk on the SMF arm specifically**, and it
  is the same confound class that disqualified the Phase 3 report. If SMF
  under-installs, an under-trained retrofit is a live alternative explanation
  and must be reported as such, not read as a fact about procedural skills.
- Everything else (layers 6,12,18; n_keys 128; heads 4; knn 16; k_dim 256;
  replacement mode; lr 5e-4; seq 1024) matches the paper's canonical config.
- The repo requires module invocation (`python -m smf_rebuild.train_retrofit`);
  running the file as a script makes `smf_rebuild/datasets.py` shadow the
  HuggingFace `datasets` package and fails with a circular import.

---

## RESULT — corrected after cold read. The collateral claim DOES NOT SURVIVE.

An initial report claimed SMF installs procedural skills with **6.6x lower
`dist_kl`** at matched install, better in 7 of 8 cells. A cold read on the raw
numbers (fresh context, no framing) found the arms were install-matched in only
4 of 8 cells, that 57% of `ft_hard`'s mean KL came from one cell where it had
*failed* to install, and that **`ft_hard`'s optimum sat on the bottom rung of
its own ladder in 7 of 8 cells** -- never bracketed. It predicted one more rung
down would reverse the sign.

**The bracketing test was run. The prediction is refuted, but the claim still
falls.** `ft_hard`'s KL does not keep falling below 1e-4 -- it plateaus at
0.03-0.39 and is not monotone in lr. But with the ladder bracketed and install
matched at a **common target** rather than each arm's best rung:

| skill | seed | target | ft_hard KL | smf KL | ratio |
|---|---|---|---|---|---|
| beacon | 0 | 0.92 | 0.03262 | 0.02145 | 1.52 |
| beacon | 1 | 0.92 | 0.36272 | 0.02107 | 17.22 |
| crate | 0 | 1.00 | 0.25006 | 0.04141 | 6.04 |
| crate | 1 | 1.00 | 0.01648 | 0.03559 | **0.46** |
| hollow | 0 | 0.75 | 0.03787 | 0.05437 | **0.70** |
| hollow | 1 | 0.58 | 0.03503 | 0.05554 | **0.63** |
| stamp | 0 | 0.83 | 0.08661 | 0.05593 | 1.55 |
| stamp | 1 | 0.83 | 0.03186 | 0.05609 | **0.57** |

**SMF better in 4 of 8 cells -- a coin flip.** Only 1 of 4 skills wins on both
seeds. Geometric mean 1.52x, median 1.11x. The 6.6x becomes 2.50x even by the
generous arithmetic-mean-of-means, and that is again carried by one ft_hard seed
outlier (beacon s1, 17.22x).

**On behavioural collateral, SMF is WORSE.** With the untrained with-spec
cross-skill baseline finally measured (35.4%): ft_hard **-13.5pp**, smf
**-16.0pp**.

**Rows: 2.3%, and it is now a measurement.** Per-batch 768 was `top_t x layers`,
an arithmetic restatement of the cap, identical in all 40 runs. The cumulative
union over 80 steps is 1117 rows = **2.3%** (range 2.1-2.5%). Genuinely sparse,
genuinely varying.

### Verdict against the pre-registration: NULL on collateral, positive on install

- **Install**: both arms install all four skills from 0%. SMF is real.
- **Collateral**: "no better than `ft_hard` at matched install" -- **met**. 4/8
  cells, median ratio 1.11x, and worse on the behavioural instrument.

The ALTERNATIVE required collateral within 2pp of base on held-out probes. SMF
is 16.0pp below base on cross-skill. **Not met.**
