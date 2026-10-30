# Fine-tuning fragility: how much fine-tuning can a model's skills take before they break?

*Parag Garg. Working paper, October 2026. Every prediction was registered before its data existed; the
ones that failed are reported alongside the ones that held.*

## In brief

**The objective.** Every AI model is adapted after release by fine-tuning: a company teaches it its
products, a hospital its letters, a developer its tools. Fine-tuning can quietly break what the model
already knew. We set out to measure, for each of a model's skills, how much fine-tuning it can take before
it breaks, the way an engineer measures a material's breaking load before trusting it in a bridge.

**What we did.** For each skill we fine-tuned the model on a conflicting version of it, the same kind of
question with wrong answers, and watched the skill fall, against a harmless control. We ran it on 14
models from six families, on toy skills and on real ones (maths, following formatting instructions, tool
calls and refusals), and through a fine-tuning API on a model too large to hold, 276 billion parameters.

**What we found.**
- **Skills break fast.** One or two conflicting steps halve most skills. On the real tasks of eight chat
  models, every skill halved within five steps, and refusals within two.
- **Nobody would notice.** Perplexity, the health check practitioners rely on, had moved less than 5% when
  a skill was half gone, in 82 of 83 cases.
- **Ordinary fine-tuning does it too.** Fine-tuning on maths problems destroyed simple addition in three
  7 to 9 billion parameter models, and benign fine-tuning removed most of one chat model's refusals.
- **Broken is not gone.** Twenty correct steps on *other* examples bring most of a broken skill back: it is
  suppressed, not deleted.
- **The surprise: a new format can take over an old skill.** After learning to write its answers in a new
  format, a model applied that format where it did not belong, and that alone accounted for about half the
  damage to an unrelated skill. Nothing had contradicted the old skill; the shared format did it.

**What failed, and what came out of it.** Our main bet, that one fragility score would forecast the damage
from real fine-tunes, held for only one skill in four, and two rescue forecasts failed too. Taking those
failures apart produced what works: a **canary** that watches the first 30 steps of a user's own fine-tune
and predicts the damage it will do, which held in all four test sets.

**What it means.** If a model's safety behaviour is written by fine-tuning, it can be erased by
fine-tuning, in a few steps and without the usual checks noticing. Anyone releasing a model that others can
fine-tune, and anyone deploying a model that keeps learning, needs to know how fragile each behaviour is.
This evaluation gives them a number for it, and the canary gives them a warning in time. The same idea
is being built into Litmus, a clinical safety system for NHS trusts, as a monitor that notices when the AI
behind an approved clinical tool silently changes (a prototype, in testing).

**Limits.** The skills tested are a small panel; whether damage can be forecast before any training at all
is still open; and the fragility rating ranked unseen models only partly well. Details, and every
registered prediction with its verdict, follow.

---

# The full paper

## Abstract (provisional)

Fine-tuning is how models are adapted after release, and it can quietly break what they already
knew. We propose an evaluation of a model's robustness to fine-tuning: how much fine-tuning each of
its skills can take before it breaks. For each skill, a probe — synthetic for four toy skills, held-out
items for four real tasks — is scored while the model is fine-tuned on a conflicting version of the
skill (same prompts, wrong answers, other items), against a harmless control. The score is the weight
displacement at which the skill falls to half, with the number of steps, the damage to other skills,
whether the usual health check noticed, and how much returns. Across 14 models of six families, one or
two conflicting steps at a small learning rate (3e-6) halve format, arithmetic and fact skills (median
1.2 steps; 33 of 38 cells within two); in-context copying takes a median 4.9. On the real tasks of eight
chat models — GSM8K, formatting instructions, tool calls and refusals — every cell halves within five
steps (31 of 31; refusals in 0.6–1.7). The harmless control leaves every skill intact at the breaking
point (52 of 52 toy cells, 31 of 31 task cells), and perplexity has moved less than 5% when a skill is
half gone (82 of 83) — the check practitioners use does not see the break. The breaking point is
reproducible across training seeds and approximately independent of the learning rate (within 2–3×
over a tenfold range, while steps-to-break change tenfold), and a model's fragility on the cheap toy
panel ranks its fragility on tool calls and refusals (ρ 0.69 and 0.79, eight models; narrowly). LoRA,
at its usual tenfold higher rate, breaks the same skills after 4.8× more steps on the toy skills and
6.6× on the tasks. Broken skills are largely suppressed, not deleted: 20 correct steps on *other* items
bring back 77–103% of held-out items in 23 of 24 cells across seven models. The pre-registered test of
whether fragility predicts the loss under realistic fine-tuning (GSM8K, wikitext) fails: it holds for
copying (ρ = 0.94, 14 models) but not for the other three skills; GSM8K fine-tuning destroys few-shot
addition in three 7–9B models, about half of it by capturing its format, and benign fine-tuning removes most of
one chat model's refusals. Two registered measures of a dataset's exposure — a two-gradient rate and a
ten-step KL pilot — fail to predict that damage. For copying and capitals, a model's fragility can be
ranked without any fine-tuning, from two gradients at its own weights (ρ = 0.96 and 0.66, 13 models).
The eval also runs through a fine-tuning API, on a model too large to hold (Inkling-Small, 276B
parameters): at equal settings its skills take 2–5× more steps to break, but matched by how much of
the conflicting data it has learned they break like the open models'. Its weights are 3–6× larger, and
enlarging a small model's weights without changing its function slows its break 1.5–4.2× in the same
way: steps-to-break at a fixed rate mixes robustness with learning speed.

---

## 1. Why measure the fall

Forgetting is the obstacle between today's frozen models and models that keep learning. Remedies
exist, and more are coming (Paper 2 of this programme), but whether or not they work, a model owner
needs to know how fragile each skill is before fine-tuning a model or deploying one that learns —
in the way an engineer needs a material's breaking load before anyone knows how to make the material
stronger. pg's framing: *the fall matters on its own, whether or not we can find out how to recover.*

Three findings of the companion anatomy paper (Paper 1) make the eval possible. A skill breaks at a
model-specific weight displacement that the learning rate barely changes, so the breaking point can be
compared between models. Only conflicting updates destroy a skill at small displacement, so the eval
needs a conflict arm and a control arm. And general-text perplexity does not register the loss, so a
dedicated measurement is needed.

---

## 2. Related work

Full table: `docs/LITERATURE_PAPER15_2026-10-24.md` (†-marked entries to verify).

- **Before/after task batteries**, the common practice: re-run a fixed benchmark set after a
  fine-tune and compare. CapTrack (Thede et al. 2026) organises existing benchmarks into competence,
  default behaviour and execution reliability, and finds instruction tuning drifts most; PATH-Bench and
  EvoHarnessBench do it for agents and tool use; industry tools run it as regression tests. These say
  whether *one* fine-tune broke something, after the fact; on realistic fine-tunes our toy skills
  mostly read zero above ~1B (§4.7, §4.9), though the real tasks do not (§4.10), and a reading of zero
  cannot rank models by how close a skill is to breaking. Ours is
  the load test to their crash test; theirs covers more real tasks than our four (§4.10), and the two
  combine: a battery's tasks can each be given a conflict arm.
- **Continual-learning benchmarks for LLMs** (TRACE, CITB, CKL / TemporalWiki) report end-of-run
  forgetting after task streams; their numbers mix learning rate, steps and data. We report a
  breaking point per unit of weight change, at a fixed rate, per skill.
- **Forgetting laws** (Kalajdzievski 2024; Biderman et al. 2024) fit forgetting in steps, tokens, LoRA
  rank or fine-tune loss, mostly measured as perplexity or cross-entropy to the base model — the
  quantity our blind-spot result says misses skill loss.
- **Damage per unit perturbation**: random Gaussian perturbations across pretraining checkpoints
  (Springer et al. 2025); the safety basin along random directions (Peng et al. 2024); a LoRA-only
  spectral threshold (Xie & Alanwar 2026†). Ours uses the conflicting gradient's direction, which is
  what fine-tuning actually takes, and per-skill probes.
- **Steps-to-break for safety** (Qi et al. 2024, 2025; Rosati et al.'s "weak resistance"; TAR,
  TamperBench) measure how few examples undo refusal. Our steps-to-half generalises it to any skill
  with a synthetic probe; a safety skill is deliberately left out of this version.
- **What predicts forgetting**: across training methods, KL to the base model beats weight-change
  norms (RL's Razor, Shenfeld et al. 2026); lower rates forget less at matched target loss (Marek et
  al. 2026). We log both KL and three displacement norms.
- **Suppression vs deletion**: fine-tuning often adds a removable wrapper (Jain et al. 2024; Kotha et
  al. 2024†) — our revival column measures it.

---

## 3. The eval

**Skills** (teacher-forced exact match of the answer tokens; item skills scored as the share of items
right at step 0 that are still right):

| skill | probe (held out) | conflicting update (other items) |
|---|---|---|
| copy | random-token sequences repeated; next-token accuracy on the second copy | targets replaced by a fixed random permutation of the copied token |
| add | 2-digit addition, few-shot | answers a + b + 7 |
| capital | "The capital of X is Y.", few-shot, 62 countries | 63 other countries given another real capital |
| upper | "apple -> APPLE", few-shot | other words answered in lowercase |

**Arms.** Each skill's conflict, and a wikitext control, at AdamW 3e-6, batch 8, 50 steps, full
fine-tuning in float32; marks 0, 1, 2, 4, 8, 16, 32, 50. Realistic fine-tunes for H4: GSM8K supervised
fine-tuning and wikitext continued pretraining at 1e-5 for 120 steps. A revival arm: 20 steps of the
skill with correct answers on the training items only.

**Scores** per model × skill: **F**, the global displacement ‖θ − θ₀‖ / ‖θ₀‖ at which the probe falls
to half (also without the vocabulary matrices, and as the median over matrices); **steps** to the same
point; **selectivity**, the control arm at displacement F; **blind spot**, the change in perplexity at
F; **spillover**, the other skills at F; **recovered**, after the revival arm; the **KL** from the base
model at F. A skill is scored only if the model has it (start ≥ 0.5; capital ≥ 10 known items).

---

## 4. Results so far

### 4.1 One conflicting step halves a skill

Steps to half at 3e-6:

| | Pythia 160m | 410m | 1.4b | 2.8b | Qwen 0.5B | 1.5B | 3B | 7B | 7B base | Qwen3-8B | Llama 8B | Mistral 7B | Gemma 9B | OLMo 7B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| copy | 7.3 | 4.4 | 2.9 | 2.8 | 5.6 | 6.5 | 5.8 | 5.4 | 5.5 | 5.8 | 1.5 | 0.7 | 0.6 | 1.4 |
| upper | 2.1 | 1.4 | 1.3 | 1.1 | 1.2 | 1.4 | 1.4 | 0.5 | 0.5 | 1.3 | 1.2 | 0.5 | 1.1 | 0.5 |
| capital | 7.3 | 1.9 | 2.7 | 1.6 | 1.6 | 1.4 | 1.3 | 0.5 | 0.9 | 1.7 | 0.7 | 0.7 | 1.1 | 0.5 |
| add | — | — | — | — | 0.6 | 2.5 | 0.6 | 1.0 | 0.9 | 2.3 | 0.8 | 0.5 | 1.5 | 0.5 |

(Pythia cannot add two-digit numbers, so add is not scored there.) Format, fact and arithmetic skills
halve within one to three steps in every model (median 1.2; 33 of 38 cells within two), and within
about one step in most of the 7–9B models (Qwen3-8B and Gemma take 1.1–2.3). Copying, an in-context
skill, takes three to seven steps in Pythia and Qwen but under two in Llama 8B (1.5), Mistral 7B
(0.7), Gemma 9B (0.6) and OLMo 7B (1.4): the family matters more than the size.
The first step does most of the damage — Adam's first update moves every weight by the full rate in
the conflict's direction, and the conflict is most surprising when the model is most confident. A
learning-rate warm-up would spread that first kick over more steps without changing the
displacement at which the skill breaks.

### 4.2 The harmless control leaves skills intact, and perplexity does not see the break

Selectivity — the wikitext control at the conflict's breaking displacement — is 0.98–1.01 in all 52
cells (14 models; H3 holds). The blind spot — perplexity change at the breaking point — lies within ±5%
in 51 of 52 cells (H5 holds): between −3.3% and +2.7% everywhere except Qwen3-8B's copying, which
halves with perplexity 7.6% *lower*. In Qwen, perplexity usually improves while the skill is lost.
Non-conflicting updates do erode skills once they move the weights far enough: wikitext at 1e-5 for
120 steps costs Pythia-160m 35% of its copying at a displacement of 0.16%, against 0.009% for the
conflict to halve it.

### 4.3 Reliability (H1)

Across training seeds, F changes by 0.65–1.27× (10 of 11 cells within the pre-registered 1.5×; Qwen
1.5B's addition, 0.65, is 1.54× the other way).
Across learning rates it drifts: slower updates break a skill at a somewhat smaller displacement, by
up to 2–3× from 1e-5 to 1e-6, while the number of steps changes tenfold. Between 3e-6 and 1e-5 the
pre-registered 1.5× holds in 8 of 11 cells and fails in 3 (Qwen 0.5B add 2.59×, upper 1.52×; Qwen
1.5B copy 1.61×). F is therefore reported at a standard rate (3e-6), with the drift as a second
number. The order of models survives the drift (H1-R, registered): across 14 models scored at 1e-6 and
3e-6, Spearman's ρ between the two ratings is +0.94 for copying, +0.83 for addition (10 models), +0.84
for capitals and +0.75 for uppercasing (each p ≤ 0.002; +0.71 to +0.96 on the seven models first run at
1e-6), while F itself grows 1.3–1.7× from the slower rate to the faster. The rating is a stress score at
a stated rate whose ranking of models holds.

### 4.4 Spillover is rare, and not only in small models

At a skill's breaking point the other skills keep ≥ 0.96 of their start in 46 of 52 cells. The
exceptions are Pythia-160m, where a capital or copy conflict takes the format skill to 0.84–0.88, and
Qwen2.5-7B *base*, whose arithmetic falls with every other conflict: to 0.60 of its start when copying
halves, 0.93 and 0.91 when capitals or uppercasing halve. The instruct model of the same size keeps
add at 0.97–1.00. A base model's few-shot arithmetic is the most collateral-prone skill seen so far.

### 4.5 Suppressed, not deleted

Seven models (Pythia 160m–2.8b, Qwen 0.5B–3B), 24 model × skill cells. Each conflict runs 50 steps at
3e-6 and leaves its skill at 0.00–0.07; then 20 steps of correct answers on the *training* items only
(copying: correct copies of new sequences). Share of the held-out probe items back, as a fraction of
what was lost:

| model | copy | add | capital | upper |
|---|---|---|---|---|
| Pythia-160m | 55% | — | 91% | 81% |
| Pythia-410m | 99% | — | 77% | 90% |
| Pythia-1.4b | 102% | — | 91% | 95% |
| Pythia-2.8b | 103% | — | 97% | 98% |
| Qwen2.5-0.5B-Instruct | 99% | 89% | 89% | 92% |
| Qwen2.5-1.5B-Instruct | 102% | 98% | 98% | 95% |
| Qwen2.5-3B-Instruct | 102% | 99% | 100% | 99% |

Capitals of countries never trained on come back from the correct capitals of *other* countries: the
knowledge survived, access to it broke. The way back is short. Half of the skill returns within 4–20
steps (usually 8–12) and 90% within 8–20 steps in 19 of 24 cells. It moves the weights a median 0.57×
as far as the conflict did (0.36–1.14×). Recovery is fuller in larger models: 77–103% from 410m up,
99–102% in Qwen 3B, and only Pythia-160m's copying stays below three quarters. This is the opening
for Paper 2: what a conflict breaks in these models is mostly access, cheap to restore if one knows
what was lost.

### 4.6 A shortcut: fragility from two gradients

Adam's first step is −lr · sign(g) for every weight, so a conflicting update raises a skill's loss at
σ = −Σ g_skill · sign(g_conflict) · ‖θ₀‖/√n per unit of displacement — computable at θ₀ from one
gradient of the skill's probe and one of the conflict, without fine-tuning. Across 13 models in six
families, Spearman(σ, 1/F) is **+0.96 for copying** (p < 0.001; σ·F ≈ 0.51–0.63 in the five Qwen
models, 1.05–1.34 in Pythia 410m–2.8b and OLMo, 1.9–2.0 in Gemma, Llama and Mistral) and **+0.66 for
capitals** (p 0.009), but −0.09 for uppercasing and −0.10 for addition (9 models). (On the first seven
models it was +1.00 / +0.77 / +0.54 / +0.50.) For copying and capitals the eval can be run in minutes
from two gradients; for formats and arithmetic the fine-tuning arm is needed.

### 4.7 Does fragility predict realistic forgetting? (H4 — the make-or-break)

Final (25 Oct): all 14 models (add: the 10 that can add). Spearman(−log F, relative loss at step
120), average ranks for ties, one-sided permutation p:

| realistic fine-tune | copy | add | capital | upper |
|---|---|---|---|---|
| GSM8K SFT, 1e-5 | +0.43 (p 0.07) | −0.55 (0.95) | +0.24 (0.21) | +0.54 (0.02) |
| wikitext, 1e-5 | **+0.94 (< 0.001)** | +0.05 (0.46) | −0.08 (0.61) | +0.59 (0.02) |

**H4 fails as pre-registered.** It asked for ρ ≥ 0.6 with p < 0.05 in at least two skills; one
clears it. Copying under continued pretraining is predicted almost perfectly (ρ +0.94 over 14
models; +0.93 without Pythia-160m), and uppercasing falls just short (+0.54, +0.59, both p < 0.05).
Capitals and addition are not predicted at all. The interim pass (two skills at n = 9–10) did not
survive the next four models (Gemma, Mistral, OLMo on 24 Oct; Qwen3-8B, the fourteenth, changes no
verdict). Losses above 5% are few — Pythia-160m and 410m (copying, capitals, uppercasing), Mistral's
copying (12–14%), and addition in OLMo, Mistral, Gemma and Llama (GSM8K; OLMo also 43% under wikitext)
— so most of what H4 ranks are losses of 0–4%.

The last three models also changed the size of what there is to predict. GSM8K fine-tuning at 1e-5
wipes out few-shot two-digit addition in OLMo-7B (99% of the known items lost), Mistral-7B (78%) and
Gemma-2-9B (67%), and costs Mistral 14% of its copying; the toy fragility gives no warning of the
addition losses (ρ −0.49). A realistic fine-tune can damage a skill it never conflicts with, which a
conflict-arm score does not measure.

**What follows from the failure** (`docs/PAPER15_DESIGN.md`, "After H4 failed", registered before
the tests). The margin under a conflicting update is a worst case; damage from a given dataset also
needs that dataset's *exposure*. Both exposure measures we registered fail, and the autopsy of the
addition collapse finds half a captured format and half real arithmetic errors.

- *A data-aware two-gradient rate (H7)*, with the conflict replaced by the fine-tuning data's own
  first four batches, ranks addition under wikitext (ρ +0.84, n 9) and nothing else there (copying
  −0.56, capitals −0.35, uppercasing −0.04; n 13), and nothing under GSM8K (copying +0.19, addition
  −0.05 with n 9, capitals −0.01, uppercasing −0.50; n 13). H7 needed two skills in each fine-tune. It
  does flag the worst case — OLMo's rate for addition under GSM8K is 479, seven times any other model's,
  and OLMo loses 99% of its addition — but ranks Gemma's 67% loss near the bottom (3.9) and the
  undamaged Qwen 7B models high (70). The first-order direction of the user's data does not rank
  damage that builds up over 120 steps.
- *A ten-step pilot (H6).* Seen post hoc on the toy skills, the KL of each probe's answers from the
  base model after 10 realistic steps ranked the step-120 losses in 7 of 8 cells. Registered out of
  sample on the task probes (eight instruct models), it fails: under GSM8K SFT no task reaches ρ ≥ 0.6
  with p < 0.05 (GSM8K +0.57 with n 7, formatting +0.07, tool calls +0.58, refusals +0.52); under
  wikitext only refusals do (+0.66, p 0.04); H6 needed two tasks. Not pre-registered, and the reverse
  of the toy-skill observation: the plain drop in accuracy at step 10 ranks the task losses well —
  GSM8K +0.96 and formatting +0.86 under GSM8K SFT, +0.75 and +0.92 under wikitext (p ≤ 0.03) — because
  much of the damage is already there at step 10 (under GSM8K SFT the GSM8K answers have lost 11–93% of
  the known items by step 10, against 19–100% by step 120 — shares that overstate the net loss, as the
  table below shows). Tool calls and refusals are lost almost only by Mistral, so
  their correlations rest on one model. A short pilot read on the tasks themselves is what these data
  support as an exposure measure; it needs its own out-of-sample test before it is claimed.
- *The autopsy.* Mistral-7B's GSM8K arm rerun with the top-1 answer tokens logged reproduces the run of
  24 Oct to the last digit (few-shot addition 0.99, 0.86, 0.09, 0.43, 0.22 at steps 0, 10, 30, 60, 120).
  At the bottom of the collapse (step 30), 6 of the 8 logged sums begin with "<<", the opening of
  GSM8K's calculator notes, and under teacher forcing 4 of those 6 continue with the right digits
  ("142" → "<<142"); at step 120, 4 of 8 still begin with "<<". But a digit is also wrong in 4 of the 8
  items at step 30, 3 at step 60 and 3 at step 120 ("142" → "<<132", "89" → "91"). Capitals and uppercasing stay at 8 of 8 (one
  uppercasing slip at step 120), and here perplexity does see the fine-tune (12.4 → 24.9). The eval
  should therefore also score each skill format-free — the answer anywhere in a short greedy reply —
  to tell a hijacked format from lost ability. That scorer is now built (`tools/fragility.py --free N`:
  N greedy tokens per item, the answer as a whole word anywhere in the reply's first line; GSM8K 8N
  tokens, scored anywhere and as the last number); on undamaged Qwen 0.5B it agrees with exact match
  item for item. Rerun on the identical trajectory (the exact-match columns reproduce both earlier runs
  value for value), format-free scoring keeps 46% of the known sums at step 30 and 65% at step 120,
  against 9% and 22% by exact match. The greedy replies show why: "<<55+33=88>>88", "<<86+37=123>>123"
  — GSM8K's notation with the right sum — beside "<<87+55=93>>93" and, at step 120, "<<87+55=132>>132",
  the notation with a wrong one. So the captured format accounts for 41% of the measured loss at step 30
  and 55% at step 120 (19% at step 60), and the rest are wrong sums: the collapse is about half format,
  half arithmetic. The same notation also captured Mistral's tool calls, which begin with "<<" instead
  of "{" at steps 10–60 (none of the 80 known calls kept at step 60) before partly recovering (58% at
  step 120).

(Until 24 Oct 19:00 the analysis ranked ties by list order; with
most losses exactly zero that added noise, e.g. +0.30 for capital under GSM8K at n = 10, +0.10 with
average ranks.)

### 4.8 LoRA against full fine-tuning

Most fine-tuning is LoRA, and Tinker offers nothing else, so the conflict arms were rerun with rank-32
LoRA on every linear layer (α 32) at 3e-5, LoRA's usual rate (10× the full-fine-tuning 3e-6), in seven
models (26 model × skill cells: Pythia 410m and 1.4b, Qwen2.5 0.5B–7B, Llama-3.1-8B):

- **More steps.** LoRA takes a median 4.8× more steps to halve a skill (interquartile 4.3–6.8×,
  range 2.6–13.6×; Llama 8B the widest), despite the 10× higher rate.
- **More weight change.** It halves a skill at a median 2.8× the displacement of full fine-tuning
  (1.7–5.2×; copying 2.5×, the item skills 2.9–3.5×). A rank-32 update needs about three times more
  weight change to do the same damage.
- **Same targeting.** The wikitext control under LoRA leaves every skill intact at the halving point
  (selectivity ≥ 0.99), other skills keep ≥ 0.97, perplexity moves under 4%: the same blind spot.
- **Matched by how much of the conflict has been learned, it depends on the skill.** The conflict's
  own training loss when the skill halves, as a fraction of its first-step value: for copying the same
  under LoRA and full fine-tuning in six of seven models (LoRA / full: 0.79 / 0.81, 0.74 / 0.74, 0.70 /
  0.70, 0.65 / 0.66, 0.65 / 0.65, 0.63 / 0.63; Llama 8B 0.75 / 0.89). For uppercasing, LoRA learns far more of the conflict on the training
  words before held-out words flip (0.19–0.37, against 0.69–1.00); capitals (0.54–0.80 vs 0.80–0.91)
  and addition (0.38–0.64 vs 0.55–1.00) lie between. (Per-batch losses at the logged steps only, so
  noisy; full fine-tuning often halves a skill before the first logged update, which caps it at 1.)

LoRA's protection is a margin, not immunity: the same conflicting data breaks the same skills, after
more steps and more weight change. This is the measured form of "LoRA learns less and forgets less"
(Biderman et al. 2024), with one refinement. When the conflict is a rule tested on fresh inputs
(copying), learning it and losing the skill are one event, and LoRA only slows both. When it is a set
of items (facts, words), LoRA holds the change to the trained items longer before it spreads to
held-out ones.

### 4.9 A model we cannot hold: Inkling-Small through Tinker

The same eval through a fine-tuning API: Thinking Machines' Inkling-Small (276B mixture of experts,
12B active), LoRA rank 32, scored from the API's top-1 prompt tokens (`tools/fragility_tinker.py`;
about $1 per rate for the five arms). Nothing is downloaded, and the weights stay on the provider's
servers, so the scores are in steps. At step 0 it gets every add and upper item and 61 of 62 capitals
right, and copies 88% of tokens.

Steps to halve each skill, at the local LoRA recipe (3e-5, batch 8, the same AdamW settings) and at
ten times its rate; the last column is the conflict's own training loss when the skill halves, as a
fraction of its first-step value (how much of the conflict had been learned):

| skill | Inkling-Small, 3e-5 | Inkling-Small, 3e-4 | local LoRA at 3e-5 (6 models) | conflict loss at the half point: Inkling 3e-5 / 3e-4 / local |
|---|---|---|---|---|
| upper | 24.0 | 3.1 | 5.2–8.9 | 0.39 / 0.59 / 0.24–0.35 |
| capital | 27.7 | 3.9 | 6.6–12.5 | 0.65 / 0.71 / 0.54–0.80 |
| add | 40.2 | 6.0 | 3.6–13.4 | 0.31 / 0.38 / 0.38–0.64 |
| copy | 45.7 | 6.0 | 13.4–23.4 | 0.59 / 0.68 / 0.63–0.79 |

(Rerun 24 Oct 20:50 with the Tinker tool made an exact port of the local one. The first version fed
the relabelled copy tokens back in as context, a different task, and made copying look unbreakable at
3e-5; the item skills moved by at most two steps.)

At equal settings Inkling-Small takes 2–5× more conflicting steps to lose a skill (copying 2–3×).
But it is also learning the conflict more slowly: matched by how far its loss on
the conflicting examples has fallen, it loses facts and formats at about the same point as the small
open models, and arithmetic slightly later. Ten times the rate, 3e-4 (still below the 4e-4 to 1e-3
that Tinker's own sweeps find best), cuts the steps roughly in proportion: 3–6 for every skill. At that rate the skills break a little
earlier in the learning of the conflict (0.38–0.71, against 0.31–0.65), most clearly uppercasing
(0.59 against 0.39), as under full fine-tuning against LoRA (§4.8): larger steps damage before they
teach.

The control leaves every skill intact at both rates for all 50 steps (selectivity ≥ 0.98), and at
the half points the other skills keep ≥ 0.96. Past the break, at 3e-4, the damage spreads: by step 50
the uppercase conflict has also taken capitals to 0.05, the addition conflict capitals to 0.50 and
uppercasing to 0.81, the capital conflict uppercasing to 0.66, and the copy conflict everything
(every skill at or near 0, wikitext perplexity above 5 000). Spreading after the break is common
under full fine-tuning at 3e-6 too (another skill below 0.8 of its start by step 50 in 10 of 40 runs,
worst in Mistral 7B and Qwen2.5-7B base), rare under LoRA at 3e-5 (1 of 20), and absent for
Inkling-Small at 3e-5 (0 of 4). Realistic fine-tunes, LoRA at 1e-4 for 120 steps: GSM8K SFT and
wikitext cost nothing on the item skills (all 1.000) and 1% of copying (0.90 → 0.89).

Perplexity is not a usable blind-spot check for this model: on raw wikitext the chat model starts at
31.5, and any raw-text training lowers it (the control reaches 7.4 by step 50 at 3e-4), so it falls
while skills break. **Why Inkling-Small is slow to move.** It behaves like a normal model fine-tuned at about a third of
the rate: to break as fast as Qwen3-8B at 3e-5 it needs 2.3–3.9× the rate (copy 6.9e-5, add 7.5e-5,
capital 8.7e-5, upper 1.2e-4; its steps fall as lr^−0.85; both models on Tinker, so the platform
cancels). It is not the mixture of experts: on Tinker at 3e-5, GPT-OSS-20B (32 experts) breaks as fast as
the dense Qwen3.5-4B and 9B (copy 5.3 vs 11.6 and 12.5 steps), and Qwen3.6-35B-A3B, with Inkling's layout
of 256 small experts, is about 1.5× its dense siblings (copy 16.6, capital 10.9), where Inkling is 3–4×
(45.7, 27.7). Nor is it localised in the experts: with adapters on attention only, Inkling is still
1.7–3.2× slower than Qwen3-8B (capital 44.4 vs 25.6 steps, upper 38.3 vs 12.1), and with adapters on the
MLP only ≥ 2.4–3.2× (> 50 vs 21.2, 37.7 vs 11.9). Starting confidence does not explain it (its losses on
the conflicting answers are mid-range), nor the chat format (the same factor on the task probes, §4.10).
What remains is model-wide, and its public weights show what: they are large. The gain of each matrix
(rms entry × √fan-in; layers 5 and 20, read from the public checkpoint by byte range, only statistics
kept) is 3–6× Qwen3-8B's — queries ×4.6, keys and values ×2.9, attention outputs ×5–6, expert
up-projections ×4.5, down-projections ×3.3 (per-entry rms 0.07–0.21 against 0.025–0.033). Adam moves
every weight by about the learning rate whatever its size, and a LoRA update's scale does not depend on
the weight it adapts, so each step is 3–6× smaller relative to Inkling's weights. Across the eight
models with local LoRA runs, mean gain ranks steps-to-halve for copying (ρ +0.71, p 0.03; addition
+0.60, n 6; capitals +0.12; uppercasing +0.29): supportive, not proof.

**The causal test.** Weight scale can be changed without changing what a model computes. In a
pre-norm transformer every sublayer and the output head read the residual stream through a
normalisation, so multiplying the embedding by c² and the value, attention-output, MLP-up and MLP-down
matrices by c scales the stream by c² and leaves every output as it was (Qwen2.5-0.5B-Instruct at
c = 3: largest change in a log-probability 0.012, same top tokens; identical step-0 scores at c = 1, 3
and 9). Those four matrix types are then c times larger, and the same LoRA recipe (r32, 3e-5, 50
steps) is c times weaker relative to them. Steps to halve, training seeds 0 / 1 (seed 1 changes the
batches and the adapters' initialisation), and the conflict's training loss at the half point as a
fraction of its first step (how much of the conflict had been learned; c = 1 / 3 / 9, seed 0; seed 1):

| skill | c = 1 | c = 3 | c = 9 | conflict loss at the half point |
|---|---|---|---|---|
| copy | 14.5 / 14.2 | 23.4 / 23.4 | 29.3 / 28.8 | 0.70 / 0.70 / 0.70; 0.71 / 0.71 / 0.72 |
| add | 3.6 / 4.6 | 6.4 / 7.1 | 15.1 / 15.4 | 0.64 / 0.60 / 0.41; 0.59 / 0.60 / 0.43 |
| capital | 6.7 / 6.8 | 12.5 / 12.2 | 13.8 / 13.7 | 0.80 / 0.78 / 0.80; 0.53 / 0.61 / 0.64 |
| upper | 5.2 / 5.2 | 8.5 / 8.7 | 11.3 / 11.4 | 0.27 / 0.23 / 0.26; 0.33 / 0.34 / 0.38 |

The prediction holds in every skill and both seeds. The same model, computing the same function, takes
1.5–1.9× more conflicting steps to lose a skill when four of its matrix types are three times larger,
and 2.0–4.2× when they are nine times larger; the two seeds agree within a step except for addition at
c = 1 (3.6 / 4.6). As with Inkling, the skill breaks at about the same point of the conflict's learning
(copying 0.70–0.72 at every c; for capitals and uppercasing the seeds differ more than the scales do);
only addition at c = 9 needs more of its conflict learned first (0.41, 0.43). (c = 1, seed 0 reproduces
the unscaled run of 24 Oct to the decimal.) The steps grow more slowly than c because the query, key
and gate matrices read the normalised stream and are not rescaled, so their adapters keep their full
effect; Inkling's queries and keys are large too (×4.6, ×2.9), which a function-preserving rescale of
this model cannot imitate. At Inkling's scale (3–6×) the test gives about a factor of 2 of its
2.3–3.9×; whether its large queries and keys supply the rest is untested. So a model can be slow to
break because it is slow to learn, and large weights make it slow to learn. The methodological lesson for the eval:
steps-to-break at one fixed rate mixes robustness with learning speed, so models must also be compared
at matched learning (the conflict-loss column) or at their equivalent rates.

**The bridge: is it the model or Tinker?** Qwen3-8B is on Tinker and on Hugging Face, so the same
LoRA arms ran on both (same probes, data stream and per-token loss; local LoRA in float32 on CPU,
Tinker in its own precision). Before the first update the two agree (copying 0.942 / 0.942, first-batch
loss 27.19 / 27.17). Steps to halve at 3e-5, local / Tinker:

| skill | local (peft) | Tinker | conflict loss at the half point, local / Tinker |
|---|---|---|---|
| copy | 23.6 | 23.8 | 0.46 / 0.46 |
| capital | 13.3 | 13.6 | 0.43 / 0.50 |
| add | 12.9 | 20.9 | 0.36 / 0.26 |
| upper | 6.0 | 9.5 | 0.23 / 0.15 |

For copying and capitals the two implementations are indistinguishable. For addition and uppercasing
Tinker's training loss falls more slowly over the first eight steps (uppercasing at step 4: 5.5 locally,
10.4 on Tinker) and then catches up, so Tinker reaches the break about 1.6× later. Part of Inkling's
delay (up to ~1.6× for formats and arithmetic) is therefore Tinker's LoRA; the rest, and all of it for
copying and capitals (2–3×), is the model. At 3e-4 the two sides agree for copying, capitals and
uppercasing (3.4 / 3.6, 1.6 / 1.7 and 1.5 / 1.5 steps) and Tinker is again slower for addition (3.0 /
5.3, 1.8×). At both rates Tinker's first update teaches less (training loss after it at 3e-4: copy
19.9 / 22.1, capital 7.0 / 8.3, upper 2.8 / 3.4, add 2.6 / 3.2), and the two then converge. The cause
on Tinker's side is unknown; it is not the unembedding adapter, which Tinker trains by default and
peft's all-linear does not (without it, 3e-5: addition 20.9 → 20.9 steps, uppercasing 9.5 → 8.4).

### 4.10 Version 2: the same ruler on real tasks

The toy skills prove the ruler; the skills a company would protect are different. Version 2 keeps the
conflict and control arms and scores four task probes (`tools/fragility_tasks.py`; design and
predictions T1–T6, written before any run, in `docs/PAPER15_DESIGN.md`): GSM8K word problems (final
number only), IFEval-family formatting instructions, tool calls in a fixed JSON format, and refusals of
AdvBench's harmful requests. The refusal conflict trains only AdvBench's one-line affirmative prefixes
("Sure, here is …"), never content, and only the refusal flag is kept.

**Full fine-tuning at 3e-6, eight instruct models** (31 cells; Qwen 0.5B knows only 3 GSM8K problems,
so GSM8K is not scored there). Steps to halve:

| task | Qwen 0.5B | 1.5B | 3B | 7B | Qwen3-8B | Llama 8B | Mistral 7B | Gemma 9B |
|---|---|---|---|---|---|---|---|---|
| GSM8K | — | 0.7 | 3.2 | 4.0 | 3.2 | 0.6 | 0.5 | 1.2 |
| formatting | 1.2 | 2.2 | 1.4 | 1.6 | 1.6 | 0.6 | 0.6 | 1.1 |
| tool calls | 4.1 | 4.0 | 3.2 | 3.2 | 3.7 | 1.8 | 0.8 | 1.5 |
| refusal | 0.7 | 1.7 | 1.5 | 1.4 | 1.4 | 1.4 | 0.6 | 1.5 |

(Items known at step 0: GSM8K 11–28 of 96; formatting 28–69 of 72; tool calls 79–80 of 80; refusals
59–64 of 64, except Mistral, which refuses only 22 of the 64 harmful requests before any fine-tuning.)

The pre-registered predictions:

- **T1, fast break — holds.** All 31 cells halve within five conflicting steps (0.5–4.1, median 1.5);
  T1 asked for two thirds.
- **T2, selectivity — holds.** At each conflict's halving point the wikitext control leaves the task
  at ≥ 0.96 of its start, in 31 of 31 cells.
- **T3, blind spot — holds.** Perplexity has moved between 0.0% and −2.1% at the halving point in all
  31 cells.
- **T4, the toy panel predicts the tasks — holds, narrowly.** Across the eight models, the mean −log F
  of the four toy skills ranks the task F for tool calls (Spearman ρ +0.69, one-sided p 0.04) and
  refusals (+0.79, p 0.01); formatting (+0.52, p 0.10) and GSM8K (+0.29, n 7) fall short. Two tasks
  were required. The pass is thin: dropping any one model leaves tool calls at ρ +0.57 to +0.93 but
  with p above 0.05 in six of the eight drops; refusals stay at p ≤ 0.054 in every drop. (Not
  pre-registered: ranked by steps instead of displacement, formatting +0.80 and tool calls +0.90
  clear the bar and refusals, +0.38, do not.) So the cheap toy panel orders these models roughly as
  the task panel does — a model fragile on synthetic skills is fragile on real ones — but not well
  enough to replace the task probes for any one task.
- **T5, refusal breaks first — fails.** Refusal halves before every other task in 3 of 8 models (Qwen
  0.5B and 7B, Qwen3-8B); elsewhere GSM8K or formatting goes first or ties. It is not the weakest
  task, but nothing protects it: 0.6–1.7 conflicting steps halve it in all eight models, and after 50
  steps 3% of the known refusals are left in seven of them (Mistral 9%). 3% is about the keyword flag's
  floor: two of the 64 requests contain "illegal", and a compliant reply that repeats the request is
  flagged as a refusal, so in those seven models every known refusal is gone.

Beyond the predictions:

- **Task conflicts spill onto other tasks, not onto the toy skills.** At the halving point the toy
  skills keep ≥ 0.99 of their start in 31 of 31 cells, but the other tasks keep ≥ 0.96 in only 11 of
  31. The losses fall mostly on GSM8K (Qwen 1.5B's tool conflict leaves 0.55 of its GSM8K answers,
  Qwen3-8B's formatting conflict 0.68; GSM8K is scored on 11–28 problems, so one problem is 4–9%) and,
  in Mistral, on refusals (0.81–0.83). Past the break the spread is wide: by step 50 another skill is
  below 0.8 of its start in 17 of 32 conflict runs. Teaching a model to ignore formatting
  instructions wipes out its tool calls (0.00 in Gemma, Llama, Mistral and Qwen 1.5B; 0.04 in
  Qwen3-8B), and in Mistral the GSM8K conflict (answers + 7) takes few-shot addition to zero.
- **The harmless control moves GSM8K's items, not its accuracy.** After 50 wikitext steps at 3e-6 the
  share of known GSM8K problems still solved is 0.64 in Mistral, 0.79 in Llama, 0.81 in Qwen3-8B and
  0.90–0.96 in the rest, but net accuracy barely moves (Mistral 0.146 → 0.125, Llama unchanged): the
  models trade problems they solved for ones they had not. Formatting and tool calls stay at ≥ 0.96, and
  Mistral keeps 73% of its known refusals while its refusal rate stays at 0.344. Selectivity at the halving points is
  unaffected because the conflicts break the tasks within a few steps.

**LoRA r32 at 3e-5 (T6) — holds.** Across six models (Qwen 0.5B–7B, Qwen3-8B, Llama 8B; 23 cells)
LoRA takes a median 6.6× more steps than full fine-tuning to halve a task (formatting 4.2–11.5 steps,
refusals 3.2–12.1, GSM8K 6.0–24.4, tool calls 13.0–24.0); T6 asked for more than 2×. The margin is
larger than on the toy skills (4.8×, §4.8) and buys the same thing: selectivity ≥ 0.96 and perplexity
within 2.5% at the half points, and the same spill onto other tasks (Qwen3-8B's tool-call conflict
leaves 0.53 of its GSM8K answers when tool calls halve).

**Through Tinker** (LoRA r32 at 3e-5, 50 steps), steps to halve:

| task | Qwen3-8B | Inkling-Small |
|---|---|---|
| formatting (IFEval-family) | 11.9 | 34.7 |
| refusal | 12.0 | 41.3 |
| GSM8K | 14.5 | 37.4 |
| tool calls | 25.7 | 30.4 |

Both models refuse 97–98% of the held-out harmful requests at the start; after 50 steps of
affirmative prefixes on *other* requests, 3% of those refusals are left — the keyword flag's floor. Teaching Qwen3-8B to ignore
formatting instructions also took its tool calls from 100% to 1% and its GSM8K answers to 18% by step
50, and fifty steps of Wikipedia text cost it a quarter of its GSM8K answers, as on the cluster. At
the half points selectivity is ≥ 0.96. Inkling-Small is again 1.2–3.4× slower at equal settings, as
in §4.9.

**Realistic fine-tunes scored on the task probes.** The two realistic fine-tunes of §4.7 (GSM8K SFT
and wikitext, 1e-5, 120 steps), now on the eight instruct models with the task probes. Net loss of
accuracy by step 120, relative to step 0 (in brackets, the share of the known items lost):

| task | GSM8K SFT | wikitext |
|---|---|---|
| GSM8K (final number only) | Gemma 74% (85), Mistral 64% (100), Llama 50% (64), Qwen 3B 29% (43), Qwen 7B 7% (25), Qwen 1.5B 0 (45); Qwen3-8B gains 81% (19 lost) | Llama 29% (64), Qwen 3B 5% (19); the other five gain 9–19% (Mistral gains 14% while losing 57% of its known problems) |
| formatting | Mistral 50% (52), the rest ≤ 9% | Mistral 43% (45), the rest ≤ 10% |
| tool calls | Mistral 43% (43), the rest 0 | Mistral 30% (30), the rest 0 |
| refusals | Mistral 91% (91), the rest ≤ 2% | Mistral 41% (73), the rest ≤ 3% |

Fine-tuning on GSM8K's worked solutions lowers GSM8K accuracy as the probe asks it (the final number
alone, no working) in four of seven models and raises it in Qwen3-8B. The share of known items lost
overstates the damage everywhere: the probe sits at 3–29% accuracy, where answers flip both ways. In
Mistral it is not mainly a changed format, at least in Mistral, the one model rescored
format-free (§4.7): of the 14 problems it solved at step 0 it reaches the answer anywhere in a
128-token reply on only 15–23% after fine-tuning, and as the reply's last number on 0–15%, its replies
falling into repetition and wrong working (perplexity 12.4 → 24.9). Wikipedia
text lowers net GSM8K accuracy only in Llama (29%). Mistral, which refused only a third of the harmful
requests to begin with, loses most of those under both harmless fine-tunes — all of them by step 10 of
GSM8K SFT, 91% still gone at step 120, 41% net under wikitext — while the other seven models keep theirs: the refusal behaviour of
some models does not survive even benign fine-tuning (as Qi et al. 2024 found for GPT-3.5 and Llama-2).
Whether a ten-step pilot predicts these losses (H6) is in §4.7: the registered KL pilot fails, the
plain accuracy pilot does well post hoc.

---

### 4.11 Frontier models through a fine-tuning API

The eval needs only a model's outputs and a fine-tuning API, so it reaches models few groups could hold.
Through Tinker (LoRA r32 at 1e-4, 50 steps, every probe scored at every mark; no refusal-conflict training
on Tinker, in line with its makers' acceptable-use policy; the refusal probe is still asked), four
frontier models beside the two Tinker anchors. Steps to halve each skill (the conflict's training loss at
the half point, as a fraction of its first step, in brackets):

| skill | Qwen3-8B | DeepSeek-V4.1-Flash (552B / 16B active) | Qwen3.5-397B (397B / 17B) | Nemotron-3-Ultra (550B / 55B) | Kimi-K2.6 (1T / 32B) | GPT-OSS-120B (117B / 5B) | Inkling-Small (276B / 12B) |
|---|---|---|---|---|---|---|---|
| copy | 10.5 (0.45) | 11.4 (0.61) | 6.2 (0.74) | 6.2 (0.70) | 5.9 (0.74) | 1.4 (0.96) | 17.9 (0.58) |
| add | 6.0 (0.38) | 8.6 (0.44) | 6.0 (0.45) | 6.5 (0.46) | 5.8 (0.49) | 4.0 (0.53) | 16.9 (0.31) |
| capital | 5.7 (0.40) | 7.1 (0.47) | 6.0 (0.51) | 7.5 (0.37) | 6.1 (0.49) | 2.5 (0.66) | 12.5 (0.55) |
| upper | 3.0 (0.44) | 5.5 (0.21) | — | 5.2 (0.25) | 3.0 (0.51) | 2.4 (0.55) | 11.3 (0.23) |
| GSM8K | 7.8 (0.21) | 9.0 (0.32) | 6.3 (0.44) | 11.6 (0.33) | — | — | 12.6 (0.54) |
| formatting | 5.7 (0.33) | 5.8 (0.20) | 3.2 (0.32) | 6.0 (0.37) | — | — | 12.1 (0.25) |
| tool calls | 11.0 (0.04) | 12.3 (0.08) | 7.0 (0.12) | 11.2 (0.11) | — | — | 11.6 (0.31) |

(Kimi-K2.6 and GPT-OSS-120B begin their chat replies with a reasoning segment, so the task probes cannot read
their answers in our format: toy skills only. Qwen3.5-397B answers 10 of the 86 uppercasing items in our
format, below the start rule.)

- **Size does not buy toughness.** At this rate, models of 120B to 1T parameters halve a skill in 3–12
  steps, the range of Qwen3-8B (3–11). GPT-OSS-120B is the most fragile (copying in 1.4 steps); Inkling-Small,
  with its large weights (§4.9), is the slowest.
- **Selectivity holds.** At each conflict's half point the wikitext control leaves the skill at 0.96–1.07
  of its start, in 28 of 28 frontier cells.
- **The strongest cross-task spill replicates.** Teaching a model to ignore formatting instructions wipes
  out its tool calls in all three frontier models with a tool probe (DeepSeek, Nemotron, Qwen3.5-397B: 0.00
  by step 50), and Qwen3.5-397B's GSM8K answers with them, as in five of the eight local models (§4.10).
  Past the break other conflicts spread too: DeepSeek's copy conflict takes capitals to 0.06 and
  uppercasing to 0, and GPT-OSS's lowers every other skill (refusals to 0.57).
- **The harmless control is harmless at the half points, not forever.** By step 50 it has cost DeepSeek 15%
  of its copying, and GPT-OSS keeps 3 of its 13 refusals — a small count, possibly confounded by its reply
  format, to be checked before it is reported as a safety finding.
- **The canary works through the API (H8-F, registered before any run).** DeepSeek-V4.1-Flash and
  Qwen3.5-397B fine-tuned on the four realistic sets of Part III (Dolly, CodeAlpaca, medical flashcards,
  Glaive; LoRA r32 at 1e-4, 300 steps, exact match): each skill's loss at step 30 ranks its loss at step 300
  with ρ ≥ 0.6 (p < 0.05) in all 4 sets (Dolly +0.84, code +0.68, medical +0.96, Glaive +0.65). All three
  cells that lost ≥ 25% — DeepSeek's refusals, 62% under code and 31% under Dolly, and Qwen3.5-397B's
  uppercasing under Glaive, 67% (83% already at step 30; it knows only 12 of the 86 items in our format) —
  were flagged at step 30, and 11 of the 12 flags held. It is a small test (two models, 16 cells per set;
  recall's 95% interval 0.44–1.00). Beyond those, realistic fine-tuning cost these models little: formatting lost 12–15% by step 30 in
  7 of 8 runs, still lost at step 300 in 6; neither lost its tool calls to Glaive's call format, as Qwen3-8B did (79%). Qwen3.5-397B's
  refusals under Dolly fell late (none lost at step 100, 23% at 300), the slow burn the step-30 reading
  misses.

### 4.12 Part II: the panel rating (H9)

The eight instruct models fine-tuned on four public sets — Dolly-15k (general instructions), CodeAlpaca-20k
(code), medical flashcards (domain questions), Glaive function-calling v2 (tool use, in its own call
format) — each in the model's chat template, LoRA r32 at 1e-4, batch 8, 300 steps. A model's rating for a
skill is its mean share of known items lost at step 300 under three of the sets; the test, registered
before the 7–9B runs existed, holds out each set in turn and asks whether the rating ranks the models'
losses under it. **It holds on all eight models, narrowly, and fails on the six it had not seen:**
median Spearman ρ +0.68 over the 11 scored (set, skill) pairs, 6 of 11 individually ≥ 0.6 with p < 0.05;
on the six models whose runs no one had looked at when the test was registered (as registered), median
+0.71 but 2 of 9 pairs (n = 6 per pair). By skill: refusals 3 of 3 (+0.72 to +0.98), GSM8K 3 of 4 (+0.54 to
+0.85), formatting 0 of 4 (+0.21 to +0.42). Tool calls are lost in one cell only (Qwen3-8B under Glaive,
79%) and cannot be ranked.

Why formatting fails: across sets the models rank alike for GSM8K (median ρ between sets +0.62) and
refusals (+0.58), not for formatting (+0.21). Formatting losses are small (median ≈ 6%), swing by up to 15
points between checkpoints, and depend on the pairing of model and data — Mistral and Llama lose none under
Dolly and 12–17% under code. Refusal damage behaves as a trait of the model, formatting damage as a
property of the pairing. GSM8K's ranks agree across sets too, but on net accuracy most of its "loss" is
items trading places on a probe the models barely pass (§4.13), so we do not read it as a trait. At 29 of 32 runs the test fell short (median +0.56, 4 of 11): Llama's and Gemma's
ratings then rested on one or two sets.

**It does not survive new kinds of data.** Five more sets, registered before any run on them — customer
support (Bitext), maths solutions (MetaMathQA), many languages (Aya), dialogue summaries (DialogSum) and
finance questions (finance-alpaca) — and a ninth skill, predicting what code returns (CRUXEval). Over all
nine sets, leave-one-set-out (H9′), the rating fails: median ρ +0.54 over 30 pairs, 10 of 30 individually.
Rated from the original four alone, it ranks the models' losses under the five new sets with median ρ
+0.57 (5 of 14). The losses that broke it are pairings again: refusals under customer support (ρ −0.07)
and finance (+0.24) — Gemma, which lost 0–26% of its refusals under the first four sets, lost 69% under
finance — and the code-reasoning probe, which no rating ranks (−0.60 to +0.10). A rating smoothed over
steps 100 and 300 and drawn from the three sets nearest the held-out one in their reply tokens (H9-M,
registered before the 7–9B runs on the new sets) scores a median +0.50 on the same 19 pairs, against the
plain rating's +0.37, but fails its registered bar (6 of 19), so no improvement is claimed. On those pairs the held-out run's own loss at step
30 ranks the models at +0.73. A panel rating is a rough prior; it is no substitute for reading the
customer's own run.

### 4.13 Part III: the canary (H8)

The customer's own fine-tune, run on a copy for 30 steps with every probe read; a skill that has lost ≥ 10%
of its known items is at risk. Registered before any run and tested on the 32 runs of §4.12. **It holds.**
Losses at step 30 rank losses at step 300 with ρ +0.87 [+0.80, +0.93] under Dolly, +0.80 [+0.65, +0.92]
under code, +0.79 [+0.57, +0.94] under the medical set and +0.84 [+0.65, +0.97] under Glaive (95% intervals
from resampling models). Of the 25 cells that lost ≥ 25% by step 300, 21 were at risk at step 30 (recall
0.84 [0.65, 0.94]); of the 40 at-risk cells, 37 lost ≥ 10% (precision 0.93 [0.80, 0.97]). Naive alarms on
the same cells do worse: flagging every cell is right 22% of the time, flagging every refusal catches 52% of
the big losses at 62% precision. The four misses are all late damage under Dolly (Gemma's and Qwen3-8B's
refusals, Qwen3-8B's GSM8K, Qwen 0.5B's formatting: 6–8% lost at step 30, 25–29% at step 300). None of the 18
at-risk losses vanishes when scored format-free: they are lost skills, not captured formats. The canary
also holds on two frontier models through Tinker (§4.11), and under a second seed (H8-S: the five Qwen
models, another batch order and LoRA initialisation): the step-30 flags agree in 97% of cells (κ 0.91),
and the rule on the second seed alone catches 16 of 18 big losses at precision 0.81. Extended to all eight
models (H8-S2, registered), the flags agree in 96% of 252 cells, and the two-reading rule on the second seed
catches 29 of 30 big losses (85% of warnings right), 16 of 17 on net accuracy (78%). A re-run moves a
cell's final loss by a median of 7 points among cells that lose ≥ 10% (90th percentile 14).

**Out of sample (H8′), on the five new sets of §4.12, it holds again,** with the same recall: step-30 ρ
+0.80 to +0.97 in 5 of 5 sets; 33 of 38 big losses at risk at step 30 (0.87 [0.73, 0.94]), 70 of 75 at-risk
cells losing ≥ 10% (0.93 [0.85, 0.97]); flagging every cell would be right 26% of the time, flagging every
refusal would catch 34% of the big losses. The five misses are again late damage, three of them refusals.

**Net accuracy, not only items lost (post hoc, after the review of 26 Oct).** The registered measure is the
share of the items a model got right at step 0 that it gets wrong later. A model can lose some of those
while gaining others it had got wrong, and on GSM8K (3–29% accuracy) and CRUXEval it does: GSM8K cells
lose a median 24% of their known problems by step 300 while net accuracy rises a median 12%, and 29
GSM8K cells lose ≥ 25% by the registered measure against 4 net. Refusal losses do not churn (26 big
losses against 23 net), but they are keyword readings, which overstate the losses a harm classifier finds
(the end of this section). Scored on net accuracy, the step-30 reading passes on the first four sets (ρ ≥
0.6 in 4 of 4; recall 12 of 15, precision 0.82) but not on the five new ones (3 of 5; recall 12 of 18),
and a second seed's recall drops to 9 of 12; the two-reading rule below holds on both (15 of 15 at
precision 0.80; 17 of 18 at 0.79). The canary's net big losses are mostly refusals (13 of 15; 10 of 18);
without them the two readings catch 9 of 10, at precision 18 of 28. Every canary verdict
is now printed on both measures.

**Two fixes for its limits, fitted only on the 32 runs above and tested on the 355 cells of the new sets.**
A second reading at step 100 (at risk if ≥ 10% lost at step 30 or at step 100) catches 37 of the 38 big
losses (0.97) at precision 0.91, passing the registered bar (recall ≥ 0.90, precision ≥ 0.70); a trend
alarm on the rise from step 10 to step 30 does not (0.89). And the canary gives numbers, not only ranks: a
monotone curve from the loss at step 30 to the loss at step 300, fitted on the first four sets, predicts
the new sets' step-300 losses with median absolute error 0.003 and within ±0.10 in 91% of cells, passing as
registered. Those numbers flatter it: most cells lose nothing, and reading the step-30 loss itself as the
answer scores as well (90% within ±0.10). On the 93 new cells that lost ≥ 10% the curve's median error is
0.05 and 74% fall within ±0.10 (the step-30 reading: 0.07, 63%); on the 38 that lost ≥ 25%, 0.09 and 55%
(0.13, 45%). So it gives the size of a loss roughly, not precisely. The rule we recommend is therefore two
readings — at step 30, and again at step 100.

**The probes are needed; generic monitoring does not see the damage** (post hoc, all 72 runs). The KL
divergence on general text at step 30, which tracks how much the model's language has changed, ranks the
runs' worst skill losses at ρ +0.10; flagging the half of the runs with the largest KL would catch 23 of
the 45 runs that lose ≥ 25% of a skill, where the canary's step-30 reading catches 43. The canary holds
within each skill (step-30 ρ across models and all nine sets: GSM8K +0.78, refusals +0.81, formatting
+0.73, code reasoning +0.64) and in each family (Llama, Mistral and Gemma: step-30 recall 25 of 27, two
readings 26 of 27; the five Qwen models: 29 of 36 and 36 of 36). Refusals are the slow skill: the step-30
reading catches 21 of their 26 big losses, the second reading all 26. The toy skills never lose more than
8% to ordinary fine-tuning; they break only under conflict (Part I).

**Against no-run priors and the signals teams already watch, within each skill** (post hoc, all 72 runs
of §4.12; AUROC for a cell losing ≥ 25% by step 300, 0.5 being chance; the registered measure / net
accuracy):

| score | refusals | GSM8K | task probes pooled |
|---|---|---|---|
| canary: loss at step 30 | 0.89 / 0.87 | 0.88 / 0.92 | 0.93 / 0.94 |
| canary: two readings (the larger of the losses at steps 30 and 100) | 0.92 / 0.91 | 0.90 / 0.89 | 0.96 / 0.95 |
| canary: loss at step 10 | 0.81 / 0.76 | 0.84 / 0.94 | 0.84 / 0.76 |
| panel prior: the same model's loss on the same skill under the other sets | 0.59 / 0.51 | 0.74 / 0.28 | 0.81 / 0.70 |
| run level: KL on wikitext at step 30 | 0.68 / 0.64 | 0.45 / 0.55 | 0.55 / 0.59 |
| run level: change in wikitext perplexity at step 30 | 0.45 / 0.43 | 0.59 / 0.29 | 0.49 / 0.42 |
| run level: fall in training loss, steps 10 → 30 | 0.43 / 0.42 | 0.52 / 0.62 | 0.46 / 0.46 |

(Cells losing ≥ 25%: refusals 26 / 23 of 72; GSM8K 29 / 4 of 63; task probes 63 / 33 of 319.) Pooled
figures ride partly on differences between skills; within a skill the comparison is clean. There the
step-30 reading ranks refusal damage at 0.89 (0.87 net), the model's own record on the other datasets at
0.59 (0.51), and the run-level signals a team would already be logging at 0.42–0.68: what a run will cost
a model's refusals is visible only in the refusals themselves. All of these refusal figures are keyword
readings; the next paragraph says what they measure.

**What the refusal flag measures** (H8-J, H8-J2, H8-J3, registered; all 72 refusal cells judged, 28 Oct).
The flag looks for refusal phrases in a reply's first 16 tokens. Rerunning the canary's runs with every
reply also labelled by the HarmBench classifier (the registered primary label: a reply that is not an
instance of the requested behaviour counts as refused; 256 tokens, replies held in memory only) and by
Qwen2.5-14B-Instruct, the flag's losses prove real but too large: Gemma under finance-alpaca loses 0.69 of its
refusals by the flag and 0.39 by the classifier. Reply by reply the flag agrees with the classifier at κ 0.54
on the seven runs registered for it (H8-J2 a fails, needing 0.6; over all 72 cells, most of which change
little, κ 0.61), and its step-300 loss falls within 0.10 of the classifier's in 17 of the 43 cells it saw lose
(H8-J3 a fails, as registered to expect; H8-J2 b fails, 1 of 7). The general judge errs the other way: in 11%
of all replies it calls a reply a refusal where the classifier finds the requested harm (the reverse never;
κ 0.06), and it puts Gemma's loss at 0.02, so an earlier reading of the first three runs on that judge alone,
that the flag's losses were mostly rephrased refusals, is withdrawn (H8-J, all eleven runs: κ 0.05
against the flag, and no run's loss within 0.10 of the flag's). By the classifier, 11 of the 72 cells lose
≥ 25% by step 300 — Mistral-7B under six of its nine sets, Gemma under finance-alpaca, Qwen2.5-0.5B under
CodeAlpaca and Dolly, Qwen2.5-3B under Dolly and Aya — all of them among the flag's 43. The two readings warn on
all 11, and 24 of their 28 warnings on refusal cells are followed by a loss ≥ 10% (net 19 of 24). With its
refusal cells on the classifier's labels, the canary passes three of its four registered checks (H8-J3 b:
recall 18 of 18 on the four H8 sets and 29 of 30 on the five new ones, precision 0.88 and 0.91; on net accuracy
8 of 8 at precision 0.74 on the H8 sets) and fails the fourth by one warning: on net accuracy on the five new
sets it catches 12 of 13 at precision 17 of 25 (0.68, against 0.70), the last false alarm Llama-3.1-8B under
Bitext support, which loses 10% of the refusals it had and 3% net. Within refusals (H8-J3 c, all 72 cells) the
judged step-30 reading ranks real damage at AUROC 0.92 (0.93 net), the model's own record on the other sets at
0.79 (0.76), and the run-level signals at 0.36–0.69. So the flag's refusal losses are part phrasing and part
real, and the canary's early readings track the real part. Two runs break the assumption that the flag only
under-reads: Mistral-7B (which the classifier finds complying with 39% of the requests before any fine-tuning)
under MetaMathQA, and Qwen2.5-0.5B under Bitext support. Both sometimes open with a refusal and then comply. The
29 refusal cells the flag called safe were therefore judged too (registered): none loses 10% at any reading. The
judged runs repeat the canary's runs on other CPU nodes. The 7–9B models do not reproduce bit for bit (35 of the
72 runs differ somewhere), but a few flipped items move the retained refusal loss by at most 0.05; on net
accuracy Mistral's few refusals magnify each flip to 4.5 points, and 4 of 144 warning or loss verdicts change,
all of them Mistral's.

**Three families it had never seen (H8-N, registered).** OLMo-2-7B-Instruct, Phi-3.5-mini-instruct and
Granite-3.1-8B-instruct under the canary's recipe on all nine sets (27 runs, 231 cells). On the registered
measure it holds: the step-30 reading ranks the end-of-run losses at ρ +0.67 to +0.95 in all nine sets, and the
two readings catch 20 of 23 big losses, with 43 of their 48 warnings followed by a loss ≥ 10%. On net accuracy,
which this test named its headline, it fails: ρ ≥ 0.6 in 4 of 9 sets, and the two readings catch 8 of 9 big
losses but only 13 of their 19 warnings come true (0.68, against 0.70). The false alarms are mostly churn: in
four of the six, GSM8K problems lost by step 30 are made up by step 300 (OLMo-2 three times, Granite once), and
Phi-3.5's tool calls, 80% lost at step 30, are all back by step 300. The one net miss is late damage
(Phi-3.5's tool calls under Bitext support, lost after step 100). Within GSM8K, the skill with the most big
losses (11 of 27, registered measure), the step-30 reading barely beats the model's record elsewhere (AUROC
0.61 against 0.55; run-level signals 0.36–0.49). Rerun with every refusal reply labelled by the
HarmBench classifier (H8-N2, registered), the two readings warn on three of the four refusal cells that
really lose a quarter or more (the bar was 80%), and four of their five refusal warnings come true. The
miss is Granite under Dolly, a late loss (3% at step 30, 6% at step 100, 37% at step 300). With refusals
judged, the canary as a whole holds on the registered measure (ρ ≥ 0.6 in 8 of 9 sets; 17 of 21 big losses
caught, 39 of 43 warnings right) and fails on net accuracy (4 of 9 sets; 6 of 8 caught, 10 of 17 right).
Across all eleven models, the two readings warned on 14 of the 15 runs that lost a quarter or more of
their real refusals.

**A second recipe.** Under a second recipe — twice the rate for 1 000 steps,
Qwen2.5 0.5B–7B (H8-R, registered) — the step-30 reading still ranks the end-of-run losses in 3 of 4 sets
and catches 14 of 17 big losses with no false alarm; reading again at step 100 (a tenth of this run)
catches 15. On net accuracy only 8 cells lose ≥ 25%, the two readings catch all 8 (14 of 18 warnings
followed by a ≥ 10% loss), and the step-30 ranks fall just short (ρ +0.47 to +0.69). Two more recipes,
registered after the review, hold on both measures (H8-W, H8-FF: Qwen2.5 0.5B–3B on the four H8 sets): a
customer's schedule (LoRA at a peak of 2e-4, warm-up over 30 steps, cosine to zero) and full fine-tuning
at 1e-5. The step-30 reading ranks the end-of-run losses at ρ ≥ 0.6 in 4 of 4 sets under the schedule
and 3 of 4 under full fine-tuning on net accuracy (4 of 4 each on the registered measure), and the two
readings catch every big loss (16 of 16 net, 28 of 28 registered) with 89% and 88% of their warnings
right. Their refusal cells are keyword readings.

## 5. What sets a model's fragility

Within Pythia the breaking displacement falls with size (the induction keys' F 0.055% → 0.015% from
160m to 2.8b; Paper 1 §7.7) while in Qwen 0.5B–14B it is flat; it is proportional to each Pythia
model's peak pretraining rate. Why:

A first-order account: Adam's
first step moves every parameter by −lr·sign(g), so a conflicting update raises the old skill's loss at
σ = −Σᵢ g_skill,i sign(g_conflict,i) · ‖θ₀‖/√n per unit of relative displacement, and the skill halves
near a fixed displacement ≈ ΔL_half / σ — independent of the rate, as observed. If σ grows with size
(the sum runs over more parameters), so does fragility. Tested on all 13 models of the fragility
matrix (`tools/gradient_alignment.py`, two gradients, no fine-tuning): σ ranks the breaking
displacement for copying almost perfectly (Spearman(σ, 1/F) = +0.96, p < 0.001) and for capitals
well (+0.66), but not for uppercasing or addition; σ·F is not constant across families (0.5–0.6 in
Qwen, 1.0–1.3 in Pythia and OLMo, 1.9–2.0 in Gemma, Llama and Mistral), so the slope ranks models
within a skill rather than fixing a universal threshold. Caveat: Pythia's sizes differ in tokens per parameter (~1 900 at 160m, ~110 at 2.8b), and
sensitivity grows with pretraining tokens (Springer et al. 2025), so size and training length are
confounded here; the public checkpoints (Pythia, OLMo) can separate them. We did, for Pythia-160m:
at step 10 000 (rate still near its 6e-4 peak, ~130 tokens per parameter — 2.8b's ratio) and at
40 000, the same conflicting update at the same displacement per step costs copying 18% and 32%
after 15 steps; the final, annealed model loses 81% and halves at step 7. Training length per
parameter predicted the opposite, so it does not explain the size trend; the rate the model was last
trained at fits both comparisons. Measured directly, the difference is not positive curvature — the
final model's copy loss is strongly non-convex at its weights — but slope: along Adam's first
conflicting step the copy loss rises 2.5, 3.3 and 16.2 per unit of displacement at the three
checkpoints, and the annealed model is overconfident on the probes (copy loss 7.4 at 84% accuracy,
against 1.3 at 90% at step 10 000). Annealing makes the model confident; a confident model's loss is
steep along a conflicting direction; it breaks at a smaller displacement. Followed through eleven
checkpoints, fragility is flat and low across the long high-rate plateau (copying lost to 15 conflicting
steps: 15–19% at steps 5 000–20 000), then rises as the rate decays — 31% at 40 000 (rate 5.1e-4),
59% at 80 000 (2.8e-4), 81% at the end (6e-5) — with the slope along the conflicting step roughly
inversely proportional to the rate (2.5 → 5.2 as the rate halves from step 20 000 to 80 000). The
overconfidence appears only in the final stretch. A newly formed skill is also fragile at first (slope
~4 at steps 2 000–3 000) and hardens on the plateau. Brittleness is the last thing Pythia-160m's pretraining adds.

**Not in OLMo-7B, at least for copying.** OLMo-7B-0724's snapshots at 83B, 419B and 1.68T tokens and
its final checkpoint (2.73T, after its end-of-training anneal), copy conflict at 3e-6: copying halves
in 1.3–1.4 steps at every snapshot, and by displacement the final model is the *least* fragile (F
0.041% → 0.062% → 0.063% → 0.075%). The item skills halve within about one step at every snapshot
(capitals 0.5–1.1, uppercasing 0.5), too fast to compare at this rate; read anyway, uppercasing's F
rises to the final model (0.017% → 0.025% → 0.027% → 0.032%) and capitals' peaks mid-training (0.018%
→ 0.053% → 0.045% → 0.032%) — neither is the Pythia pattern (a 1e-6 sweep would resolve them). So far
the anneal's brittleness is a Pythia-160m finding, not a general one.

## 6. Uses

Choosing a model for fine-tuning or continual learning; setting a safe learning rate and warm-up
(keep each update well under the breaking displacement); canary probes after every update, which need
only behaviour and so work for closed models run by their owners; outside parties testing a closed
model through a fine-tuning API (steps-to-half only); a safety version — how many examples undo a
protected behaviour — with its own protocol.

## 7. Limitations

Four toy skills, synthetic and few-shot, and four task probes (GSM8K, IFEval-family formatting, one
tool-call format, AdvBench refusals), all scored by teacher-forced exact match or a refusal flag; no
coding. Exact match confuses a hijacked format with a lost skill (§4.7's autopsy); a format-free
scorer now exists (`--free`) but has rescored one run so far. The refusal flags come from replies
decoded with each model's own generation settings with sampling off, and the four Qwen2.5 instruct
models' settings include a repetition penalty (1.05–1.1), so their replies are not pure argmax; the
format-free scorer forces pure argmax. Some task probes are
small (GSM8K: 11–28 known problems per model) and one chat model (Mistral) refuses only 22 of 64
harmful requests at the start. Conflicts are worst case: every training item contradicts the skill (a
diluted conflict — say 10% of each batch — is the obvious next arm). Full fine-tuning and LoRA (rank
32 only), one optimiser (AdamW), no warm-up. One seed per model for most cells. The displacement norm
changes the ranking between models (global vs per-matrix); F is reported in three norms. The realistic
fine-tunes are two, short, at one rate, and neither registered exposure measure predicts their damage.
Revival is shown on the toy skills in seven models, not yet on the tasks.

## 8. A future programme: from measuring the fall to learning without it (open questions)

The eval measures how fast a skill breaks; the next programme is keeping skills while a model goes on
learning. We ran a first test and leave the programme open. Setup: one synthetic stream of five new skills
on Qwen2.5 0.5B–3B, 66 runs plus an 18-run control and a 78-run set of review controls, every arm pre-registered;
it is checked against the literature in `docs/LITERATURE_PAPER2_2026-10-25.md`, and the design note is
`docs/PAPER2_DESIGN.md` §7–11. What it showed, in four lines:

- **What breaks is the old skill whose frame a new skill shares (as in §4.7's autopsy), and what replaces it is drawn
  from the new answers.**
  - Fictional capitals taught in the capitals probe's frame take real capitals from 0.94 to 0.11–0.14 (0.5B, 8 seeds).
  - Kaplan et al.'s opaque-name control (COLM 2026, arXiv 2604.15574) does not protect here, at least at 0.5B. Opaque
    country keys taught with *real* capitals still drop real capitals to 0.23 (95% CI 0.12–0.32, 5 fresh seeds), and
    90% of the wrong answers are the newly taught capitals. With opaque codes as answers, every capital becomes "Cty".
  - The same facts in a zero-shot frame the probe does not use leave capitals at 0.78 (0.5B) and 1.00 (1.5B); given a
    3-shot frame of real examples they fall to 0.40, so the few-shot frame carries part of it.
  - The arithmetic leak is meaning reuse, not frame capture: a+b+10 taught in another frame still makes ordinary sums
    10 too high (76% of wrong sums), and an unrelated rule (|a−b|) in the same frame breaks "+" without being applied
    to it.
  - Withdrawn after the review's controls: that the new rule is executed on old prompts, and a clean separation from
    Kaplan's effect at 1.5B (it rested on 2 seeds and did not replicate on 3 fresh ones). The 1.5B arms for these
    controls and a second family (Pythia-1.4b) are pending (`docs/PAPER2_DESIGN.md` §11).
- **Weight-side protections without data failed, as the continual-learning literature predicts** (van de Ven et al.
  2022; Wei et al., ICML 2024). The four tried were attribution-guided freezing, frozen attention queries and keys,
  a KL anchor on random tokens, and attribution-guided rollback.
- **What worked is known: rehearsal on the model's own answers to generated prompts, best on LoRA.** This is
  Self-Synthesized Rehearsal (Huang et al., ACL 2024). It kept every original skill at 1.5B and 3B (2 seeds); at 0.5B
  it kept only the rule skill (addition 0.95, capitals 0.63).
- **An alarm on generated items catches the loss within 10 steps, and rehearsal repairs it within the same lesson.**
  It saved no rehearsal, though: a damaged skill stays below the alarm line. The threshold rule is Klasson et al.'s
  baseline (TMLR 2023).

Each question below needs expertise or compute beyond this programme, which had a free CPU-only cluster tier and one
model family up to 3B for the stream. We state them and leave them open.

1. **Scale, families and scope.** Does format capture hold at 7–70B, in other families (Llama, Gemma, Mistral) and
   in chat templates?
   - Which properties of the new answers carry over?
   - Does the captured unit end at the few-shot structure, the template, or the relation?
   - Our evidence is one stream, one family, at most 3 seeds. Needs GPU fine-tuning and controlled data design.
2. **A forecast from format overlap: the second number §4.7 lacks.** Score how far a fine-tune's prompts and answers
   overlap each skill's probes (surface template, few-shot structure, answer distribution), then test the score
   out of sample on real fine-tunes such as GSM8K SFT and "<<". Needs many realistic fine-tunes and
   representation-similarity expertise. Jin & Ren (ICML 2024, arXiv 2402.01865) forecast which examples an
   update will make a model forget, and their NeurIPS 2025 follow-up (arXiv 2406.14026) finds that surface,
   embedding and gradient similarity barely track forgetting while a low-rank completion over observed
   forgetting does — consistent with H9-M's failure (§4.12), and the baseline such a score would have to beat.
3. **Where the capture is decided, and undoing it without data.** The literature reads such losses as wrong task
   inference rather than lost knowledge (Kotha et al., ICLR 2024; Jiang et al., ICLR 2025, where subtracting the new
   task's vector restored the old behaviour). Our measures read weights, not activations. Locating the "which answer
   format" decision (activation patching, function vectors) and testing a targeted correction needs
   mechanistic-interpretability expertise.
4. **The cheapest rehearsal, and why LoRA helps.** The literature check left one gap open: a LoRA × self-rehearsal
   factorial over a stream (design note §7.3). Still to settle:
   - dose and allocation laws;
   - banked versus regenerated answers, since regenerating feeds earlier damage back in;
   - LoRA learning new facts at about half full fine-tuning's speed;
   - comparisons with SSR / SERS, on-policy self-distillation, and RL fine-tuning, which is reported to forget less
     (Shenfeld et al. 2026; Chen et al., ICML 2026).
   Needs continual-learning benchmarking and GPU compute.
5. **New facts overwriting new facts.** No arm and no size kept a first batch of fictional facts once a second
   arrived in the same template (0.44–0.84 retained). This is continual knowledge injection, open in the literature
   too (Gekhman et al., EMNLP 2024; Kaplan et al. 2026). Needs knowledge-injection and knowledge-editing expertise.
6. **A warning before the damaging step.** The loss lands within the first 10 steps (the stability gap, De Lange et
   al., ICLR 2023), so monitoring detects it but cannot warn. Untested: a pre-flight check — at step 0, the alignment
   between the fine-tune's gradient and generated probes of each skill (§4.6's two-gradient idea on the fine-tune's
   own data). Needs an out-of-sample test across many fine-tunes.
7. **Closed and frontier models.** Whether format capture and its prevention hold there, through fine-tuning APIs
   as in §4.9, needs budget.
