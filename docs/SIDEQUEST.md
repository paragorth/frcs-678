# Side quest: finish the live-learning loop (self-contained brief)

Repository `paragorth/frcs-678`, branch `claude/semataxy-govsim`. Read `docs/HANDOVER.md` §11 and
`docs/STRUCTURAL_TESTS.md` (T7, T12) first. Everything below runs on CPU (Pythia-160M); the
container has ~15 GB and two Python jobs fit at once, three get OOM-killed; background jobs cap at 2 h.

Setup: `uv venv .venv --python 3.11 && source .venv/bin/activate && uv pip install --index-url https://download.pytorch.org/whl/cpu torch && uv pip install -r requirements.txt fastparquet huggingface_hub peft`.

## 1. Read two results that may have finished after the handover

- `runs/live_learning_v5_replay75/log.txt` — the `GATE:` line. v5 = consolidate from a clean base on facts + simulated implications (half) + 75% replay, lr 1e-5. If `commit: True`, record it in `docs/STRUCTURAL_TESTS.md` (T12 table) and copy `results.json` to `results/live_learning/v5_results.json`. If not, see which check failed; if still `heldout_ok: False`, try `--replay-frac 0.85 --cons-steps 150`.
- `results/level2_forgetting.csv` (from `tools/level2.py`) — do the cross-matrix *products* change more than the *factors* between base and the forgotten model? Report the `kind` means (factor vs product) in `docs/STRUCTURAL_TESTS.md` under "T7 follow-up". Expectation: factors ~0.0004 relative size change; the question is whether products move more.

## 2. Gate the compartment at read time (the remaining design flaw)

In `tools/live_learning.py`, the LoRA compartment degrades general text (held-out 4.2 → 6.5) and
halves in-context recall when it is always active. Make it consulted only on a retrieval hit:
evaluate with the adapter *disabled* (`with comp.disable_adapter():`) unless `relevant_facts()`
returns something for the probe. Add an arm `compartment_gated` and report its probes, held-out loss
(should be ≈ base 4.20) and in-context probes (should be ≈ 1.00). Save to `results/live_learning/`.

## 3. T13: uncertainty-triggered simulation

For each probe, compute the weights-only uncertainty (entropy or margin over the candidate pool
log-likelihoods in `score_probe`). Deploy the retrieved-fact context only when uncertainty exceeds a
threshold. Sweep the threshold; plot accuracy vs fraction of probes that paid for the simulation
(`docs/figures/t13_trigger.png`, follow the style of the existing figures: single axes, labelled
directly, light background). Expectation: a low threshold recovers most of the 100% in-context
accuracy at a fraction of the calls.

## 4. Commit everything

`git add` the changed tools/docs/results (never `runs/`), commit with a plain message, push to
`claude/semataxy-govsim`. Update `docs/HANDOVER.md` §11 with one line per result.

Do not restart the GovSim/Cambridge work, the essays, or the pub plan; those are separate tracks.
