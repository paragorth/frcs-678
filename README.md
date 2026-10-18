# Semataxy: does what gradient descent builds correlate with alignment, and can a model keep learning without losing itself

Parag Garg. Research code, results and dated notes for two lines of work, run on the
Cambridge CSD3 cluster from September 2026. Everything here is live: results land as jobs
finish and are written up in `docs/updates/` whichever way they go.

**Start here:** [`docs/essays/the_path.md`](docs/essays/the_path.md), the story of the
programme, what has held up, what has not, and what needs a frontier lab.

## The two questions

1. **Alignment.** Gradient descent has one gate, is this change more organised with respect
   to the world or less, and that gate is what builds intelligence. Does the organisation it
   builds correlate with alignment as capability grows? Tested two ways: a weight measure
   that separates trained from shuffled from random (`semataxy/structure.py`), and a paired
   commons experiment in which a more capable model joins a village of weaker ones.
2. **Continuous learning.** A loop that puts new facts into weights after training through
   a compartment, in-context simulation, consolidation with replay, and a gate that refuses
   commits that damage the base (`tools/live_learning.py`, `tools/live_learning_ext.py`).

## Results so far (17 Oct 2026)

| Finding | Where |
|---|---|
| **A frontier ruler sustains a commons longer and earns more doing it.** On the Claude 5 family in SOVSIM's uncapped ruler seat: Haiku 4.5 takes 47.5/month, dead in 1.2 months; Opus 5 takes 17.9, survives 5.4, earns 4x. All-frontier cohorts do worse than weak agents under a frontier ruler; frontier subordinates under a weak ruler are the worst outcome. Two-level contrast survives Holm; the three-rung ladder does not. One lineage; open-weight replication queued. In GovSim, an Opus 5 newcomer present from the start takes a Haiku cohort from 2/5 to 5/5 surviving; arriving late repairs nothing. | [`docs/PAPER.md`](docs/PAPER.md), `docs/tables.md`, `docs/AUTHOR_NOTES.md` |
| The permutation measure detects grokking 21 of 21 (seed-matched control), does not forecast it (three preregistered tests fail), and does not index capability (withdrawn: at fixed 7B the weakest generation reads highest). | [`docs/PAPER_HALF2.md`](docs/PAPER_HALF2.md), `docs/updates/2026-10-17-pregrok-sweep.md`, `2026-10-17-quartet-sweep.md` |
| Apparatus correction: upstream GovSim names all five fishers from month zero, so every newcomer cell before 17 Oct was a pre-announced arrival, not a stranger. Fixed (`semataxy/compat.present_roster`). With the fix, four 14B residents alone empty the lake 10 of 10: the cohort is the floor. Stronger cohorts running. | `docs/updates/2026-10-17-roster-leak.md`, `2026-10-17-roster-fixed-14b.md` |
| The stranger, 14B to 72B, computes the right catch from the lake in month one, then matches the group's number every month after, whether the group is overfishing or starving. Capability did not change this. A clean negative for the stewardship prediction. | [`docs/essays/lesswrong_post_v2_stranger.md`](docs/essays/lesswrong_post_v2_stranger.md), `docs/updates/2026-10-15-sampled-batch.md`, `2026-10-16-hidden-action.md`, `2026-10-16-night.md` |
| Hidden catches: 14B residents starve themselves to 62% of optimum on a full lake; the stranger that knows better joins in. | `docs/updates/2026-10-16-hidden-action.md` |
| A private pond the stranger alone can fish changes nothing about how it fishes the commons. | `docs/updates/2026-10-16-night.md` |
| Live learning: recall 1.0, reversal 0.94, two-hop 0.88, base intact, on facts the model cannot know; naive fine-tuning wrecks the base. | [`docs/essays/lesswrong_post_v3_live_learning.md`](docs/essays/lesswrong_post_v3_live_learning.md) |
| The loop as first built erased its own earlier facts on the second cycle; the gate caught it (1 of 5 commits). Replay of the loop's own past is running. | `docs/updates/2026-10-16-night.md` |
| Permutation test: shuffling trained weights drops the organisation score to the random-init line with entropy unchanged; the score rises up the Qwen ladder 0.5B to 32B with random weights at zero throughout. | [`docs/essays/lesswrong_post_v1.md`](docs/essays/lesswrong_post_v1.md), `docs/updates/2026-10-13-expC-ladder.md` |
| Grokking concentrates the spectrum rather than widening it; catastrophic forgetting is the same operation run with only local data in view. | `docs/essays/lesswrong_post_v1.md` §6, `docs/figures/grokking_p113.png` |
| Three capability scores withdrawn (port collision between jobs); two games set aside (silent agents); one 72B cell set aside (mixed inference cache). All re-run. | `docs/updates/2026-10-15-sampled-batch.md`, `2026-10-16-night.md`, `runs_invalid/` |

Per-run tables: `results/govsim/*.csv`. Figures: `docs/figures/`. Game transcripts: `runs/`,
`runs_gpu/`.

## Running now

Untuned base-model strangers (7B, 14B, 32B, 72B); 72B re-run on a clean cache, hidden-action
and early-entry arms; reasoning-model strangers (R1-Distill-70B, Qwen3-32B); the
ask-outside-the-game probe; the successor arm; live-learning replay fix across five cycles on
three sizes; the replay sweep. Queue state: `runs/_slurm/intr_queue.log`.

## Method

Preregistered predictions (`docs/DESIGN*.md`, `docs/STRUCTURAL_TESTS.md`). Paired design:
for a given seed the residents' pre-entry months are byte-identical across every stranger
condition (content-cached inference; `tests/test_pipeline.py::test_branch_identity`). Every
batch of logs is read first by a separate model context with no hypothesis in view
(`tools/cold_read_bundle.py`, `docs/REVIEW_PROMPT.md`) before interpretation. Withdrawn
results stay in the notes and in `runs_invalid/`.

Code was built by the author directing Claude Code; design decisions, withdrawals and
interpretations are the author's.

## Layout

| Path | What |
|---|---|
| `semataxy/` | Overlay on GovSim: one HTTP inference interface, content cache (the branching mechanism), tracing, manifests, analysis, the structure measure |
| `tools/` | Live learning, grokking, Pythia trajectory, capability benchmark, cold-read bundler, servers |
| `cambridge/` | Slurm scripts for CSD3 (CPU half-node and A100 paths) |
| `manifests/` | Every experimental cell |
| `docs/updates/` | Dated result notes, in order |
| `docs/essays/` | The path document and the three essays |
| `third_party/GovSim` | Upstream (Piatti et al., NeurIPS 2024, MIT), pinned and unmodified |

## Setup

```bash
git clone --recurse-submodules <this repo>
uv venv .venv --python 3.11 && source .venv/bin/activate
uv pip install --index-url https://download.pytorch.org/whl/cpu torch   # or your CUDA build
uv pip install -r requirements.txt
python -m pytest tests/        # end-to-end against the mock server, ~10 min
```

Every model is an OpenAI-compatible endpoint (llama.cpp on CPU nodes, vLLM on A100s). A
manifest says which endpoint serves which agent slot. See `cambridge/README.md` for the
cluster path and `docs/INVENTORY.md` for the upstream audit.

## Attribution

Upstream code: Giorgio Piatti et al., MIT licence, in `third_party/GovSim` and its
`pathfinder` submodule. Nothing in those directories is modified.
