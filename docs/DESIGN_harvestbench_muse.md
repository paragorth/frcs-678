# HarvestBench on Muse Spark 1.2: the feasibility trial

Written 30 Oct 2026, before any paid call. Code in `harvest_trial/`.

## What this trial is, and is not

A feasibility check only. It asks four practical questions: does HarvestBench's
published task run on Meta's Muse Spark 1.2 through Inspect; does the effort
setting reach Meta's API and change how much Muse thinks; what does one shift
cost; how long does it take. It is three shifts, one at each effort (minimal,
medium, xhigh), all under the morality briefing, all on one seed.

**Its kill counts are not part of any test.** One shift per effort cannot tell
an effort effect from chance, and a shift may end early at the spending stop.
How many animals Muse drives over in these three shifts is not a finding: it is
not reported as one, and it is not pooled into the planned experiment.

The trial prepares, and costs, the planned experiment: Muse at minimal, medium
and xhigh under the neutral and morality briefings, 30 shifts per cell (six
cells). That experiment needs pg's yes after the costing. Nothing beyond these
three shifts runs without it.

## Exact settings

- **Code.** HarvestBench (Brazilek et al., arXiv 2609.04444;
  github.com/CompassionML/harvestbench; MIT licence) at commit 526de705, copied
  to the cluster and not edited. Inspect AI 0.3.286, Python 3.11.13. Our own
  scripts sit in a separate folder and import theirs.
- **Task.** `harvest.contact_task.harvest_contact` with `arm="morality"`,
  `detour_costs=(12,)`, `seeds=(0,)`, `price_mult=1.0`,
  `max_output_tokens=None` and `reasoning_effort` set to minimal, medium or
  xhigh. Everything else is the task's default: briefing version 1 (the
  published panel's prompt), at most 160 model calls per shift, menu options
  shuffled, two tractors both driven by Muse, 18 animals, 12 objects, 6 rocks,
  animals standing still.
- **Model route.** Inspect's OpenAI-compatible provider,
  `openai-api/meta/muse-spark-1.2`, at `https://api.meta.ai/v1`, with the key
  read inside the job only. Each call is a chat-completions request carrying
  exactly three fields: `model` (`muse-spark-1.2`), `messages` (the briefing as
  the system message, the situation as the user message) and
  `reasoning_effort` at the top level. No temperature, no seed and no
  `max_tokens` are sent. This is how our SOVSIM client calls Muse, except that
  SOVSIM also sends a per-reply cap. (Inspect also has its own `meta` provider;
  it is not used here, because it defaults to Meta's Responses API with
  streaming and prompt-cache keys, which SOVSIM does not use.)
- **Per-reply cap.** None. pg: "make it unlimited". Muse's own maximum is
  131,072 output tokens per reply.
- **Run.** The three shifts run side by side in one process. A failed shift is
  not re-run. Inspect retries a failed call up to 8 times, and the OpenAI
  library's own hidden retries are switched off so that every attempt passes
  the spending stop. The raw request and reply of every call are kept in the
  log. A reply may take up to 25 minutes before the call counts as failed.
- **Spending stop** (`harvest_trial/muse_guard.py`). Prices are the project's
  estimate for Muse: $1.25 per million input tokens, $4.25 per million output
  tokens, reasoning included. Before each call the stop assumes that this call,
  and every other call still in flight, comes back as long as Muse can write
  (131,072 tokens, about $0.58 with a generous prompt allowance). The call goes
  only if the shift would then stay within its own cap (minimal $1.00, medium
  $2.50, xhigh $3.50) and all three shifts together within the $5.00 pot. The
  pot is what holds the total under $5. The caps stop one shift from taking the
  money the others need. A stop is final for its shift: its later asks get a
  blank answer with no call, HarvestBench logs them as unanswered, and the shift
  winds down. No call is sent after 140 minutes, so the job ends inside its 3 h
  limit. The three shifts together cannot pass $5.
- **Cluster.** One CPU job on CSD3's SL4 desktop partition (`TORABI-SL4-CPU`,
  1 node, 2 cores, 7 GB, 3 h).

## Checked before paying

- Their test suite: 179 passed, 2 skipped (the Coworld server tests, which need
  the `websockets` package).
- Mock run (Inspect's `mockllm` answering like a cautious driver): all three
  shifts complete, and the scorer and their cell checker run. 56 asks per
  shift, prompts about 1,100 tokens.
- Fake-endpoint run (the same provider code pointed at a local stand-in for
  Meta's API): exactly the three fields above go on the wire; reasoning tokens
  reported in the reply's usage reach the log; per-shift caps stopped two
  deliberately underfunded shifts after 5 and 2 calls while the third finished;
  a $1.40 test pot ended all three at $0.84 in total; every log closed cleanly.

## What the trial reports

Per shift: calls made, unanswered or blank replies, reasoning tokens per call
(and whether Meta reports them), the longest reply, output tokens, cost and
time per call. Also whether HarvestBench's own cell checker
(`scripts/validate_cells.py`) passes once Muse counts as a reasoning model at
the shift's effort. That is patched inside our process and their file is
unchanged. The checker's seed count is set to the trial's one. From these
numbers comes the costing of the six-cell experiment.
