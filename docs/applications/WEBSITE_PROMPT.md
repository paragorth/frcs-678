# Prompt for a fresh session: add a Research section to surgeonledai.co.uk

Paste everything below this line into a new session.

---

Add a Research section to the live site surgeonledai.co.uk. The site source is at
`/Users/pg/Documents/Codex/2026-08-24/movi/surgeon-led-ai` (Next.js 16 via vinext, Tailwind 4,
deployed to Cloudflare through the OpenAI Codex site tooling; `.openai/hosting.json` holds the
project id; GitHub remote is badlydivided/surgeon-led-ai). Read `app/page.tsx`, `app/layout.tsx`,
`app/globals.css` and `app/ig-companion/page.tsx` first and match their style exactly: same fonts,
palette, spacing, the numbered-card pattern used for the app catalogue, and the governance-tier
voice. No new design language.

## What the section is for

Parag Garg is applying to AI safety research fellowships (Anthropic Fellows, deadline 18 Oct 2026;
MATS Residency; Pivotal; OpenAI Residency). Reviewers will open `surgeonledai.co.uk/research`
from the application and need to see, in ten seconds, the shape of the work and where to read
more. The content already exists as markdown in the research repo
`/Users/pg/frcs-678` (branch `claude/semataxy-govsim`). Render it; do not rewrite it.
The research is live: results land weekly and the pages must be re-deployable from a commit.

## Design: this is not a list of cards

The research section should feel like walking into the work, not reading a bibliography.
Match the site's type, palette and spacing so it belongs, but give the section its own
identity. Dark, glass-like surfaces are fine if the site's palette allows; no stock
illustrations, no icons-for-the-sake-of-it, nothing that looks like a game.

**Hero: the permutation test, live.** The whole programme rests on one move: shuffle the
entries of a trained weight matrix and the network dies while its entropy is unchanged. Show
it. A small real weight matrix (take a 64x64 slice from `results/structure/l2_qwen2.5-0.5b.csv`
or render one from the repo's data; if no raw matrix is available, generate a trained-looking
matrix with low-rank plus noise structure and say so in a code comment) drawn as a heat map,
with a single control: "shuffle". On shuffle the cells animate to their permuted positions and
two numbers beside it change: "entropy" stays identical, "organisation (octaves above noise)"
drops from its value to 0. One sentence under it: "Everything training builds is in the part a
shuffle destroys." Pure SVG or canvas, no library, runs without JavaScript as a static before
and after image, animates with it. This is the one piece of motion on the page.

**Second scene: the stranger.** Below the hero, the stranger result as a small inline figure:
the stranger's monthly catch across the game as a line, with the lake's sustainable level
drawn flat and the group's number drawn as a step. Month one on the sustainable line; month
two onward snapping to the group's step. Four small multiples, 7B, 14B, 32B, 72B, so the eye
sees that from 14B up the lines are identical. Data from `results/govsim/*_runs.csv`; if
per-month series are not in the CSVs, draw the figure from the numbers in the stranger essay
and label it as a schematic. One line under it, in the author's words: "It knows the answer in
month one and agrees with the group from month two."

**Third scene: the loop.** Compartment, simulate, consolidate, gate, as four stations on one
horizontal line with the gate drawn as the only place the line can stop. Static. Next to it
the one number that matters, held-out loss under naive fine-tuning (4.20 to 8.71) against the
loop (base intact), and the honest line: "On the second cycle it forgot its own first. The gate
caught it."

Then, and only then, the reading list: the path document first, the three essays, the fixed
papers with DOIs, the clinical tools, the repository. Each as a row, not a card: title, one
plain line, status (Fixed with DOI, or Live with last-updated date), link. Rows, dates, done.

Every number on the page must trace to a file in the repo; put the path in a code comment
next to it. No made-up data in the figures.

**Taste rules.** No coined terms outside the essays. No gradients-as-decoration, no particles,
no hero photographs. The motion budget is the hero's shuffle and nothing else. If the
existing site is light, this section may be dark; it should read as the same hand with a
different subject. Mobile: the hero becomes a static before/after pair side by side.

## Pages to add

1. `/research` (index), as designed above, ending with the "About" block: Parag Garg,
   consultant orthopaedic foot and ankle surgeon, NHS; reading for the MSt in Healthcare Data
   Science, Cambridge; research on AI alignment and continuous learning; links to the repo and
   Zenodo; contact (ask Parag which email; do not invent one).
2. `/research/<slug>` for each of the four essays, rendering the markdown to HTML with tables,
   headings, blockquotes and the image in the Semataxy essay
   (`docs/essays/structure_qwen05b.png`, copy into `public/research/`). Keep the author line
   and the AI-assistance note at the top exactly as written. Under the title: a two-line plain
   summary and a Status line (Fixed with DOI, or Live with the file's git date). Negatives say
   so in the summary. Link back to the index.
3. A "Research" link in the site navigation and one panel on the home page, in the existing
   panel style, pointing at `/research`.

## How to source the content

Copy the four markdown files and the image from `/Users/pg/frcs-678/docs/essays/` into the
site repo under `content/research/` so the site builds without the research repo present.
Add a small script (`scripts/sync-research.sh`) that re-copies them, so an update is:
run the script, commit, deploy. Do not edit the essays' text in the site repo; edits happen in
the research repo and are synced.

## What a reviewer must see in ten seconds (design the index around this)

The reader is an AI safety researcher deciding whether to shortlist. The index must answer,
above the fold and without scrolling on a laptop:
- What the two questions are (one line each, plain words, no coined terms).
- That there are results, including negatives, with dates. Put a short results strip near the
  top: three findings, one line each, each linking to its essay. Take the lines from the
  "Results so far" table in the research repo's `README.md`. Lead with the stranger result,
  which went against the author's prediction, and say so.
- That the method catches its own mistakes: one line on preregistration, paired design,
  cold reads, and withdrawn results, linking to `docs/updates/` in the repo.
- Where the code is.

Vocabulary: the essays use the author's coined terms (Semataxy, Xenataxy, Zenice). The index,
cards, summaries and navigation must not. Say "organisation in the weights", "the alignment
question", "continuous learning". Coined terms appear only inside the essays themselves.

Each essay page carries, directly under the title, a two-line plain-language summary and a
"Status" line: Fixed (with DOI) or Live (last updated date, "results are added as runs land").
Essays that report a negative say so in the summary line.

## Rules

- Match the existing site exactly; the reviewer should not be able to tell the section was
  added later.
- No analytics, no tracking, no third-party scripts.
- Everything must render with JavaScript disabled (static content).
- Mobile first; tables scroll inside their container.
- Do not deploy. Build locally (`pnpm build`), check `/research` and each essay page in the
  preview, screenshot them, then stop and report. Parag deploys from his Codex session.
- Do not commit the research repo's `docs/applications/` folder or anything from it to the
  site.
- Dates: use the actual git dates of the source files; never invent one.
