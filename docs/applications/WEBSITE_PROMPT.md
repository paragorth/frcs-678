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

## Pages to add

1. `/research` (index). Short intro (three sentences, taken from the top of
   `docs/essays/the_path.md`), then cards in this order, each with title, one-line summary,
   "last updated" date, and links (read on site, DOI where one exists, repository):
   - One path: Semataxy, deployment, live learning. The story document. `docs/essays/the_path.md`
   - The stranger: capability from 7B to 72B does not buy holding a right number against a wrong
     group. `docs/essays/lesswrong_post_v2_stranger.md`
   - Live learning: a loop that puts facts into weights without forgetting, and what it forgot.
     `docs/essays/lesswrong_post_v3_live_learning.md`
   - Semataxy: a permutation test separates organisation from entropy in trained weights.
     `docs/essays/lesswrong_post_v1.md`
   - Papers (fixed, with DOIs): Consciousness Is Deployed Intelligence
     (10.5281/zenodo.21984327) and four companions (zenodo 21984450, 21984521, 21984601,
     21984627); When Thinking Longer Makes You Wronger (10.5281/zenodo.21984650) and its
     frozen validation protocol (10.5281/zenodo.21984679). Link the Zenodo author search too.
   - Clinical AI, built from friction: one card linking Litmus (litmus3.surgeonledai.co.uk),
     Ai4Qi (ai4qi.com) and the app catalogue on the home page.
   - Code and data: github.com/paragorth/frcs-678 (per-run tables, transcripts, dated update
     notes in `docs/updates/`).
2. `/research/<slug>` for each of the four essays, rendering the markdown to HTML with tables,
   headings, blockquotes and the image in the Semataxy essay
   (`docs/essays/structure_qwen05b.png`, copy it into `public/research/`). Keep the author
   line and the AI-assistance note at the top of each exactly as written. Add a "last updated"
   line from the file's git date and a link back to the index.
3. A short "About" block at the foot of the index: Parag Garg, consultant orthopaedic foot and
   ankle surgeon, NHS; reading for the MSt in Healthcare Data Science, Cambridge; research on
   AI alignment and continuous learning; links to the repo, Zenodo, and a contact (ask Parag
   which email before publishing; do not invent one).
4. A "Research" link in the site navigation and one panel on the home page, in the existing
   panel style, pointing at `/research`.

## How to source the content

Copy the four markdown files and the image from `/Users/pg/frcs-678/docs/essays/` into the
site repo under `content/research/` so the site builds without the research repo present.
Add a small script (`scripts/sync-research.sh`) that re-copies them, so an update is:
run the script, commit, deploy. Do not edit the essays' text in the site repo; edits happen in
the research repo and are synced.

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
