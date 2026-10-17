# Prompt for a fresh session: the Research story on surgeonledai.co.uk

Paste everything below this line into a new session.

---

Build `/research` on the live site surgeonledai.co.uk as a single scrolling story, with one
page per essay behind it. The site source is at
`/Users/pg/Documents/Codex/2026-08-24/movi/surgeon-led-ai` (Next.js 16 via vinext, Tailwind 4,
deployed to Cloudflare through the OpenAI Codex site tooling; `.openai/hosting.json` holds the
project id). Read `app/page.tsx`, `app/layout.tsx`, `app/globals.css` and
`app/ig-companion/page.tsx` first and match the type, palette and spacing so the section reads
as the same hand. The research content lives in the repo `/Users/pg/frcs-678`
(branch `claude/semataxy-govsim`). The story's text is `docs/essays/the_path.md`; use its
sections, in its order, in its words. Do not rewrite the author's sentences; cut for length
where a chapter is long, and link to the full essay.

## Who reads it

AI safety researchers deciding whether to shortlist Parag Garg for a fellowship, and later,
anyone he sends the link to. They give it ninety seconds unless it earns more. It earns more
by being a story with real results in it, including the ones that went against him.

## The shape: a story, scrolled

One long page. Each chapter is a short passage from the path document, a scene beside or
beneath it, and the links that chapter earns. Scenes are small, quiet, and made from the real
data in the repo; every number traces to a file, with the path in a code comment. Motion is
scroll-triggered and runs once; nothing loops, nothing bounces, nothing is decorative. Without
JavaScript every scene is a static image and the page still reads top to bottom.

Vocabulary on the page: plain. The author's coined terms (Semataxy, Xenataxy, Zenice) appear
only inside the full essays, not in chapter text, captions or navigation. Where the path
document uses one, substitute "organisation in the weights" or link the essay.

## Chapters and scenes

1. **The problem.** Ten years believing superintelligence ends us. Scene: none. Just the
   passage, large type, and the one line: "I am a surgeon, not a computer scientist, and I had
   no standing to say anything about it. I also could not leave it alone."

2. **The Entropy Machine.** Three hundred iterations of an agent whose deepest objective was
   to increase its own organisation, failing in every way such things fail. Scene: a slow
   counter of the failure modes from the text, one at a time on scroll: pathological order,
   over-compression, novelty as noise, death by committee, arbitration, reflexes, a self. Text
   only, typographic, no illustration.

3. **The inversion.** "The architecture I had ended up with was gradient descent." Scene:
   **the permutation test, live.** A real weight-matrix slice drawn as a heat map (from the
   repo's structure results if a raw matrix is retrievable; otherwise a low-rank-plus-noise
   matrix generated in code with a comment saying so) with one control, "shuffle". On scroll
   into view, cells move to their permuted positions; beside it two figures: entropy, which
   does not change, and organisation in octaves above noise, which falls to zero. Caption:
   "Everything training builds is in the part a shuffle destroys." Links: the theory essay,
   the Zenodo DOI once minted.

4. **Measuring it.** The ladder. Scene: the organisation score up the Qwen ladder, 0.5B to
   32B, as a rising row of bars with the random-weight control flat at zero beneath, from
   `results/structure/ladder_summary.csv` and `docs/figures/structure_ladder.png`. Caption from
   the text: the rise is not matrix size. Then one honest line: the between-layer measures
   saturate early and see "trained", not "how well". Link: `docs/updates/2026-10-13-expC-ladder.md`.

5. **Consciousness as deployment.** The driver and the child in the road. Scene: a single
   line, flat, that at one point lifts into a brief raised episode and settles back; the
   "opening transfers authority, maintenance turns a trigger into an episode" idea as one
   drawn gesture. Then the ignition retractions, in text: three findings died the same way,
   the preregistered criterion, the null reported as a null. Links: the five Zenodo papers
   (10.5281/zenodo.21984327, 21984450, 21984521, 21984601, 21984627), PhilArchive when there.

6. **The write path that did not exist.** Built on Anthropic's global-workspace paper. Scene:
   two bars side by side, "state match" rising to cosine 0.91 and "skill installed" staying at
   0.0, then ordinary fine-tuning: cosine 0.27 and install 92%. Numbers from
   `/Users/pg/jlens-experiment/RESULT-phase3-jspace.md` (copy the figures into the site; cite
   the file). Caption: "The objective selected against the outcome." This chapter is the
   author's favourite negative; give it room.

7. **Grokking, forgetting, and the loop.** Scene A: the grokking spectrum concentrating,
   drawn from `docs/figures/grokking_p113.png` (use the figure). Scene B: the loop as four
   stations on one line, compartment, simulate, consolidate, gate, with the gate the only place
   the line can stop. Scene C: the one table, as a small animated reveal: naive fine-tuning
   held-out loss 4.20 to 8.71 against the loop's base intact, recall 1.0, reversal 0.94. Then
   the honest line, large: "On the second cycle it forgot its own first. The gate caught it."
   Links: the live-learning essay, `docs/updates/2026-10-16-night.md`.

8. **The stranger.** The lake, the village, the newcomer. Scene: four small multiples, 7B,
   14B, 32B, 72B. In each, the sustainable level drawn flat at 10, the group's number as a
   step, and the stranger's catch as a line that sits on 10 in month one and snaps to the
   group's step from month two. Series from `results/govsim/*_runs.csv` if per-month data is
   present; otherwise drawn from the numbers in the stranger essay and labelled schematic.
   The 7B panel is the exception: its line goes above everyone. Caption: "It knows the answer
   in month one and agrees with the group from month two. More capability did not change
   this." Then the hidden-action and pond lines from the text. Then, plainly: a clean negative
   for the prediction. Links: the stranger essay, the three update notes, per-run CSVs,
   transcripts.

9. **How I work.** Preregister, pair, read cold, withdraw in writing, report the complication
   rate. Scene: a short list of the withdrawn results with dates, each linking to the note
   that withdrew it. This is the chapter that makes the negatives credible; do not shorten it.

10. **Clinical AI, built from friction.** One paragraph from the path document. Links:
    Litmus (litmus3.surgeonledai.co.uk), Ai4Qi (ai4qi.com), the app catalogue on the home
    page.

11. **What is running now, and the wall.** The list from the path document, then the
    frontier-lab paragraph. Scene: the open-weight ladder drawn with its top rung at 72B/235B
    and a gap above it, unlabelled. Links: the repo's `runs/_slurm/intr_queue.log` path in
    text, the README.

12. **Reading.** Rows, not cards: the path document, the three essays, the papers with DOIs,
    the repository. Title, one plain line, status (Fixed with DOI, or Live with last-updated
    date), link. Then the About block: Parag Garg, consultant orthopaedic foot and ankle
    surgeon, NHS; MSt in Healthcare Data Science, Cambridge; research on AI alignment and
    continuous learning; repo, Zenodo, contact (ask Parag which email; do not invent one).

## Essay pages

`/research/<slug>` for each of the four essays, rendering the markdown with tables, headings,
blockquotes and the Semataxy essay's image (`docs/essays/structure_qwen05b.png`, copy into
`public/research/`). Keep the author line and the AI-assistance note at the top exactly as
written. Under the title: a two-line plain summary and a Status line (Fixed with DOI, or Live
with the file's git date). Negatives say so in the summary. Link back to the story.

Add a "Research" link to the navigation and one panel on the home page, in the existing panel
style, pointing at `/research`.

## Scenes: how to build them

Inline SVG or canvas, written by hand, no charting library, no animation library. Scroll
triggering with IntersectionObserver; each scene plays once. Every scene has a static
fallback that is what renders with JavaScript off. Respect `prefers-reduced-motion`: static
only. Mobile: scenes stack under their passage; the four small multiples become a 2x2 grid.
If a figure's data is not in the repo in usable form, draw it from the numbers in the essay
text and label it "schematic, numbers from the essay"; never invent a number.

## Content sync

Copy the four essays, the path document and the figures from `/Users/pg/frcs-678/docs/`
into the site repo under `content/research/`, and add `scripts/sync-research.sh` that
re-copies them. Updating the site is: run the script, commit, deploy. Text edits happen in
the research repo, never in the site repo.

## Rules

- Match the existing site; the section should read as the same hand with a different subject.
  Dark glass surfaces are welcome if the palette allows. No stock illustration, no icons as
  decoration, no gradients as decoration, no particles, no hero photographs, nothing that
  looks like a game.
- Taste: quiet, exact, confident. The story carries it; the scenes confirm it.
- No analytics, no tracking, no third-party scripts, no external fonts beyond what the site
  already loads.
- Dates come from the source files' git history; never invent one.
- Do not deploy. Build locally (`pnpm build`), open `/research` and each essay page in the
  preview, screenshot the story at three scroll positions and one essay page, then stop and
  report. Parag deploys from his Codex session.
- Do not copy anything from the research repo's `docs/applications/` folder.
