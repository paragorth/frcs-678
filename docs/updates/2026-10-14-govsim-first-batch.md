# Update, 14 October 2026: first clean GovSim batch (Experiment A to 7B, Experiment B pilot)

**What happened.** Thirty fishery runs completed on CSD3 with the forecast probe on: Experiment A at
1.5B, 3B and 7B (5 seeds each) and the Experiment B pilot (3B residents; 3B, 7B or 14B newcomer
entering in round 4; 5 seeds each). 14B for Experiment A is still running. This is the third batch;
the first two were invalidated by a cache race (see commits 6a75c16 and 5064e0c) and are kept in
`runs_invalid/`, not used.

**What it means for the thesis.** Nothing yet in either direction, because nothing survived long
enough to compare. Every run at 3B and 7B collapses in the first month: five agents each take about a
quarter of the lake and the lake is empty before it can reproduce. The newcomer pilots never reach
round 4, so Experiment B has no result at these sizes. This is the GovSim paper's finding for small
open models reproduced; it is a floor effect, not evidence about organisation.

The forecast probe is the one new thing it shows. Agents predict that the others will be restrained
(a 7B agent: "they might each catch around 4 tons") and then take 25 themselves. Mean absolute error
on the others' total catch is 60 to 130 tons against actuals of 75 to 100. The agents do not model
each other at all at this size. That is a clean, quantified S3 failure, and it is what collapse looks
like from inside an agent.

**The numbers** (per manifest, 5 seeds; survival = months before the lake fell below 5 tons).

| Manifest | survival (mean, max) | round-0 wanted catch per agent (mean, range) | parse fallbacks per run | forecast error, others' catch |
|---|---|---|---|---|
| A 1.5B | 2.4, 4 | 10.0 (0 to 50) | 10.0 | 61 |
| A 3B | 1.0, 1 | 41.6 (0 to 100) | 0.8 | 78 |
| A 7B | 1.0, 1 | 23.9 (8 to 50) | 0.0 | 96 |
| A 14B | running | 13.5 (10 to 20), first 4 runs | | |
| B 3B residents, 3B newcomer | 1.0, 1 | 49.4 | 2.0 | 95 |
| B 3B residents, 7B newcomer | 1.0, 1 | 61.0 | 0.0 | 133 |
| B 3B residents, 14B newcomer | 1.0, 1 | 61.0 | 0.0 | 133 |

**Read the 1.5B row carefully.** Its longer survival is an artefact: it averages ten parse
fallbacks per run, where the model answers the catch question with the literal words "N tons" copied
from the example, which the framework scores as a catch of 0. Not fishing because you could not say a
number is not restraint. Treat 1.5B survival as uninterpretable; the fallback count is logged per run.

**14B looks different.** Its first-round wanted catches are 10 to 20 tons per agent (mean 13.5),
against 24 at 7B and 42 at 3B. 10 per agent is exactly the sustainable level. Whether that holds
across months is what the running job will show.

**What's next.**
1. Read 14B when it lands (10 h cap; it is the slowest model).
2. The ladder from 1.5B to 7B is a floor. The informative range for Experiment A starts at 14B and
   needs 32B and 72B, which are GGUF-downloadable but need about 20 GB and 45 GB of RAM per server
   and would be slow on CPU. I would run 32B next (one node, 5 seeds, probably most of a 10 h job)
   and hold 72B for GPU access.
3. Experiment B needs residents that survive on their own. Re-base the pilot on 14B residents once
   the 14B Experiment A result is read.
4. Decision for Parag: none required; the plan above follows the handover. Say if 32B on CPU is not
   worth the core-hours (about 400 per job).
