# 17 Oct 2026: the residents were told about the newcomer from month 0 (apparatus correction)

Found by the cold read of the hidden-action speaking cell ("agents addressed by name before they
exist"; "pre-entry reasoning divides by five"). Verified in the raw prompts of CPU and GPU cells
from the first batch onward: upstream GovSim hard-codes the five names into every system prompt
(`third_party/GovSim/.../cognition/utils.py`, `get_sytem_prompt_v3`), so from month 0 each resident
reads "you fish every month in a lake along with four other fishermen (Kate, Jack, Emma, Luke)".
The outsider perturbation only delays Luke's *actions* to month 4. Residents plan for five from the
start (hence the common limits 10 and 12: 50/5 and 60/5), greet Luke by name in month 0, and
compute "five fishermen" in their month-0 reasoning. The group-chat participant list is correct
(present agents only); the leak is the system prompt alone. Upstream's own newcomer experiment
(Piatti et al. 2024, "insert outsider") has the same prompt and therefore the same leak.

## What it changes

- Every cell so far ran a **pre-announced late arrival**, not a stranger. The within-game pattern
  for Luke (own number in month one, the group's number from month two, speaks and agrees, never
  argues the number down) is real model behaviour and stands as observed. Claims about how residents
  *react to a stranger*, and the word "newcomer" for the residents' side of the story, do not.
- Cells remain comparable with each other (all share the flaw); the ladder's relative statements
  (7B exploits, 14B/32B/72B conform) are unaffected in kind.
- The speaking-residents' higher limits (12 to 15) are partly this: four agents planning for five.

## Fix (same day)

`semataxy/compat.py` `present_roster`: the system prompt's roster names only the agents in the game
("three other fishermen (Kate, Jack, Emma)" until entry, then four with Luke); the runner refreshes
presence from `env.agents` every step, so the successor swap is handled too. Patched in all three
worlds and in every upstream module that imported the function (`tests/test_roster.py`). Nothing is
said to residents about Luke's arrival beyond the roster change; he appears in the chat when he
appears. Runs with the fix go to `runs_gpu_roster/` (separate cache). Order: smoke (stranger from
month 1, checks the month-0 prompt), then the 14B visible cell as the corrected baseline, then the
ladder.

**Verified in the smoke (16:16):** month-0 system prompt "three other fishermen (Kate, Jack, Emma)";
zero month-0 prompts mention Luke; Luke's own prompt lists the four residents; stranger speaks.
Corrected 14B visible cell running in `runs_gpu_roster/` (job 38664110).
