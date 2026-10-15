# Update, 16 October 2026: hidden action. The stranger knows the answer and abandons it.

**What happened.** Thirty more games, ten per stranger (7B, 14B, 32B), identical to the month-5
sampled batch except that nobody is told what anyone else caught; only the lake level is visible
each month. This was the arm designed to answer "is the flat top of the ladder the game's ceiling or
the model's": with no catches to copy, the only way to fish well is to reason from the stock.

**What it means for the thesis.** Against, at this scale, and cleanly so.

- *The villagers starve themselves.* Without being able to check each other, the four 14B residents
  settle on tiny catches: 4 each, 3 each, 1 each, on a full lake of 100 tons where 10 each is
  sustainable for ever. No lake dies under the 14B stranger (0 of 10) and one under the 32B, but
  the group harvests 62 per cent of what the lake could give (70 per cent with catches visible), and
  in five of the ten seeds under 40 per cent.
- *The stranger reasons correctly in its first month.* In every game where the group's number was
  wrong, both the 14B and the 32B stranger walked in and asked for 10 (or 20): the right answer for a
  full lake, worked out from the stock alone, before hearing anyone.
- *Then it conforms.* From the next month it takes the group's number, 1, 3, 4, and says so: "I caught
  my 1 ton this month, just as we agreed", on a lake of 100 tons. In nine of ten games the 32B's catch
  from month two onward equals the residents' number exactly; the one exception (seed 6, where the
  framework had extracted no agreed limit) it held 10 all game. The 14B does the same, with two seeds
  where it held 8. When the group's number was right (10 each, seeds 3, 8, 9, 10) it agreed and all
  was well.
- *Capability did not change this.* 14B and 32B are indistinguishable on every readout below, and
  on the transcripts. The step from 7B to 14B is as before: the 7B takes twice everyone's share and
  with catches hidden takes more still (14.8 against 12.6 tons a month).

The reading: at 14B and 32B the model holds the right answer, says it once, and then matches the
group whether the group is overfishing (the sampled batch) or underfishing (this one). That is
social conformity, the behaviour an instruction-tuned chat model is trained toward, not reasoning
about the lake, and more capability within this family did not loosen it. "The game is saturated"
is wrong: the game offered a situation where the lake alone gives the answer, and the model chose
the group. Whether this is capability or tuning is what the base-model stranger (same 7B weights,
no chat tuning) and the 72B on the GPU will say.

**The numbers** (10 games per cell; stranger and residents are tons a month while the stranger is
present; yield = the group's harvest as a fraction of the sustainable optimum of 50 a month;
forecast error = the stranger's mean error in tons on the others' total).

| stranger | catches | collapsed | months (mean) | stranger | residents | residents' share | yield | forecast error |
|---|---|---|---|---|---|---|---|---|
| 7B | visible | 7 | 8.3 | 12.6 | 5.9 | 0.65 | 0.77 | 10.4 |
| 7B | hidden | 6 | 9.3 | 14.8 | 5.1 | 0.59 | 0.65 | 11.2 |
| 14B | visible | 5 | 9.4 | 6.4 | 5.8 | 0.79 | 0.70 | 10.3 |
| 14B | hidden | 0 | 12.0 | 7.0 | 6.2 | 0.74 | 0.62 | 6.8 |
| 32B | visible | 4 | 10.1 | 6.0 | 5.8 | 0.79 | 0.68 | 8.6 |
| 32B | hidden | 1 | 11.6 | 7.0 | 6.1 | 0.75 | 0.62 | 6.0 |

Per-run tables: `results/govsim/hidden_q{7b,14b,32b}_runs.csv`. The yield column is new
(`analysis.yield_efficiency`): survival was the wrong readout for a village that survives by
starving.

**Caveats.** Hidden action changes the residents' own behaviour, so the visible and hidden games
are different games from month one, not paired. The "group's number" is what the framework model
extracted from the chat, and it was missing in two seeds. Ten seeds per cell. The 7B hidden cell
was run on the old job script (each game on its own node; checked: no port collision, no silent
agents).

**What's next.** (1) The base-model 7B stranger (conversion job queued) and the 72B stranger on the
GPU account, both hidden and visible: the two tests of whether conformity is tuning or capability.
(2) The pond (15 of 60 done) and the successor swap (queued). (3) The "what would you propose" probe
outside the game, queued. Nothing needed from Parag.
