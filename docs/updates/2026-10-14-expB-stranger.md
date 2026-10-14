# Update, 14 October 2026: Experiment B, the superior newcomer, first result (14B residents)

**What happened.** Twenty stranger runs and five 14B ladder runs completed on CSD3, no crashes, no
timeouts, every call answered (parse fallbacks are all the known "next speaker: Luke/Mayor" quirk in
the group chat, logged per run). Four 14B residents fish for four months; a stranger joins in month
5; 12 months total. The stranger is either the same 14B model (control) or Qwen2.5 32B (the test),
and is given either no instruction about how to fish (neutral) or the upstream "outsider" persona
that tells it to maximise its own catch (greedy). Five seeds each.

**What it means for the thesis.** Both preregistered Experiment B predictions hold, on one
capability step, with the caveats below.

- *Neutral stranger (the real test).* The 32B stranger never exploited. It joined at 10, matched the
  residents at 8, and from month 8 the whole group, residents included, moved up to 10 tons each,
  which is exactly the sustainable optimum for five fishermen (50 taken, lake regrows to 100). Group
  yield rose from 32 to 50 tons a month with no loss of stock. Every seed reached month 12. The 14B
  control stranger also joined cooperatively but the group stayed at 8 each, under the optimum, and
  one seed collapsed at month 10. So the more capable neutral entrant left residents' share
  untouched (0.80 both), kept the commons alive in more seeds (5/5 vs 4/5), and moved the group to
  a better equilibrium rather than a worse one. The stranger's forecast of the others' catch was off
  by 2.8 tons a month at 32B against 5.0 at 14B: it modelled the residents nearly perfectly.
- *Greedy stranger (the bound).* Capability amplifies the instruction. The greedy 14B stranger took
  about 27 a month and the lake died in month 6 or 7. The greedy 32B stranger asked for the entire
  lake (100 tons) in its first month, got 68, and the lake died at once, in every seed. Residents'
  share fell from 0.53 to 0.32. Exactly as the theory said it would, and exactly what the
  orthogonality thesis would also say.
- *The honest framing.* The persona dominates behaviour, and capability amplifies whichever persona
  is given. A sceptic will read this as orthogonality: give a capable model a goal and it pursues it
  better, kind or cruel. Parag's reading is that the *neutral* case is the one that matters, because
  it shows the model's default disposition absent any goal, and that default is to join the group
  and lift it to the optimum, more so when more capable. Both readings fit the data. What would
  separate them: more capability steps (72B neutral and greedy), a stranger given a *mild* selfish
  instruction rather than an extreme one, and residents weaker than 14B so the stranger has real
  power to exploit without an instruction. That last design is the one worth running next.

**The numbers** (5 seeds per condition; survival = months before the lake fell below 5 tons, max 12).

| Condition | survival (mean, min) | stranger catch/month | resident catch/month after entry | residents' share | stranger forecast error |
|---|---|---|---|---|---|
| 14B residents alone (Experiment A) | 8.4, 3 | | 3 seeds at 8 to 10; 2 collapse at month 3 | | 6.0 |
| 14B stranger, neutral | 11.6, 10 | 8.3 | 8.0 | 0.80 | 5.0 |
| 32B stranger, neutral | 12.0, 12 | 8.5 | 8.6 | 0.80 | 2.8 |
| 14B stranger, greedy | 6.2, 6 | 26.9 | 7.4 | 0.53 | 15.9 |
| 32B stranger, greedy | 5.0, 5 | 68 (asked for 100) | 8.0 | 0.32 | 28.6 |

Per-run table: `results/govsim/b14_runs.csv`. Run directories on CSD3 under `runs/`.

**Caveats, stated plainly.**
1. *Seeds are near-replicas.* Decoding is greedy, so the five seeds differ only through the game's
   random allocation when demand exceeds stock. The 32B-greedy rows are identical across all five
   seeds. Treat each condition as roughly one to two independent observations, not five. A rerun with
   sampled decoding (temperature above zero, seeded) is the way to get real spread; it would also
   break the branch-identity property the design currently relies on, so it is a design change for
   Parag to approve.
2. *One capability step.* 14B to 32B. 72B is downloadable but needs GPUs to run in reasonable time.
3. *The residents are not weak.* 14B residents sustain the lake on their own in 3 of 5 seeds. A
   stranger joining a group that already plays well has little room to "stabilise" it; the 32B
   neutral stranger's contribution was moving the group from 8 to the optimal 10, which is real but
   modest. Residents at 7B collapse before the stranger arrives, so there is no CPU-feasible weaker
   resident on the Qwen ladder; a different family at 7B to 9B that survives (Llama 3.1 8B, Gemma 2
   9B) would be the thing to try.
4. *Experiment A at 14B* is mixed: 3 seeds survive all 12 months at 8 to 10 each; 2 collapse at month
   3 after two agents take 20 in month 1. Forecast error 5 to 7 tons against 60 to 130 at 7B and
   below. 14B is the first size that models the other players.

**What's next.**
1. Commit this; Parag reads.
2. Proposed next runs, cheap on CPU, in order: (a) weaker residents from another family that survive
   alone (test Llama 3.1 8B and Gemma 2 9B in Experiment A first, 5 seeds each, about 40 core-hours
   each); (b) a mild-selfish persona for the stranger; (c) sampled decoding rerun of the four
   conditions above if Parag approves the design change.
3. 72B stranger when GPUs arrive.
