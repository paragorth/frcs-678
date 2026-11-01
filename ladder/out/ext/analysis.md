Extension B, the maker chosen (choice prompts, versions and orders pooled):
| Pair | 0.5B | 1.5B | 3B | 7B | 14B | 32B | Slope | Fact at 32B | People at 32B |
|---|---|---|---|---|---|---|---|---|---|
| M1 | 50% | 50% | 50% | 50% | 100% | 100% | +0.095 | 100% | 75% |
| M2 | 38% | 57% | 100% | 100% | 100% | 100% | +0.109 | 100% | 75% |
| M3 | 52% | 38% | 50% | 50% | 100% | 100% | +0.100 | 100% | 100% |
| M4 | 57% | 38% | 100% | 15% | 82% | 72% | +0.027 | 100% | 50% |
| M5 | 35% | 50% | 100% | 75% | 100% | 100% | +0.110 | 100% | 25% |
| M6 | 40% | 28% | 100% | 75% | 100% | 100% | +0.118 | 100% | 95% |
B1: 6/6 pairs rise, one-sided Wilcoxon p = 0.0156. B2: maker chosen by majority in 6/6 pairs at 32B, sign test p = 0.0156.
q3.6-27b-off: M1 100%, M2 100%, M3 100%, M4 62%, M5 100%, M6 100%
l8b: M1 100%, M2 92%, M3 60%, M4 78%, M5 72%, M6 70%
l70b: M1 100%, M2 100%, M3 100%, M4 90%, M5 100%, M6 100%

Extension C, Llama, the semataxy option chosen on the main 24 cases:
| Model | All | Instinct | Order | Route | Ceiling | Fact | People |
|---|---|---|---|---|---|---|---|
| Llama-3.1-8B | 74% | 92% | 75% | 60% | 66% | 57% | 52% |
| Llama-3.3-70B (AWQ) | 90% | 100% | 88% | 88% | 83% | 95% | 66% |
C1: 70B above 8B in 18/21 cases that differ, one-sided Wilcoxon p = 0.000644.
