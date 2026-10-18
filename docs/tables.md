## Table 1 — SOVSIM: the ruler's capability

Subordinates identical in every row; only the model in the ruler's seat
changes. Metrics are SOVSIM's own. `leader extr` is the fraction of the
ruler's permitted maximum that it actually took.

| cell                                               | batch        | n              | survival mean [95% CI] | collapsed | ruler take (all) | ruler take (parsed) | empty reasonings | efficiency | provenance        |
|----------------------------------------------------|--------------|----------------|------------------------|-----------|------------------|---------------------|------------------|------------|-------------------|
| Sonnet 5 subordinates / **(not recorded)** ruler   | out_frontier | 5              | 2.0 [2.0, 2.0]         | 5/5       | 6.30             | 6.30                | 0/10             | 0.241      | unknown           |
| Haiku 4.5 subordinates / Opus 5 ruler              | out_king     | 10             | 7.3 [5.2, 9.5]         | 9/10      | 15.04            | 18.30               | 13/73            | 0.575      | declared (medium) |
| Sonnet 5 subordinates / **(not recorded)** ruler   | out_king     | 10             | 2.0 [2.0, 2.0]         | 10/10     | 0.60             | 0.67                | 2/20             | 0.225      | unknown           |
| Haiku 4.5 subordinates / Haiku 4.5 ruler           | out_ladder   | 10             | 1.7 [1.2, 2.2]         | 10/10     | 29.82            | 29.82               | 0/17             | 0.206      | declared (high)   |
| Opus 5 subordinates / Opus 5 ruler                 | out_ladder   | 10             | 4.9 [4.2, 5.6]         | 10/10     | 18.86            | 20.09               | 3/49             | 0.430      | declared (high)   |
| Sonnet 5 subordinates / Sonnet 5 ruler             | out_ladder   | 10             | 1.8 [1.5, 2.0]         | 10/10     | 8.67             | 8.67                | 0/18             | 0.220      | declared (high)   |
| Haiku 4.5 subordinates / Sonnet 5 ruler            | out_mid      | 10             | 4.0 [2.1, 6.6]         | 10/10     | 20.55            | 21.08               | 1/40             | 0.376      | declared (medium) |
| Sonnet 5 subordinates / **(not recorded)** ruler   | out_test     | 1              | 2.0 [2.0, 2.0]         | 1/1       | 0.00             | 0.00                | 0/2              | 0.250      | unknown           |
| Haiku 4.5 subordinates / Haiku 4.5 ruler           | out_v2       | 10             | 1.2 [1.0, 1.5]         | 10/10     | 47.50            | 47.50               | 0/12             | 0.178      | stamped in file   |
| Haiku 4.5 subordinates / Opus 5 ruler              | out_v2       | 5 (+1 running) | 5.4 [3.8, 7.0]         | 5/5       | 17.89            | 17.89               | 0/27             | 0.447      | stamped in file   |
| Haiku 4.5 subordinates / Opus 5 ruler from round 3 | out_v2       | 10             | 1.5 [1.1, 1.9]         | 10/10     | 23.20            | 23.20               | 0/15             | 0.191      | stamped in file   |
| Haiku 4.5 subordinates / Sonnet 5 ruler            | out_v2       | 10             | 3.4 [2.0, 5.6]         | 10/10     | 18.71            | 18.71               | 0/34             | 0.310      | stamped in file   |
| Opus 5 subordinates / Haiku 4.5 ruler              | out_v2       | 10             | 1.5 [1.1, 1.9]         | 10/10     | 51.20            | 51.20               | 0/15             | 0.190      | stamped in file   |
| Sonnet 5 subordinates / Opus 5 ruler               | out_v2       | 10             | 2.0 [2.0, 2.0]         | 10/10     | 0.90             | 0.90                | 0/20             | 0.224      | stamped in file   |

`ruler take (all)` averages every round. `ruler take (parsed)` excludes
rounds whose ruler reasoning came back empty: a parse fallback yields an
extraction of 0, which is maximal restraint, so those rounds inflate the
apparent restraint of precisely the cells that look best. The difference
is material for the Opus rung and must be read as the honest figure.

### Survival statistics

Runs reaching the 12-round horizon without collapsing are right-censored:
we know they lasted at least that long, not that they would have collapsed
afterwards. RMST is restricted mean survival time over the 12-round
window, which handles that correctly; the Mann-Whitney is kept as a
robustness check and its p-values are Holm-adjusted across the family.

| cell                                          | n  | observed collapses | RMST (12-round window) |
|-----------------------------------------------|----|--------------------|------------------------|
| Haiku 4.5 subs, Opus 5 ruler [out_king]       | 10 | 9/10               | 7.30                   |
| Haiku 4.5 subs, Haiku 4.5 ruler [out_ladder]  | 10 | 10/10              | 1.70                   |
| Opus 5 subs, Opus 5 ruler [out_ladder]        | 10 | 10/10              | 4.90                   |
| Sonnet 5 subs, Sonnet 5 ruler [out_ladder]    | 10 | 10/10              | 1.80                   |
| Haiku 4.5 subs, Sonnet 5 ruler [out_mid]      | 10 | 10/10              | 4.00                   |
| Haiku 4.5 subs, Haiku 4.5 ruler [out_v2]      | 10 | 10/10              | 1.20                   |
| Haiku 4.5 subs, Opus 5 ruler [out_v2]         | 5  | 5/5                | 5.40                   |
| Haiku 4.5 subs, Opus 5 ruler from r3 [out_v2] | 10 | 10/10              | 1.50                   |
| Haiku 4.5 subs, Sonnet 5 ruler [out_v2]       | 10 | 10/10              | 3.40                   |
| Opus 5 subs, Haiku 4.5 ruler [out_v2]         | 10 | 10/10              | 1.50                   |
| Sonnet 5 subs, Opus 5 ruler [out_v2]          | 10 | 10/10              | 2.00                   |

#### Pairwise Mann-Whitney on survival time, Holm-adjusted

| cell A                                        | cell B                                        | U     | z     | p raw  | p Holm    |
|-----------------------------------------------|-----------------------------------------------|-------|-------|--------|-----------|
| Haiku 4.5 subs, Opus 5 ruler [out_king]       | Haiku 4.5 subs, Haiku 4.5 ruler [out_ladder]  | 98.0  | +3.69 | 0.0002 | 0.0039  * |
| Haiku 4.5 subs, Opus 5 ruler [out_king]       | Haiku 4.5 subs, Sonnet 5 ruler [out_mid]      | 84.0  | +2.62 | 0.0087 | 0.0779    |
| Haiku 4.5 subs, Opus 5 ruler [out_king]       | Haiku 4.5 subs, Haiku 4.5 ruler [out_v2]      | 100.0 | +3.92 | 0.0001 | 0.0018  * |
| Haiku 4.5 subs, Opus 5 ruler [out_king]       | Haiku 4.5 subs, Opus 5 ruler from r3 [out_v2] | 99.0  | +3.78 | 0.0002 | 0.0029  * |
| Haiku 4.5 subs, Opus 5 ruler [out_king]       | Haiku 4.5 subs, Sonnet 5 ruler [out_v2]       | 87.0  | +2.89 | 0.0039 | 0.0430  * |
| Haiku 4.5 subs, Haiku 4.5 ruler [out_ladder]  | Haiku 4.5 subs, Sonnet 5 ruler [out_mid]      | 23.0  | -2.17 | 0.0302 | 0.1951    |
| Haiku 4.5 subs, Haiku 4.5 ruler [out_ladder]  | Haiku 4.5 subs, Opus 5 ruler [out_v2]         | 1.0   | -3.02 | 0.0026 | 0.0308  * |
| Haiku 4.5 subs, Haiku 4.5 ruler [out_ladder]  | Haiku 4.5 subs, Opus 5 ruler from r3 [out_v2] | 56.5  | +0.55 | 0.5840 | 0.5994    |
| Haiku 4.5 subs, Haiku 4.5 ruler [out_ladder]  | Haiku 4.5 subs, Sonnet 5 ruler [out_v2]       | 25.5  | -2.00 | 0.0455 | 0.2277    |
| Opus 5 subs, Opus 5 ruler [out_ladder]        | Opus 5 subs, Haiku 4.5 ruler [out_v2]         | 99.5  | +3.81 | 0.0001 | 0.0026  * |
| Sonnet 5 subs, Sonnet 5 ruler [out_ladder]    | Sonnet 5 subs, Opus 5 ruler [out_v2]          | 40.0  | -1.45 | 0.1462 | 0.4387    |
| Haiku 4.5 subs, Sonnet 5 ruler [out_mid]      | Haiku 4.5 subs, Haiku 4.5 ruler [out_v2]      | 94.0  | +3.56 | 0.0004 | 0.0059  * |
| Haiku 4.5 subs, Sonnet 5 ruler [out_mid]      | Haiku 4.5 subs, Opus 5 ruler [out_v2]         | 11.0  | -1.78 | 0.0752 | 0.3007    |
| Haiku 4.5 subs, Sonnet 5 ruler [out_mid]      | Haiku 4.5 subs, Opus 5 ruler from r3 [out_v2] | 84.0  | +2.74 | 0.0061 | 0.0614    |
| Haiku 4.5 subs, Haiku 4.5 ruler [out_v2]      | Haiku 4.5 subs, Opus 5 ruler [out_v2]         | 0.0   | -3.32 | 0.0009 | 0.0124  * |
| Haiku 4.5 subs, Haiku 4.5 ruler [out_v2]      | Haiku 4.5 subs, Opus 5 ruler from r3 [out_v2] | 39.0  | -1.04 | 0.2997 | 0.5994    |
| Haiku 4.5 subs, Haiku 4.5 ruler [out_v2]      | Haiku 4.5 subs, Sonnet 5 ruler [out_v2]       | 7.0   | -3.53 | 0.0004 | 0.0062  * |
| Haiku 4.5 subs, Opus 5 ruler [out_v2]         | Haiku 4.5 subs, Opus 5 ruler from r3 [out_v2] | 49.5  | +3.11 | 0.0018 | 0.0240  * |
| Haiku 4.5 subs, Opus 5 ruler [out_v2]         | Haiku 4.5 subs, Sonnet 5 ruler [out_v2]       | 42.0  | +2.20 | 0.0279 | 0.1951    |
| Haiku 4.5 subs, Opus 5 ruler from r3 [out_v2] | Haiku 4.5 subs, Sonnet 5 ruler [out_v2]       | 18.0  | -2.63 | 0.0087 | 0.0779    |

## Table 2 — GovSim: one capable newcomer in a cohort of five

Five agents in every cell; only the fifth agent's model changes.
Sustainable total harvest is 50 tons/month. Completed runs only.

| cell                | n valid | excluded | reached month 12 | months [95% CI]   | total/month       | 5th agent         | its 4 peers       |
|---------------------|---------|----------|------------------|-------------------|-------------------|-------------------|-------------------|
| 5x Haiku (control)  | 3       | 2        | 1/3              | 6.3 [3.0, 12.0]   | 26.8 [16.2, 38.1] | 5.8 [3.4, 8.5]    | 5.2 [4.1, 6.4]    |
| 4x Haiku + Sonnet 5 | 4       | 1        | 1/4              | 6.0 [3.5, 10.0]   | 31.8 [22.8, 40.9] | 5.8 [4.0, 7.5]    | 6.5 [5.5, 7.5]    |
| 4x Haiku + Opus 5   | 3       | 5        | 3/3              | 12.0 [12.0, 12.0] | 30.7 [26.6, 34.9] | 5.9 [5.1, 6.7]    | 6.2 [5.7, 6.7]    |
| 5x Sonnet (control) | 2       | 3        | 2/2              | 12.0 [12.0, 12.0] | 50.0 [50.0, 50.0] | 10.0 [10.0, 10.0] | 10.0 [10.0, 10.0] |
| 4x Sonnet + Opus 5  | 0       | 5        | -                | -                 | -                 | -                 | -                 |

Runs are excluded when more than 5% of their model calls fell back to a
default value. An exhausted API quota returns an empty body that the
harness logs as a default rather than an error, which makes a broken run
look like a cohort of unusually cautious agents.

### Survival to the horizon, against the matched control (Fisher exact)

| treatment         | treated | control | p     |
|-------------------|---------|---------|-------|
| 4x Haiku + Sonnet | 1/4     | 1/3     | 1.000 |
| 4x Haiku + Opus   | 3/3     | 1/3     | 0.400 |
