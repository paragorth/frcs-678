"""Cost and run time of the six-cell experiment, from the trial's measured shifts.

The experiment: Muse Spark 1.2 at minimal, medium and xhigh effort, under the
neutral and morality briefings, 30 shifts per cell on map seeds 0-29. The trial
measured one morality shift per effort on seed 0. Two numbers set the cost of a
cell: calls per shift and dollars per call.

Calls per shift. Seed 0 is one map of thirty. A scripted driver that copies
Muse's pattern in the trial (own crops only, "stay" otherwise, around every
obstacle) was run free over all thirty seeds, and its asks on each map scale
Muse's seed-0 count:
  low      Muse's seed-0 count, as measured
  central  Muse's seed-0 count x (scripted driver's 30-seed mean / its seed 0)
  high     the scripted driver's 30-seed mean itself, or the central figure if
           larger (Muse made fewer asks than the script on seed 0)

Dollars per call. The trial's own calls at that effort, at the project's Muse
prices ($1.25 / $4.25 per million input / output tokens, reasoning included,
all input at the full price although Meta served about half from its cache):
the mean for low and central, the upper end of a 95% bootstrap interval of the
mean for high. The neutral cells are priced like the morality cells: their
briefing is the same less one paragraph; the trial has no neutral shift.

Run time: calls within a shift are sequential and take the trial's mean seconds
per call; shifts run side by side up to the number of calls in flight at once.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from muse_guard import PRICE_IN, PRICE_OUT  # noqa: E402

SHIFTS_PER_CELL = 30
BRIEFINGS = ("neutral", "morality")
EFFORTS = ("minimal", "low", "medium", "high", "xhigh")
IN_FLIGHT = (3, 10)   # calls at once: the panel's max_connections, and 10
JOB_HOURS = 12        # wall limit per job on the SL4 QOS


def boot_mean_ci(xs: list[float], n: int = 10_000) -> tuple[float, float]:
    rng = random.Random(0)
    means = sorted(statistics.fmean(rng.choices(xs, k=len(xs))) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def cost(summary: dict, ledger: list[dict], asks_by_seed: dict) -> dict:
    script = [asks_by_seed[s] for s in sorted(asks_by_seed)]
    script_mean, script_seed0 = statistics.fmean(script), asks_by_seed[0]
    ratio = script_mean / script_seed0
    rows = {}
    for effort in sorted(summary["shifts"], key=EFFORTS.index):
        s = summary["shifts"][effort]
        calls0 = s["calls_sent"]
        per = [r["cost"] for r in ledger
               if r.get("shift") == effort and r.get("event") == "call"]
        lo_ci, hi_ci = boot_mean_ci(per)
        mean = statistics.fmean(per)
        calls = {"low": calls0, "central": calls0 * ratio,
                 "high": max(script_mean, calls0 * ratio)}
        usd_call = {"low": mean, "central": mean, "high": hi_ci}
        shift = {k: calls[k] * usd_call[k] for k in calls}
        secs = s["seconds_per_call_mean"]
        rows[effort] = {
            "trial_calls_seed0": calls0, "trial_cost_usd": s["cost_usd"],
            "trial_shift_minutes": round(s["shift_wall_s"] / 60, 1),
            "usd_per_call": {"mean": round(mean, 5),
                             "ci95": [round(lo_ci, 5), round(hi_ci, 5)]},
            "seconds_per_call": secs,
            "calls_per_shift": {k: round(v, 1) for k, v in calls.items()},
            "usd_per_shift": {k: round(v, 3) for k, v in shift.items()},
            "usd_per_cell": {k: round(SHIFTS_PER_CELL * v, 2) for k, v in shift.items()},
            "call_hours_per_cell": {k: round(SHIFTS_PER_CELL * calls[k] * secs / 3600, 2)
                                    for k in calls},
        }
    nb = len(BRIEFINGS)
    keys = ("low", "central", "high")
    total = {k: round(nb * sum(r["usd_per_cell"][k] for r in rows.values()), 2)
             for k in keys}
    call_hours = {k: round(nb * sum(r["call_hours_per_cell"][k] for r in rows.values()), 1)
                  for k in keys}
    calls_total = {k: round(nb * SHIFTS_PER_CELL * sum(r["calls_per_shift"][k]
                                                      for r in rows.values()))
                   for k in keys}
    return {
        "cells": rows, "briefings": BRIEFINGS, "shifts_per_cell": SHIFTS_PER_CELL,
        "seed_sweep": {"asks_by_seed": asks_by_seed, "seed0": script_seed0,
                       "mean": round(script_mean, 1), "min": min(script),
                       "max": max(script), "ratio_mean_to_seed0": round(ratio, 3)},
        "total_usd": total, "total_calls": calls_total, "call_hours": call_hours,
        "wall_hours": {f"{n}_in_flight": {k: round(call_hours[k] / n, 1) for k in keys}
                       for n in IN_FLIGHT},
        "job_hours_limit": JOB_HOURS,
        "prices_per_million": {"input": PRICE_IN * 1e6, "output": PRICE_OUT * 1e6}}


def markdown(c: dict) -> str:
    sw = c["seed_sweep"]
    out = ["| Effort | Trial, seed 0 | Dollars per call (95% range) | Calls per shift: "
           "low / central / high | Per cell of 30: low / central / high |",
           "|---|---|---|---|---|"]
    for e, r in c["cells"].items():
        cp, uc, ci = r["calls_per_shift"], r["usd_per_cell"], r["usd_per_call"]
        out.append(
            f"| {e} | {r['trial_calls_seed0']} calls, ${r['trial_cost_usd']:.2f}, "
            f"{r['trial_shift_minutes']} min | ${ci['mean']:.4f} "
            f"(${ci['ci95'][0]:.4f}-${ci['ci95'][1]:.4f}) "
            f"| {cp['low']:.0f} / {cp['central']:.0f} / {cp['high']:.0f} "
            f"| ${uc['low']:.2f} / ${uc['central']:.2f} / ${uc['high']:.2f} |")
    t, n, h, w = c["total_usd"], c["total_calls"], c["call_hours"], c["wall_hours"]
    out += ["",
            f"Scripted driver over seeds 0-29: {sw['min']}-{sw['max']} asks per shift, "
            f"mean {sw['mean']}, seed 0 {sw['seed0']} (seed 0 is a light map: "
            f"x{sw['ratio_mean_to_seed0']}).",
            "",
            f"All six cells (180 shifts): **${t['central']:.0f} central, "
            f"${t['low']:.0f} to ${t['high']:.0f}**; {n['central']:,} calls "
            f"({n['low']:,} to {n['high']:,}).",
            f"Run time: {h['central']} hours of calls ({h['low']}-{h['high']}); "
            f"with 3 calls in flight about {w['3_in_flight']['central']} h "
            f"({w['3_in_flight']['low']}-{w['3_in_flight']['high']}), with 10 about "
            f"{w['10_in_flight']['central']} h ({w['10_in_flight']['low']}-"
            f"{w['10_in_flight']['high']}); a job may run {c['job_hours_limit']} h."]
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--summary", required=True, type=Path,
                    help="trial_summary.json from check_trial.py")
    ap.add_argument("--ledger", required=True, type=Path,
                    help="spend_ledger_live.jsonl from the trial")
    ap.add_argument("--sweep", required=True, type=Path,
                    help="run_mock.json of the scripted driver over seeds 0-29")
    ap.add_argument("--out", required=True, type=Path,
                    help="costing JSON; a .md table is written beside it")
    a = ap.parse_args()
    ledger = [json.loads(x) for x in a.ledger.read_text().splitlines() if x.strip()]
    asks = {int(k): v for k, v in
            json.loads(a.sweep.read_text())["logs"][0]["asks_by_seed"].items()}
    c = cost(json.loads(a.summary.read_text()), ledger, asks)
    a.out.write_text(json.dumps(c, indent=1))
    md = markdown(c)
    a.out.with_suffix(".md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
