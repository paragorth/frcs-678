"""Extension A's registered analysis: does thinking amplify what a model already leans to?

  python analyse_think.py --results out/think/*.jsonl --out out/think/analysis
Per model and case: the share of choices for the semataxy option with thinking off and on (versions and
orders pooled, unparsed left out). Test T (one per model, Holm across the two): polarisation gain
|share_on - 0.5| - |share_off - 0.5| > 0 across the 24 cases, one-sided Wilcoxon, zeros dropped.
Reported, not tested: the direction of the change (share_on - share_off, two-sided Wilcoxon), by category,
parse rates and reply length.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


def share(rr: list[dict]) -> float:
    got = [r["is_s"] for r in rr if r["is_s"] is not None]
    return sum(got) / len(got) if got else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    rows = []
    for p in a.results:
        rows += [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    by = defaultdict(list)
    for r in rows:
        by[(r["label"], r["thinking"], r["case"])].append(r)
    labels = sorted({r["label"] for r in rows})
    cases = sorted({r["case"] for r in rows}, key=lambda c: ("IORC".index(c[0]), c))
    res, ps, lines = {}, {}, []
    lines.append("| Model | Thinking | Semataxy choice | Instinct | Order | Route | Ceiling | Parsed | Mean reply tokens |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for l in labels:
        per = {}
        for c in cases:
            off, on = share(by[(l, "off", c)]), share(by[(l, "on", c)])
            per[c] = {"off": off, "on": on,
                      "polar_gain": (abs(on - 0.5) - abs(off - 0.5)) if not (math.isnan(on) or math.isnan(off)) else float("nan"),
                      "shift": on - off}
        gains = [v["polar_gain"] for v in per.values() if not math.isnan(v["polar_gain"]) and abs(v["polar_gain"]) > 1e-12]
        shifts = [v["shift"] for v in per.values() if not math.isnan(v["shift"]) and abs(v["shift"]) > 1e-12]
        t = {"n_nonzero": len(gains), "n_up": sum(g > 0 for g in gains),
             "median_gain": float(np.median(gains)) if gains else None,
             "p": float(stats.wilcoxon(gains, alternative="greater").pvalue) if gains else 1.0,
             "shift_median": float(np.median(shifts)) if shifts else None,
             "shift_p_two_sided": float(stats.wilcoxon(shifts).pvalue) if shifts else 1.0}
        ps[l] = t["p"]
        res[l] = {"per_case": per, "test": t}
        for mode in ("off", "on"):
            rr = [r for r in rows if r["label"] == l and r["thinking"] == mode]
            cat = {k: share([r for r in rr if r["cat"] == k]) for k in "IORC"}
            parsed = sum(r["parsed"] is not None for r in rr)
            f = lambda x: "-" if math.isnan(x) else f"{100 * x:.0f}%"
            lines.append(f"| {l} | {mode} | {f(share(rr))} | {f(cat['I'])} | {f(cat['O'])} | {f(cat['R'])} | "
                         f"{f(cat['C'])} | {parsed}/{len(rr)} | {np.mean([r['out_tokens'] for r in rr]):.0f} |")
    order = sorted(ps, key=ps.get)
    run = 0.0
    for i, l in enumerate(order):
        run = max(run, min(1.0, (len(order) - i) * ps[l]))
        res[l]["test"]["p_holm"] = run
    lines.append("")
    lines.append("| Model | Cases more polarised / non-zero | Median gain | p | Holm p | Median shift toward semataxy | Shift p (two-sided) |")
    lines.append("|---|---|---|---|---|---|---|")
    for l in labels:
        t = res[l]["test"]
        lines.append(f"| {l} | {t['n_up']}/{t['n_nonzero']} | {t['median_gain']:+.3f} | {t['p']:.3g} | "
                     f"{t['p_holm']:.3g} | {t['shift_median']:+.3f} | {t['shift_p_two_sided']:.3g} |")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.with_suffix(".json").write_text(json.dumps(res, indent=1))
    a.out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
