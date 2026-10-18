"""PRIMARY analysis: per-prompt margin trajectories across alpha.

Asks, for each prompt separately, whether its margin curve steps or saturates
smoothly. Pooling is deliberately avoided here -- see stage4_bimodality.py for
why a pooled test cannot answer this.

Per prompt, the normalised curve d(a) = m(a) - m(0) is fitted with a 4-parameter
logistic. The discriminator is the transition width w = 4/k, the alpha range
over which the curve moves 12% -> 88% of its total rise, expressed as a
fraction of the swept range:

    w/range small  + alpha50 varying across prompts -> stepped units, jittered
                                                       thresholds (ignition)
    w/range ~1 or larger, for most prompts           -> smooth saturation

`step_index` (max local slope / mean slope) is reported alongside as a
fit-free check: ~1 for a straight line, large for a step.
"""
from __future__ import annotations
import gzip, json, sys
from collections import defaultdict
import numpy as np


def logistic(a, L, k, a50, c):
    return L / (1.0 + np.exp(-np.clip(k * (a - a50), -60, 60))) + c


def fit_logistic(a, y):
    """Grid + local refinement. Returns (params, r2). No scipy dependency."""
    span = y.max() - y.min()
    if span < 1e-9:
        return (0.0, 1.0, float(np.median(a)), float(y.mean())), 0.0
    best, bestsse = None, np.inf
    arange = a.max() - a.min()
    for k in np.concatenate([np.linspace(0.02, 2.0, 40), np.linspace(2.2, 12, 20)]):
        for a50 in np.linspace(a.min() - 0.2 * arange, a.max() + 0.2 * arange, 40):
            X = np.stack([1.0 / (1.0 + np.exp(-np.clip(k * (a - a50), -60, 60))),
                          np.ones_like(a)], 1)
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            sse = float(((X @ coef - y) ** 2).sum())
            if sse < bestsse:
                bestsse, best = sse, (float(coef[0]), float(k), float(a50), float(coef[1]))
    sst = float(((y - y.mean()) ** 2).sum())
    return best, (1 - bestsse / sst if sst > 0 else 0.0)


def step_index(a, y):
    d = np.diff(y) / np.diff(a)
    mean_slope = (y[-1] - y[0]) / (a[-1] - a[0])
    return float(np.max(np.abs(d)) / abs(mean_slope)) if abs(mean_slope) > 1e-9 else np.nan


def load(path, condition):
    cur = defaultdict(dict)
    with gzip.open(path, "rt") as fh:
        for line in fh:
            r = json.loads(line)
            if r["condition"] == condition and r["alpha"] is not None:
                cur[r["prompt"]][r["alpha"]] = r["margin"]
    out = {}
    for p, d in cur.items():
        a = np.array(sorted(d)); out[p] = (a, np.array([d[x] for x in a]))
    return out


def analyse(path, condition):
    curves = load(path, condition)
    rows = []
    for p, (a, m) in sorted(curves.items()):
        y = m - m[0]                      # normalise to change, not level
        (L, k, a50, c), r2 = fit_logistic(a, y)
        w = 4.0 / k if k > 1e-6 else np.inf
        rows.append({"prompt": p, "L": L, "k": k, "a50": a50, "r2": r2,
                     "width": w, "width_frac": w / (a.max() - a.min()),
                     "step_index": step_index(a, y), "total_rise": float(y[-1])})
    return rows


def report(rows, label):
    wf = np.array([r["width_frac"] for r in rows])
    si = np.array([r["step_index"] for r in rows])
    a50 = np.array([r["a50"] for r in rows])
    rise = np.array([r["total_rise"] for r in rows])
    ok = rise > 0.5                        # prompts that actually responded
    print(f"\n=== {label}  (n={len(rows)}, responded={ok.sum()}) ===")
    print(f"  transition width / swept range : median {np.median(wf[ok]):.2f}  "
          f"IQR [{np.percentile(wf[ok],25):.2f}, {np.percentile(wf[ok],75):.2f}]")
    print(f"  step index (1 = linear)        : median {np.median(si[ok]):.2f}  "
          f"p90 {np.percentile(si[ok],90):.2f}")
    print(f"  alpha50 across prompts         : median {np.median(a50[ok]):.2f}  "
          f"SD {a50[ok].std():.2f}  IQR [{np.percentile(a50[ok],25):.2f}, "
          f"{np.percentile(a50[ok],75):.2f}]")
    sharp = (wf[ok] < 0.25).mean()
    print(f"  prompts with sharp transition (width < 25% of range): {100*sharp:.0f}%")
    print(f"  -> {'STEPPED units' if sharp > 0.5 else 'SMOOTH saturation'} "
          f"by the pre-stated criterion (>50% sharp)")
    return {"median_width_frac": float(np.median(wf[ok])),
            "median_step_index": float(np.median(si[ok])),
            "a50_sd": float(a50[ok].std()), "frac_sharp": float(sharp),
            "n_responded": int(ok.sum())}


if __name__ == "__main__":
    path = sys.argv[1]
    out = {}
    for cond in ("concept", "random_matched"):
        rows = analyse(path, cond)
        out[cond] = report(rows, cond)
        out[cond + "_rows"] = rows
    json.dump(out, open(path.replace(".jsonl.gz", "_trajectories.json"), "w"), indent=1)
    print(f"\n-> {path.replace('.jsonl.gz', '_trajectories.json')}")
