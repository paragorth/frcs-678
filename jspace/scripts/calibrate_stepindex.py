"""Calibrate the step-index / width discriminator against MATCHED references.

The statistic over-detects steps on flat, noisy curves -- shown last run, where
`random_matched` scored *sharper* than the concept condition purely because it
barely moved. That cuts both ways: if the references are generated at arbitrary
rise and zero noise, the decision boundary is in the wrong place and the null
is as unsafe as a false positive would be.

So both references are generated at the **rise and residual variance measured
from the concept condition itself**, per prompt:

  * STEP reference: logistic with a narrow transition, alpha50 jittered across
    prompts (a population of stepped units with distributed thresholds -- the
    exact thing a pooled test would hide), matched rise, matched noise.
  * SATURATION reference: smooth compressive curve, matched rise, matched noise.

The real data is then classified by which reference distribution its statistics
sit in, rather than against a threshold picked by hand.
"""
from __future__ import annotations
import json, sys
import numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from analyse_trajectories import fit_logistic, load, step_index


def curve_stats(a, y):
    (L, k, a50, c), r2 = fit_logistic(a, y)
    w = 4.0 / k if k > 1e-6 else np.inf
    resid = y - (L / (1 + np.exp(-np.clip(k * (a - a50), -60, 60))) + c)
    return w / (a.max() - a.min()), step_index(a, y), float(resid.std())


def measure(curves):
    W, S, R, RISE = [], [], [], []
    for p, (a, m) in sorted(curves.items()):
        y = m - m[0]
        wf, si, rs = curve_stats(a, y)
        W.append(wf); S.append(si); R.append(rs); RISE.append(float(y[-1]))
    return np.array(W), np.array(S), np.array(R), np.array(RISE)


def synth(a, rises, noise, kind, rng):
    """Generate reference curves at matched rise and matched residual noise."""
    out = {}
    for i, (rise, sd) in enumerate(zip(rises, noise)):
        if kind == "step":
            # narrow transition; threshold jittered across the population
            a50 = rng.uniform(a.min() + 0.15 * np.ptp(a), a.max() - 0.15 * np.ptp(a))
            k = rng.uniform(1.5, 4.0)          # width ~ 4/k = 1.0-2.7 alpha units
            y = rise / (1 + np.exp(-k * (a - a50)))
        else:
            tau = rng.uniform(0.25, 0.6) * np.ptp(a)
            y = rise * (1 - np.exp(-(a - a.min()) / tau))
        y = y - y[0] + rng.normal(0, sd, a.size)
        out[i] = (a, y)
    return out


if __name__ == "__main__":
    path, cond = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "concept")
    rng = np.random.default_rng(0)
    real = load(path, cond)
    a = real[sorted(real)[0]][0]
    W, S, R, RISE = measure(real)
    print(f"REAL ({cond}, n={len(W)}): rise median {np.median(RISE):.2f}, "
          f"residual sd median {np.median(R):.3f}")
    print(f"{'':14s} {'width/range':>18} {'step index':>18}")
    print(f"{'REAL':14s} {np.median(W):8.2f} [{np.percentile(W,25):.2f},"
          f"{np.percentile(W,75):.2f}] {np.median(S):8.2f} "
          f"[{np.percentile(S,25):.2f},{np.percentile(S,75):.2f}]")
    res = {"real": {"width_med": float(np.median(W)), "step_med": float(np.median(S)),
                    "rise_med": float(np.median(RISE)), "resid_med": float(np.median(R))}}
    for kind in ("step", "saturation"):
        cw, cs, _, _ = measure(synth(a, RISE, R, kind, rng))
        res[kind] = {"width_med": float(np.median(cw)), "step_med": float(np.median(cs)),
                     "width_iqr": [float(np.percentile(cw, 25)), float(np.percentile(cw, 75))],
                     "step_iqr": [float(np.percentile(cs, 25)), float(np.percentile(cs, 75))]}
        print(f"{'REF ' + kind:14s} {np.median(cw):8.2f} [{np.percentile(cw,25):.2f},"
              f"{np.percentile(cw,75):.2f}] {np.median(cs):8.2f} "
              f"[{np.percentile(cs,25):.2f},{np.percentile(cs,75):.2f}]")
    d_step = abs(np.log(res["real"]["width_med"] / res["step"]["width_med"]))
    d_sat = abs(np.log(res["real"]["width_med"] / res["saturation"]["width_med"]))
    res["verdict"] = "STEPPED" if d_step < d_sat else "SMOOTH"
    res["log_dist_to_step"], res["log_dist_to_saturation"] = float(d_step), float(d_sat)
    print(f"\nreal is closer to the {res['verdict']} reference "
          f"(log-dist {min(d_step,d_sat):.2f} vs {max(d_step,d_sat):.2f})")
    json.dump(res, open(path.replace(".jsonl.gz", f"_calib_{cond}.json"), "w"), indent=1)
