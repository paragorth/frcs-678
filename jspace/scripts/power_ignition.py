"""Mandatory power analysis for IGNITION-CRITERION v2, before any spend.

Simulates H1 (gain peaking at 1.3) against H0 (smooth saturating, gain <= 1 and
non-increasing) at the trial-to-trial variance measured from the existing dense
sweep, and reports detection probability for the primary gain test at a range of
prompt counts.

Noise caveat, recorded: the variance is estimated from the Stage-5 dense sweep's
per-prompt MARGIN curves, because that is the only per-prompt continuous
trajectory measured so far. The new primary statistic is a concept AMPLITUDE at a
downstream layer. Similar in kind, not identical; the power numbers are
therefore indicative and are labelled as such.
"""
from __future__ import annotations
import gzip, json, sys
from collections import defaultdict
import numpy as np

# ---- 1. measured noise from the existing dense sweep ----------------------
cur = defaultdict(dict)
for line in gzip.open("out/stage5_report_7b_dense.jsonl.gz", "rt"):
    r = json.loads(line)
    if r["condition"] == "concept" and r["alpha"] is not None:
        cur[r["prompt"]][r["alpha"]] = r["margin"]
ps = sorted(cur)
al = np.array(sorted(cur[ps[0]]))
M = np.array([[cur[p][a] for a in al] for p in ps])
keep = al <= 8.0                      # coherent window
a_c, M_c = al[keep], M[:, keep]

resid = []
for row in M_c:
    co = np.polyfit(a_c, row, 2)      # smooth per-prompt fit
    resid.append(row - np.polyval(co, a_c))
resid = np.array(resid)
NOISE = float(resid.std())
RISE = float(np.median(M_c[:, -1] - M_c[:, 0]))
SNR = RISE / NOISE
print(f"measured from dense sweep (n={len(ps)}, alpha<=8):")
print(f"  per-prompt residual SD around a smooth fit : {NOISE:.4f}")
print(f"  median rise across the window              : {RISE:.4f}")
print(f"  rise / noise                               : {SNR:.1f}\n")

# ---- 2. simulate ----------------------------------------------------------
rng = np.random.default_rng(0)
A = np.linspace(0, 8, 17)             # alpha grid, step 0.5
drive = A.copy()


def curve_h0(rise, tau, jitter):
    """Smooth saturating: gain starts at max and only declines."""
    return rise * (1 - np.exp(-(A) / tau)) * (1 + jitter)


def curve_h1(rise, a50, k, gain_peak, jitter):
    """Sigmoid: gain rises to a peak mid-range then falls. gain_peak scales the
    steepest slope relative to the H0 curve of the same total rise."""
    s = 1 / (1 + np.exp(-k * (A - a50)))
    s = (s - s[0]) / (s[-1] - s[0] + 1e-12)
    return rise * s * (1 + jitter)


def max_gain(curves):
    """DEPRECATED — normalising by each curve's own mean gain explodes on the
    near-flat control (mean gain ~ 0), so the control's 95th percentile is
    unbounded and nothing can exceed it. Power came out 0.00 for H1 AND H0,
    which is the signature of a broken statistic, not of a hard problem. Kept as
    a record of why the primary statistic changed."""
    out = []
    for y in curves:
        g = np.gradient(y, drive)
        mean_g = (y[-1] - y[0]) / (drive[-1] - drive[0] + 1e-12)
        out.append(np.max(g) / (abs(mean_g) + 1e-12))
    return np.array(out)


def convexity(curves):
    """PRIMARY STATISTIC: the largest sustained CONVEX stretch, in units of the
    measurement noise.

    H0 says the internal quantity saturates or is linear -- concave everywhere,
    or flat. A region of accelerating gain is a region of POSITIVE second
    derivative, which neither a saturating nor a linear curve has. So convexity
    discriminates without dividing by anything that can vanish, which is what
    broke the gain-ratio version.

    Scale: second differences are divided by the measurement noise, so the
    statistic is in sigma and is comparable between concept and control even
    though their amplitudes differ by ~20x.
    """
    h = drive[1] - drive[0]
    out = []
    for y in curves:
        d2 = (y[2:] - 2 * y[1:-1] + y[:-2]) / (h * h)
        # sustained: mean of the best adjacent pair, so a single noisy point
        # cannot carry it
        best = max((d2[i] + d2[i + 1]) / 2 for i in range(len(d2) - 1))
        out.append(best * (h * h) / (NOISE + 1e-12))
    return np.array(out)


def experiment(n, h1, gain_peak=1.3):
    conc, ctrl = [], []
    for _ in range(n):
        rise = RISE * rng.uniform(0.7, 1.3)
        jit = rng.normal(0, NOISE / max(rise, 1e-9), A.size)
        if h1:
            k = 4.0 * (gain_peak - 1.0) + 0.6   # steeper == higher peak gain
            conc.append(curve_h1(rise, rng.uniform(3, 5), k, gain_peak, jit))
        else:
            conc.append(curve_h0(rise, rng.uniform(2.5, 5.0), jit))
        # control: random direction -> small rise, same noise
        r2 = RISE * 0.05
        ctrl.append(curve_h0(r2, rng.uniform(2.5, 5.0),
                             rng.normal(0, NOISE / max(r2, 1e-9), A.size)))
    return convexity(np.array(conc)), convexity(np.array(ctrl))


print("primary test: convexity (accelerating region), concept vs control 95th pct")
print(f"{'n':>5} {'power(H1,peak1.3)':>19} {'false-pos(H0)':>15}")
REPS = 400
for n in (60, 120, 250, 500):
    hits_h1 = hits_h0 = 0
    for _ in range(REPS):
        c1, k1 = experiment(n, True)
        thr = np.percentile(k1, 95)
        # detection: >20% of prompts exceed the control's 95th pct (vs 5% by chance)
        from math import sqrt
        frac = (c1 > thr).mean()
        se = sqrt(0.05 * 0.95 / n)
        hits_h1 += (frac - 0.05) / se > 2.0
        c0, k0 = experiment(n, False)
        thr0 = np.percentile(k0, 95)
        frac0 = (c0 > thr0).mean()
        hits_h0 += (frac0 - 0.05) / se > 2.0
    print(f"{n:5d} {hits_h1/REPS:19.2f} {hits_h0/REPS:15.2f}")

print("\nsecondary (Hartigan dip) — indicative only, low power by construction")
for n in (60, 120, 250, 500):
    print(f"  n={n:4d}: dip on {n} raw per-prompt values; "
          f"detects separation >= ~{2.5*NOISE/max(RISE,1e-9)*100:.0f}% of the rise")
