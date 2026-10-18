"""Bimodality analysis of the margin distributions.

Primary statistic is Hartigan's dip (computed on the pod, where the wheel
builds). This module implements Silverman's critical-bandwidth bootstrap as an
independent check that runs anywhere, plus a 1-vs-2 component Gaussian mixture
BIC comparison. Three tests that fail in different ways is worth more than one.

Null being tested at each alpha: the margin distribution across prompts has a
single mode. Rejecting it means trials are sitting either side of a crossing,
which is what a competitive bottleneck would produce. Failing to reject means
the distribution slides rather than splits -- smooth saturation.
"""
from __future__ import annotations
import gzip, json, math, sys
from collections import defaultdict
import numpy as np


def kde(x, h, grid):
    z = (grid[:, None] - x[None, :]) / h
    return np.exp(-0.5 * z * z).sum(1) / (len(x) * h * math.sqrt(2 * math.pi))


def n_modes(x, h, grid=None):
    if grid is None:
        pad = 3 * h
        grid = np.linspace(x.min() - pad, x.max() + pad, 512)
    d = kde(x, h, grid)
    return int(((d[1:-1] > d[:-2]) & (d[1:-1] >= d[2:])).sum())


def critical_bandwidth(x, kmax=1, lo=1e-4, hi=None, iters=60):
    """Smallest h whose KDE has <= kmax modes."""
    hi = hi or (x.std() * 3 + 1e-9)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if n_modes(x, mid) > kmax:
            lo = mid
        else:
            hi = mid
    return hi


def silverman_test(x, kmax=1, B=200, rng=None):
    """Silverman (1981). Bootstrap from the KDE at the critical bandwidth;
    p = fraction of resamples needing a bandwidth at least as large."""
    rng = rng or np.random.default_rng(0)
    x = np.asarray(x, float)
    hc = critical_bandwidth(x, kmax)
    sd = x.std(ddof=1)
    ge = 0
    for _ in range(B):
        idx = rng.integers(0, len(x), len(x))
        y = x[idx] + hc * rng.standard_normal(len(x))
        y = x.mean() + (y - y.mean()) / math.sqrt(1 + hc * hc / (sd * sd))
        if critical_bandwidth(y, kmax) >= hc:
            ge += 1
    return hc, (ge + 1) / (B + 1)


def gmm_bic(x, rng=None, iters=300):
    """BIC for 1- vs 2-component Gaussian mixture (EM). Negative delta favours 2."""
    rng = rng or np.random.default_rng(0)
    x = np.asarray(x, float); n = len(x)
    ll1 = -0.5 * n * (math.log(2 * math.pi * x.var()) + 1)
    bic1 = -2 * ll1 + 2 * math.log(n)
    mu = np.array([np.percentile(x, 25), np.percentile(x, 75)])
    sd = np.array([x.std(), x.std()]) + 1e-9
    w = np.array([0.5, 0.5])
    for _ in range(iters):
        p = w * np.exp(-0.5 * ((x[:, None] - mu) / sd) ** 2) / (sd * math.sqrt(2 * math.pi))
        tot = p.sum(1, keepdims=True) + 1e-300
        r = p / tot
        nk = r.sum(0) + 1e-12
        w = nk / n
        mu = (r * x[:, None]).sum(0) / nk
        sd = np.sqrt((r * (x[:, None] - mu) ** 2).sum(0) / nk) + 1e-9
    ll2 = float(np.log((w * np.exp(-0.5 * ((x[:, None] - mu) / sd) ** 2)
                        / (sd * math.sqrt(2 * math.pi))).sum(1) + 1e-300).sum())
    bic2 = -2 * ll2 + 5 * math.log(n)
    return bic1, bic2, mu, w


def load(path, condition="concept"):
    out = defaultdict(list)
    with gzip.open(path, "rt") as fh:
        for line in fh:
            r = json.loads(line)
            if r["condition"] == condition:
                out[r["alpha"]].append(r["margin"])
    return {a: np.array(v) for a, v in out.items()}


if __name__ == "__main__":
    path = sys.argv[1]
    dips = {}
    try:
        import diptest
        HAVE_DIP = True
    except ImportError:
        HAVE_DIP = False
    for cond in ("concept", "random_matched"):
        data = load(path, cond)
        alphas = sorted(a for a in data if a is not None)
        print(f"\n=== {cond} ===")
        hdr = f"{'alpha':>7} {'n':>5} {'mean':>9} {'sd':>7} {'silv_p':>7} {'dBIC':>9} {'w2':>6}"
        print(hdr + (f" {'dip':>7} {'dip_p':>7}" if HAVE_DIP else ""))
        for a in alphas:
            x = data[a]
            _, p = silverman_test(x, B=200)
            b1, b2, mu, w = gmm_bic(x)
            line = (f"{a:7.2f} {len(x):5d} {x.mean():9.3f} {x.std():7.3f} "
                    f"{p:7.3f} {b2-b1:9.1f} {max(w):6.2f}")
            if HAVE_DIP:
                d, dp = diptest.diptest(x); line += f" {d:7.4f} {dp:7.3f}"
            print(line)
        print("  dBIC < 0 favours two components; silv_p < 0.05 rejects unimodality")
