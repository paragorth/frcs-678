"""Pre-grokking analysis: the onset rules of docs/DESIGN_pregrok.md, applied to grokking logs.

    python tools/pregrok_analysis.py results/grokking/*_log.jsonl [--plateau 2000 10000] [--out results/grokking/pregrok_summary.csv]

Per run and per (matrix, measure): structural onset epoch; per run: behavioural onsets B1 (test
loss) and B2 (test accuracy); lead = behavioural minus structural. Then across runs: sign test on
the lead and Spearman correlation between structural and behavioural onset, per (matrix, measure);
false-positive rates on control runs (those whose B1 never fires).
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import pandas as pd

MEASURES = ("mass_above", "frac_above", "conc", "alpha", "sigma", "fro", "erank16")


def load(path):
    rows = [json.loads(l) for l in open(path)]
    ep = np.array([r["epoch"] for r in rows], float)
    beh = pd.DataFrame({"epoch": ep, "test_loss": [r["test_loss"] for r in rows], "test_acc": [r["test_acc"] for r in rows],
                        "train_acc": [r["train_acc"] for r in rows]})
    mats = {}
    for name in rows[0]["layers"]:
        d = {}
        for m in ("mass_above", "frac_above", "alpha", "sigma", "fro"):
            d[m] = np.array([r["layers"][name].get(m) if r["layers"][name].get(m) is not None else np.nan for r in rows], float)
        d["conc"] = d["mass_above"] / np.where(d["frac_above"] > 0, d["frac_above"], np.nan)
        # effective rank over the top 16 singular values (Yunis et al. form; full-spectrum version from checkpoints)
        er = []
        for r in rows:
            sv = r["layers"][name].get("top_sv")
            if sv:
                p_ = np.array(sv, float); p_ = p_ / p_.sum(); er.append(float(np.exp(-(p_ * np.log(p_ + 1e-12)).sum())))
            else:
                er.append(np.nan)
        d["erank16"] = np.array(er)
        mats[name] = d
    # total norm across matrices as its own pseudo-matrix
    if all("fro" in d and not np.isnan(d["fro"]).all() for d in mats.values()):
        tot = np.sqrt(np.nansum([d["fro"] ** 2 for d in mats.values()], axis=0))
        mats["_total"] = {m: np.full(len(ep), np.nan) for m in MEASURES}; mats["_total"]["fro"] = tot
    return beh, mats


def rolling_median(x, k=5):
    return pd.Series(x).rolling(k, center=True, min_periods=1).median().values


def behavioural_onsets(beh, lo, hi):
    ep = beh.epoch.values; pl = (ep >= lo) & (ep <= hi)
    tl = rolling_median(beh.test_loss.values); plateau = np.median(tl[pl])
    b1 = None
    below = tl < 0.8 * plateau
    for i in np.where(ep > hi)[0]:
        if below[i:].all():
            b1 = float(ep[i]); break
    acc = beh.test_acc.values; b2 = None
    above = acc > 0.05
    for i in np.where(ep > hi)[0]:
        if above[i:].all():
            b2 = float(ep[i]); break
    return b1, b2


def structural_onset(ep, x, lo, hi, cusum_thr=8.0, z_thr=3.0, k=10):
    x = pd.Series(x).interpolate(limit_direction="both").values
    pl = (ep >= lo) & (ep <= hi)
    if pl.sum() < 10 or np.nanstd(x[pl]) == 0:
        return None
    a, b = np.polyfit(ep[pl], x[pl], 1)           # plateau trend (absorbs weight-decay drift)
    resid = x - (a * ep + b)
    sd = np.std(resid[pl]) + 1e-12
    z = resid / sd
    post = np.where(ep > hi)[0]
    cus = 0.0
    for j, i in enumerate(post):
        cus = max(0.0, cus + abs(z[i]) - 1.0)      # one-sided CUSUM on |z| above its plateau level (~0.8)
        if cus > cusum_thr and j + k <= len(post) and (np.abs(z[post[j:j + k]]) > z_thr).all():
            return float(ep[i])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--plateau", nargs=2, type=float, default=(2000, 10000))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    lo, hi = a.plateau
    rows = []
    for path in a.logs:
        run = os.path.basename(path).replace("_log.jsonl", "")
        beh, mats = load(path)
        b1, b2 = behavioural_onsets(beh, lo, hi)
        ep = beh.epoch.values
        for name, d in mats.items():
            for m in MEASURES:
                so = structural_onset(ep, d[m], lo, hi)
                rows.append({"run": run, "matrix": name, "measure": m, "structural_onset": so, "B1": b1, "B2": b2,
                             "lead_B1": (None if so is None or b1 is None else b1 - so),
                             "lead_B2": (None if so is None or b2 is None else b2 - so)})
        # the norm comparator: how does each spectral onset sit relative to the norm's own onset in that matrix?
        norm_on = {r_["matrix"]: r_["structural_onset"] for r_ in rows if r_["run"] == run and r_["measure"] == "fro"}
        for r_ in rows:
            if r_["run"] == run and r_["measure"] != "fro":
                n_on = norm_on.get(r_["matrix"])
                r_["lead_vs_norm"] = (None if r_["structural_onset"] is None or n_on is None else n_on - r_["structural_onset"])
    df = pd.DataFrame(rows)
    if a.out:
        df.to_csv(a.out, index=False)
    grok = df[df.B1.notna()]; ctrl = df[df.B1.isna()]
    print(f"runs: {df.run.nunique()} (grokked under B1: {grok.run.nunique()}, controls/no-grok: {ctrl.run.nunique()})")
    if len(ctrl):
        fp = ctrl.groupby(["matrix", "measure"]).structural_onset.apply(lambda s: s.notna().mean())
        print(f"control false-positive rate (share of control runs with a structural onset), mean over pairs: {fp.mean():.2f}; worst pairs:")
        print(fp.sort_values(ascending=False).head(5).round(2).to_string())
    if len(grok):
        from scipy.stats import binomtest, spearmanr
        out = []
        for (mat, m), g in grok.groupby(["matrix", "measure"]):
            g = g.dropna(subset=["lead_B1"])
            if len(g) < 3:
                continue
            pos = int((g.lead_B1 > 0).sum()); n = len(g)
            p = binomtest(pos, n, 0.5, alternative="greater").pvalue
            rho = spearmanr(g.structural_onset, g.B1).correlation if n >= 4 else float("nan")
            lvn = g["lead_vs_norm"].dropna() if "lead_vs_norm" in g else pd.Series(dtype=float)
            out.append({"matrix": mat, "measure": m, "n": n, "lead_pos": pos, "median_lead": float(g.lead_B1.median()),
                        "sign_p": p, "spearman_onset_vs_B1": rho,
                        "median_lead_vs_norm": (float(lvn.median()) if len(lvn) else float("nan"))})
        if out:
            res = pd.DataFrame(out).sort_values(["sign_p", "median_lead"], ascending=[True, False])
            print("\nper (matrix, measure), grok runs, lead under B1 (positive = structure first):")
            print(res.round(3).to_string(index=False))
        else:
            print("\nfewer than 3 grok runs: per-run onsets and leads (B1 = test-loss onset, B2 = accuracy onset):")
            print(grok.dropna(subset=["structural_onset"]).sort_values("structural_onset")[["run", "matrix", "measure", "structural_onset", "B1", "B2", "lead_B1", "lead_B2"]].to_string(index=False))
    return df


if __name__ == "__main__":
    main()
