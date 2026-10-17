"""Amendment 3 (docs/DESIGN_pregrok.md): control-calibrated onset rule.

Picks the most sensitive (CUSUM, |z|, k) setting on a fixed grid whose false-positive rate on the
no-weight-decay controls is <= 0.20, using the controls only, then applies that single setting to the
grok runs. Behavioural onsets always come from the 100-epoch level-1 logs. The Fourier comparator
(Nanda; task-aware, known to form before the jump) is the power check.

    python tools/pregrok_calibrated.py --family level1  results/grokking/s*_f0.*_log.jsonl
    python tools/pregrok_calibrated.py --family level2  results/grokking/s*_level2_log.jsonl
    python tools/pregrok_calibrated.py --family acts    results/grokking/s*_acts_log.jsonl
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np, pandas as pd
from scipy.stats import binomtest, spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pregrok_analysis import load, behavioural_onsets, structural_onset, MEASURES  # noqa: E402

CLEAN = ['s0_wd1.0_f0.3', 's1_wd0.3_f0.3', 's2_wd0.3_f0.3', 's2_wd1.0_f0.3', 's3_wd1.0_f0.3', 's3_wd3.0_f0.3', 's4_wd0.3_f0.3', 's4_wd1.0_f0.3']
GRID_CUSUM = [8, 16, 32, 64, 128, 256]
GRID_Z = [3, 4, 6, 8, 12, 16]


def run_name(path):
    b = os.path.basename(path)
    for suf in ("_level2_log.jsonl", "_acts_log.jsonl", "_log.jsonl"):
        if b.endswith(suf):
            return b[: -len(suf)]
    return b


def arm(run):
    wd = float(run.split("_wd")[1].split("_")[0]); f = float(run.split("_f")[-1])
    return "fast" if f == 0.6 else ("wd0" if wd == 0 else "grok")


def fourier_series(path):
    rows = [r for r in (json.loads(l) for l in open(path)) if "resumed_from" not in r]
    f = rows[0]["layers"].get("fourier", {})
    keys = [k for k in f if k.startswith("gini")]
    return {k: np.array([r["layers"]["fourier"][k] for r in rows], float) for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--family", required=True, choices=["level1", "level2", "acts"])
    ap.add_argument("--level1-dir", default="results/grokking")
    ap.add_argument("--fp-max", type=float, default=0.20)
    a = ap.parse_args()
    lo, hi = 2000, 10000
    ks = [10, 20] if a.family == "level1" else [4, 8]
    data = {}
    for p in a.logs:
        run = run_name(p)
        if not run.startswith("s") or "_wd" not in run:
            continue
        beh, mats = load(p)
        l1 = os.path.join(a.level1_dir, f"{run}_log.jsonl")
        beh1, _ = load(l1) if os.path.exists(l1) else (beh, None)
        b1, b2 = behavioural_onsets(beh1, lo, hi)
        data[run] = {"ep": beh.epoch.values, "mats": mats, "B1": b1, "arm": arm(run), "fourier": fourier_series(p)}
    ctrl = [r for r in data if data[r]["arm"] == "wd0"]
    grok = [r for r in data if data[r]["arm"] == "grok" and data[r]["B1"] is not None]
    print(f"family {a.family}: runs {len(data)}, no-decay controls {len(ctrl)}, grok runs with B1 {len(grok)}")

    def onsets(runs, cus, z, k):
        out = {}
        for r in runs:
            d = data[r]
            for name, mm in d["mats"].items():
                for m in MEASURES:
                    out[(r, name, m)] = structural_onset(d["ep"], mm[m], lo, hi, cusum_thr=cus, z_thr=z, k=k)
            for fk, series in d["fourier"].items():
                out[(r, "fourier", fk)] = structural_onset(d["ep"], series, lo, hi, cusum_thr=cus, z_thr=z, k=k)
        return out

    def fp_rate(on, runs):
        df = pd.DataFrame([(r, n, m, v) for (r, n, m), v in on.items()], columns=["run", "matrix", "measure", "on"])
        df = df[df.measure.isin(MEASURES) & (df.matrix != "fourier")]
        g = df.groupby(["matrix", "measure"]).on.apply(lambda s: s.notna().mean())
        return float(g.mean()), g

    # calibration on controls only, grid order = most sensitive first
    chosen = None
    for k in ks:
        for cus in GRID_CUSUM:
            for z in GRID_Z:
                rate, _ = fp_rate(onsets(ctrl, cus, z, k), ctrl)
                if chosen is None:
                    first = (cus, z, k, rate)
                if rate <= a.fp_max:
                    chosen = (cus, z, k, rate); break
            if chosen: break
        if chosen: break
    if chosen is None:
        print("no grid setting brings the control false-positive rate to", a.fp_max, "- the sweep cannot be read with this rule family"); return
    cus, z, k, rate = chosen
    print(f"calibrated setting: CUSUM > {cus}, |z| > {z}, k = {k} logs -> control false-positive rate {rate:.2f} (uncalibrated rule: {first[3]:.2f})")

    # apply once to the grok runs
    on_g = onsets(grok, cus, z, k)
    _, fp_pairs = fp_rate(onsets(ctrl, cus, z, k), ctrl)
    rows = []
    for (r, n, m), v in on_g.items():
        rows.append({"run": r, "matrix": n, "measure": m, "onset": v, "B1": data[r]["B1"], "lead": (None if v is None else data[r]["B1"] - v), "clean": r in CLEAN})
    df = pd.DataFrame(rows)
    out = []
    for (n, m), g in df.groupby(["matrix", "measure"]):
        lead = g.lead.dropna(); nn = len(lead)
        if nn < 3:
            continue
        pos = int((lead > 0).sum()); p = binomtest(pos, nn, 0.5, alternative="greater").pvalue
        gc = g.dropna(subset=["onset"]); rho = spearmanr(gc.onset, gc.B1).correlation if len(gc) >= 4 else float("nan")
        cl = g[g.clean].lead.dropna()
        out.append({"matrix": n, "measure": m, "fired": nn, "of": len(g), "lead_pos": pos, "median_lead": float(lead.median()), "sign_p": round(p, 3),
                    "spearman": round(rho, 2) if rho == rho else rho, "clean_pos": f"{int((cl > 0).sum())}/{len(cl)}",
                    "ctrl_FP": round(float(fp_pairs.get((n, m), np.nan)), 2) if n != "fourier" else "n/a"})
    res = pd.DataFrame(out).sort_values(["lead_pos", "median_lead"], ascending=[False, False])
    print("\npower check (task-aware Fourier comparator):")
    print(res[res.matrix == "fourier"].to_string(index=False))
    print(f"\nall pairs under the calibrated rule (grok runs with B1 = {len(grok)}; a pair that never fires has fired=0):")
    print(res[res.matrix != "fourier"].head(30).to_string(index=False))
    res.to_csv(f"results/grokking/pregrok_calibrated_{a.family}.csv", index=False)


if __name__ == "__main__":
    main()
