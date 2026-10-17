"""Amendment 5 (docs/DESIGN_pregrok.md): top-k energy share from the logs and top-k subspace persistence
from the checkpoints, with the same-decay non-grokked runs as the null.

    python tools/pregrok_topk.py persist runs/grokking/s1_wd0.3_f0.3 [--k 4 8] [--out results/grokking/s1_wd0.3_f0.3_persist.jsonl]
    python tools/pregrok_topk.py analyse results/grokking   # energy share from *_log.jsonl + persistence files if present
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MATS = ["block.q.weight", "block.k.weight", "block.v.weight", "block.o.weight", "block.up.weight", "block.down.weight", "unembed.weight"]


def persist(run_dir, ks, out):
    import torch
    ckpts = sorted(c for c in glob.glob(os.path.join(run_dir, "ckpt", "ep*.pt")) if not c.endswith(".opt"))
    prev = {}
    with open(out, "w") as f:
        for ck in ckpts:
            ep = int(os.path.basename(ck)[2:8]); sd = torch.load(ck, map_location="cpu")
            rec = {"epoch": ep, "layers": {}}
            for n in MATS:
                W = sd[n].double().numpy()
                U, s, Vt = np.linalg.svd(W, full_matrices=False)
                d = {}
                for k in ks:
                    if n in prev:
                        Up, Vtp = prev[n]
                        d[f"P{k}_left"] = float(np.linalg.norm(U[:, :k].T @ Up[:, :k]) ** 2 / k)
                        d[f"P{k}_right"] = float(np.linalg.norm(Vt[:k] @ Vtp[:k].T) ** 2 / k)
                    d[f"E{k}"] = float((s[:k] ** 2).sum() / (s ** 2).sum())
                prev[n] = (U, Vt)
                rec["layers"][n] = d
            f.write(json.dumps(rec) + "\n")
    print(out)


def load_log(path):
    rows = [r for r in (json.loads(l) for l in open(path)) if "layers" in r]
    ep = np.array([r["epoch"] for r in rows], float)
    beh = {"test_loss": np.array([r["test_loss"] for r in rows], float), "test_acc": np.array([r["test_acc"] for r in rows], float)}
    series = {}
    for n in MATS:
        for k in (2, 4, 8):
            e = []
            for r in rows:
                L = r["layers"][n]; sv = np.array(L["top_sv"], float); fro = L.get("fro")
                e.append((sv[:k] ** 2).sum() / fro ** 2 if fro else np.nan)
            series[(n, f"E{k}")] = np.array(e)
        series[(n, "fro")] = np.array([r["layers"][n].get("fro", np.nan) for r in rows], float)
    return ep, beh, series


def load_persist(path):
    rows = [json.loads(l) for l in open(path)]
    ep = np.array([r["epoch"] for r in rows], float); series = {}
    for n in MATS:
        for key in rows[-1]["layers"][n]:
            if key.startswith("P"):
                series[(n, key)] = np.array([r["layers"][n].get(key, np.nan) for r in rows], float)
    return ep, series


def separation(ep, x, ctrl, lo=2000, hi=10000, k=10):
    """First epoch from which x stays outside the band of the control runs (min..max at the same epoch,
    widened by the controls' plateau spread) for k consecutive logs."""
    C = np.vstack(ctrl); band_lo, band_hi = C.min(0), C.max(0)
    pl = (ep >= lo) & (ep <= hi); w = float(np.nanstd(C[:, pl] - C[:, pl].mean(0))) if pl.sum() else 0.0
    out = (x > band_hi + w) | (x < band_lo - w)
    post = np.where(ep > hi)[0]
    for j, i in enumerate(post):
        if j + k <= len(post) and out[post[j:j + k]].all():
            return float(ep[i])
    return None


def analyse(resdir):
    import pandas as pd
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pregrok_analysis import load as load1, behavioural_onsets
    B1 = {}
    for p in glob.glob(os.path.join(resdir, "s*_wd0.3_f0.3_log.jsonl")):
        run = os.path.basename(p).replace("_log.jsonl", ""); beh, _ = load1(p); b1, b2 = behavioural_onsets(beh, 2000, 10000); B1[run] = (b1, b2)
    grok = sorted(r for r, (b1, b2) in B1.items() if b1 is not None and b2 is not None)
    ctrl = sorted(r for r, (b1, b2) in B1.items() if b2 is None)   # never above 5 per cent accuracy: the same-decay null
    B1 = {r: b[0] for r, b in B1.items()}
    print("wd0.3 grokked:", {r: B1[r] for r in grok}, "| non-grokked (null):", ctrl)
    rows = []
    for kind, loader, pat in (("E/fro (logs)", load_log, "{}_log.jsonl"), ("P (checkpoints)", load_persist, "{}_persist.jsonl")):
        data = {}
        for r in grok + ctrl:
            p = os.path.join(resdir, pat.format(r))
            if os.path.exists(p):
                d = loader(p); data[r] = (d[0], d[-1])
        if len([r for r in ctrl if r in data]) < 2:
            print(f"{kind}: persistence files missing for the null runs; skipped"); continue
        ep = data[ctrl[0]][0]
        for key in data[ctrl[0]][1]:
            C = [data[r][1][key] for r in ctrl if r in data]
            for r in grok:
                if r not in data or key not in data[r][1]: continue
                x = data[r][1][key]; n = min(len(x), len(ep), *[len(c) for c in C])
                sep = separation(ep[:n], x[:n], [c[:n] for c in C])
                rows.append({"kind": kind, "matrix": key[0], "measure": key[1], "run": r, "B1": B1[r], "separation": sep, "lead": (None if sep is None else B1[r] - sep)})
    df = pd.DataFrame(rows)
    if not len(df): return
    g = df.groupby(["kind", "matrix", "measure"]).agg(n=("run", "count"), separated=("separation", lambda s: int(s.notna().sum())),
                                                     lead_ge5k=("lead", lambda s: int((s.dropna() >= 5000).sum())), median_lead=("lead", "median")).reset_index()
    g = g.sort_values(["lead_ge5k", "separated", "median_lead"], ascending=[False, False, False])
    print(g.head(40).to_string(index=False))
    df.to_csv(os.path.join(resdir, "pregrok_topk_separation.csv"), index=False)
    print("\nper-run detail for the top pairs:")
    for _, row in g.head(6).iterrows():
        sub = df[(df.kind == row.kind) & (df.matrix == row.matrix) & (df.measure == row.measure)]
        print(f"  {row.matrix} {row.measure}: " + "; ".join(f"{r.run} sep {r.separation} B1 {r.B1} lead {r.lead}" for r in sub.itertuples()))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["persist", "analyse"]); ap.add_argument("path")
    ap.add_argument("--k", type=int, nargs="+", default=[4, 8]); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.cmd == "persist":
        persist(a.path, a.k, a.out or os.path.join(a.path, "persist.jsonl"))
    else:
        analyse(a.path)


if __name__ == "__main__":
    main()
