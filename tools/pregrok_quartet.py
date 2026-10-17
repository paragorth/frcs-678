"""The quartet test (17 Oct 2026): split each trained weight matrix by the permutation-null edge into the
part above the edge (few strong directions) and the part below (the bulk), and ask which part solves the
task. Prediction written before running: after the grok the above-edge part alone solves it and the
below-edge part alone does not; before the grok neither does; in the never-grok control neither ever does.

    python tools/pregrok_quartet.py <ckpt_dir> --seed 0 --frac 0.3 [--epochs 10000 20000 ...]
"""
from __future__ import annotations
import argparse, glob, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from semataxy.structure import spectrum, shuffled  # noqa: E402
from tools.grokking import Model  # noqa: E402

MATS = ["block.q.weight", "block.k.weight", "block.v.weight", "block.o.weight", "block.up.weight", "block.down.weight", "unembed.weight"]


def split_by_edge(W, rng, n_shuffles=4, tol=1.02):
    """W = above + below, where 'above' keeps the singular directions whose spectrum entry exceeds the shuffled edge."""
    W64 = np.asarray(W, np.float64)
    U, s, Vt = np.linalg.svd(W64, full_matrices=False)
    n = max(W64.shape)
    lam = s ** 2 / n
    edge = max(spectrum(shuffled(W64, rng))[0] for _ in range(n_shuffles)) * tol
    keep = lam > edge
    above = (U[:, keep] * s[keep]) @ Vt[keep]
    return above, W64 - above, int(keep.sum()), len(s)


def accuracy(model, pairs, labels, idx):
    with torch.no_grad():
        out = model(pairs[idx]).argmax(-1)
        return float((out == labels[idx]).float().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_dir"); ap.add_argument("--seed", type=int, required=True); ap.add_argument("--frac", type=float, default=0.3)
    ap.add_argument("--p", type=int, default=113); ap.add_argument("--epochs", type=int, nargs="*", default=None)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    p = a.p
    pairs = torch.tensor([(i, j) for i in range(p) for j in range(p)]); labels = (pairs[:, 0] + pairs[:, 1]) % p
    perm = torch.randperm(len(pairs)); n_train = int(a.frac * len(pairs)); tr, te = perm[:n_train], perm[n_train:]
    rng = np.random.default_rng(0)
    ckpts = sorted(c for c in glob.glob(os.path.join(a.ckpt_dir, "ep*.pt")) if not c.endswith(".opt"))
    print(f"{'epoch':>6} | {'full tr/te':>12} | {'above-edge tr/te':>17} | {'below-edge tr/te':>17} | kept directions per matrix")
    for ck in ckpts:
        ep = int(os.path.basename(ck)[2:8])
        if a.epochs and ep not in a.epochs:
            continue
        sd = torch.load(ck, map_location="cpu")
        full = Model(p); full.load_state_dict(sd); full.eval()
        sd_above, sd_below, kept = dict(sd), dict(sd), []
        for n in MATS:
            above, below, k, tot = split_by_edge(sd[n].numpy(), rng)
            sd_above[n] = torch.tensor(above, dtype=sd[n].dtype); sd_below[n] = torch.tensor(below, dtype=sd[n].dtype)
            kept.append(f"{n.split('.')[-2] if 'block' in n else 'unembed'}:{k}/{tot}")
        ma = Model(p); ma.load_state_dict(sd_above); ma.eval()
        mb = Model(p); mb.load_state_dict(sd_below); mb.eval()
        f = lambda m: f"{accuracy(m, pairs, labels, tr):.2f}/{accuracy(m, pairs, labels, te):.2f}"
        print(f"{ep:>6} | {f(full):>12} | {f(ma):>17} | {f(mb):>17} | {' '.join(kept)}")


if __name__ == "__main__":
    main()
