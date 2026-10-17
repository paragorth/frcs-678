"""Level-2 (between-matrix) organisation along the grokking circuit, from saved checkpoints.

The modular-addition circuit (Nanda et al. 2023) is relational: it lives in the products that carry
a token through the network, not in any one matrix. For each checkpoint this computes the composed
matrices of the main paths and measures them against a level-2 null that destroys only the relation
between the factors (one factor's entries shuffled), leaving each factor's own spectrum intact:

    QK      W_E W_Q W_K^T W_E^T      token-token attention logits (p x p)
    OV      W_E W_V W_O W_U           the attention path to the logits (p x p)
    MLP_in  W_E W_up                  what the MLP reads from each token (p x 4d)
    MLP_out W_down W_U                what the MLP writes to the logits (4d x p)
    MLP     W_E W_up W_down W_U       the full MLP path (p x p)

For each path: mass above the null edge, fraction above, concentration, tail exponent, span, the
excess of each over the shuffled-factor null, effective rank, Frobenius norm, top singular values;
plus Nanda's weight-only comparator, the Gini of per-frequency norms of W_E and W_U under a DFT over
the token axis. Output is a log in the same shape as tools/grokking.py's, so tools/pregrok_analysis.py
runs on it unchanged (each path is a "matrix"), with test loss and accuracy merged in from the run's
own log at the same epoch.

    python tools/pregrok_level2.py runs/grokking/s0_wd1.0_f0.3 [--p 113] [--n-shuffles 4] [--out results/grokking/s0_wd1.0_f0.3_level2_log.jsonl]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from semataxy.structure import analyze_matrix  # noqa: E402


def gini(x):
    x = np.sort(np.asarray(x, float)); n = len(x)
    if x.sum() == 0:
        return 0.0
    return float((2 * np.sum(np.arange(1, n + 1) * x) / (n * x.sum())) - (n + 1) / n)


def fourier_gini(W_tok):
    """Nanda et al.: DFT over the token axis of an embedding-like matrix, norm per frequency, Gini."""
    F = np.fft.rfft(W_tok, axis=0)            # (p//2+1, d)
    norms = np.linalg.norm(F, axis=1)
    return gini(norms), norms


def stats(name, M, rng, n_shuffles, null_factors):
    """Level-1 stats of the product M, and of the same product with one factor shuffled (level-2 null)."""
    real = analyze_matrix(name, M, rng)
    nulls = []
    for _ in range(n_shuffles):
        nulls.append(analyze_matrix(name, null_factors(rng), rng))
    def mean(attr):
        v = [getattr(s, attr) for s in nulls if getattr(s, attr) is not None and not (isinstance(getattr(s, attr), float) and np.isnan(getattr(s, attr)))]
        return float(np.mean(v)) if v else float("nan")
    sv = np.linalg.svd(np.asarray(M, np.float64), compute_uv=False)
    p_ = sv / sv.sum()
    out = {"mass_above": real.mass_above, "frac_above": real.frac_above, "alpha": real.alpha, "sigma": real.sigma_struct,
           "octaves_total": real.octaves_total,
           "null_mass_above": mean("mass_above"), "null_frac_above": mean("frac_above"), "null_sigma": mean("sigma_struct"),
           "excess_mass": real.mass_above - mean("mass_above"), "excess_sigma": real.sigma_struct - mean("sigma_struct"),
           "erank": float(np.exp(-(p_ * np.log(p_ + 1e-12)).sum())), "fro": float(np.linalg.norm(M)),
           "top_sv": [float(x) for x in sv[:16]]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--p", type=int, default=113)
    ap.add_argument("--n-shuffles", type=int, default=4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    ckpts = sorted(glob.glob(os.path.join(a.run_dir, "ckpt", "ep*.pt")))
    if not ckpts:
        sys.exit(f"no checkpoints under {a.run_dir}/ckpt")
    beh = {}
    logp = os.path.join(a.run_dir, "log.jsonl")
    if os.path.exists(logp):
        for l in open(logp):
            r = json.loads(l); beh[int(r["epoch"])] = r
    out = a.out or os.path.join(a.run_dir, "level2_log.jsonl")
    with open(out, "w") as f:
        for ck in ckpts:
            ep = int(os.path.basename(ck)[2:8])
            sd = torch.load(ck, map_location="cpu")
            g = lambda k: sd[k].double().numpy()
            E = g("tok.weight")[: a.p]            # tokens 0..p-1 (drop '=')
            Q, K, V, O = g("block.q.weight").T, g("block.k.weight").T, g("block.v.weight").T, g("block.o.weight").T  # x @ W
            Up, Down = g("block.up.weight").T, g("block.down.weight").T
            U = g("unembed.weight").T             # (d, p)
            def shuf(M):
                flat = M.flatten().copy(); rng.shuffle(flat); return flat.reshape(M.shape)
            paths = {
                "QK":      (E @ Q @ K.T @ E.T,    lambda r: E @ shuf(Q) @ K.T @ E.T),
                "OV":      (E @ V @ O @ U,        lambda r: E @ shuf(V) @ O @ U),
                "MLP_in":  (E @ Up,               lambda r: E @ shuf(Up)),
                "MLP_out": (Down @ U,             lambda r: shuf(Down) @ U),
                "MLP":     (E @ Up @ Down @ U,    lambda r: E @ shuf(Up) @ Down @ U),
            }
            layers = {name: stats(name, M, rng, a.n_shuffles, nf) for name, (M, nf) in paths.items()}
            gE, _ = fourier_gini(E); gU, _ = fourier_gini(U.T)
            layers["fourier"] = {"mass_above": float("nan"), "frac_above": float("nan"), "alpha": float("nan"), "sigma": float("nan"),
                                 "gini_E": gE, "gini_U": gU, "fro": float(np.linalg.norm(E)), "top_sv": []}
            rec = {"epoch": ep, "layers": layers,
                   "test_loss": beh.get(ep, {}).get("test_loss"), "test_acc": beh.get(ep, {}).get("test_acc"),
                   "train_acc": beh.get(ep, {}).get("train_acc"), "train_loss": beh.get(ep, {}).get("train_loss")}
            f.write(json.dumps(rec) + "\n"); f.flush()
            print(f"ep {ep:6d} test_acc {rec['test_acc']} | QK mass {layers['QK']['mass_above']:.3f} excess {layers['QK']['excess_mass']:+.3f} erank {layers['QK']['erank']:.1f} | OV excess {layers['OV']['excess_mass']:+.3f} | MLP excess {layers['MLP']['excess_mass']:+.3f} | giniE {gE:.3f}", flush=True)
    print(out)


if __name__ == "__main__":
    main()
