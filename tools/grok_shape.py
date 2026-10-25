"""Shape of the organisation across a grokking run, and whether the grok replaces or refines it.

Two questions, one pass over the checkpoints (both need the same SVDs).

1. SHAPE vs WIDTH (20 Oct 2026). On the published trajectory sigma_struct rises during pure
   memorisation (0 -> 0.714 by epoch 400, while test accuracy is 0.0028), peaks at the grok, then
   falls to 0.501 -- below its own memorisation plateau of 1.100 -- and never settles
   (post-grok spread/mean 2.28). frac_above ends exactly where memorisation left it (0.067 -> 0.067).
   Both are width measures. Parag's point: semataxy is a claim about shape, and within one space
   shape is all that varies. So this reports the null-native shape statistics added to
   semataxy.structure -- alpha_above (Hill exponent with xmin = the permutation edge),
   entropy_above and gini_above (how the above-edge mass is spread over its own directions) -- and
   asks whether any of them separates memorisation from generalisation, which sigma does not.

2. REPLACEMENT vs REFINEMENT. Parag's hypothesis is that grokking is new associations replacing old
   memory. The direction counts (+61% then -33%) are consistent with that but cannot distinguish it
   from refinement, because they never ask whether the surviving directions are the SAME ones.
   Principal angles between the above-edge subspace at each checkpoint and at a reference
   checkpoint answer it: a near-orthogonal post-grok subspace is replacement, a nested one is
   refinement. Reported as the mean cosine of the principal angles (the "subspace overlap":
   1 = identical span, 0 = orthogonal) and the fraction of pre-grok directions retained.

    python tools/grok_shape.py runs/grok_ckpt/ckpt --ref 40000 --out results/grokking/shape.csv
"""
from __future__ import annotations
import argparse, csv, glob, math, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from semataxy.structure import (alpha_above_edge, hill_alpha, mass_shape,  # noqa: E402
                                shuffled, spectrum)

MATS = ["block.q.weight", "block.k.weight", "block.v.weight", "block.o.weight",
        "block.up.weight", "block.down.weight", "unembed.weight"]


def above_edge(W, rng, n_shuffles=4, tol=1.02):
    """Right-singular basis of the above-edge subspace, plus the above-edge eigenvalues.

    n_shuffles and tol are carried over from tools/pregrok_quartet.py so that the kept-direction
    count matches what the quartet test used. Both are undocumented fudge factors there and the
    count is sensitive to them; --tol and --shuffles expose them here so the sensitivity is visible.
    """
    W64 = np.asarray(W, np.float64)
    if W64.shape[0] < W64.shape[1]:
        W64 = W64.T
    n = W64.shape[0]
    U, s, Vt = np.linalg.svd(W64, full_matrices=False)
    lam = s ** 2 / n
    edge = max(spectrum(shuffled(W64, rng))[0] for _ in range(n_shuffles)) * tol
    keep = lam > edge
    return Vt[keep], lam[keep], float(edge), lam


def overlap(A, B, retain_cos=0.9):
    """Principal angles between the row spaces of A (current) and B (reference).

    Returns (mean cosine, retained fraction). The mean cosine is the overall subspace overlap:
    1 = identical span, 0 = orthogonal. The retained fraction is the share of B's directions that
    still lie in A's span, counted as principal angles with cosine above retain_cos -- this is the
    quantity that distinguishes replacement (low) from refinement (high) when the subspace shrinks,
    since a shrinking span lowers the mean cosine either way.
    """
    if A is None or B is None or len(A) == 0 or len(B) == 0:
        return float("nan"), float("nan")
    sv = np.clip(np.linalg.svd(A @ B.T, compute_uv=False), 0.0, 1.0)
    return float(np.mean(sv)), float(np.sum(sv > retain_cos) / len(B))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_dir")
    ap.add_argument("--ref", type=int, default=40000,
                    help="reference epoch for the subspace comparison (default: late memorisation)")
    ap.add_argument("--out", default="results/grokking/shape.csv")
    ap.add_argument("--tol", type=float, default=1.02)
    ap.add_argument("--shuffles", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    cks = sorted(c for c in glob.glob(os.path.join(a.ckpt_dir, "ep*.pt")) if not c.endswith(".opt"))
    if not cks:
        sys.exit(f"no checkpoints in {a.ckpt_dir}")
    epochs = [int(os.path.basename(c)[2:8]) for c in cks]
    ref_ep = min(epochs, key=lambda e: abs(e - a.ref))
    print(f"{len(cks)} checkpoints, epochs {epochs[0]}-{epochs[-1]}; reference = {ref_ep}")

    ref_basis = {}
    for ck, ep in zip(cks, epochs):
        if ep != ref_ep:
            continue
        sd = torch.load(ck, map_location="cpu")
        for m in MATS:
            V, _, _, _ = above_edge(sd[m].numpy(), np.random.default_rng(a.seed),
                                    a.shuffles, a.tol)
            ref_basis[m] = V

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    cols = ["epoch", "n_above", "mass_above", "sigma_proxy_octaves", "alpha_hill",
            "alpha_above", "entropy_above", "gini_above", "overlap_with_ref", "retained_frac"]
    rows = []
    for ck, ep in zip(cks, epochs):
        sd = torch.load(ck, map_location="cpu")
        per = []
        for m in MATS:
            V, above, edge, lam = above_edge(sd[m].numpy(), np.random.default_rng(a.seed),
                                             a.shuffles, a.tol)
            ent, gini = mass_shape(above)
            k = len(above)
            ov, ret = overlap(V, ref_basis.get(m))
            per.append(dict(
                n_above=k,
                mass_above=float(above.sum() / lam.sum()) if lam.sum() > 0 else 0.0,
                octaves=math.log2(lam[0] / edge) if k else 0.0,
                alpha_hill=hill_alpha(lam, k=max(5, min(k, len(lam) // 10))),
                alpha_above=alpha_above_edge(above, edge),
                entropy_above=ent, gini_above=gini, ov=ov, ret=ret))
        def mean(key):
            v = [p[key] for p in per if not (isinstance(p[key], float) and math.isnan(p[key]))]
            return sum(v) / len(v) if v else float("nan")
        row = dict(epoch=ep, n_above=mean("n_above"), mass_above=mean("mass_above"),
                   sigma_proxy_octaves=mean("octaves"), alpha_hill=mean("alpha_hill"),
                   alpha_above=mean("alpha_above"), entropy_above=mean("entropy_above"),
                   gini_above=mean("gini_above"), overlap_with_ref=mean("ov"),
                   retained_frac=mean("ret"))
        rows.append(row)
        print(f"  ep {ep:>6}  k={row['n_above']:>6.1f}  mass={row['mass_above']:.3f}  "
              f"a_hill={row['alpha_hill']:>5.2f}  a_above={row['alpha_above']:>5.2f}  "
              f"ent={row['entropy_above']:.3f}  gini={row['gini_above']:.3f}  "
              f"overlap={row['overlap_with_ref']:.3f}", flush=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
