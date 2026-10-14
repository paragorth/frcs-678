"""Level-2 Semataxy: organisation BETWEEN matrices, with a cross-layer permutation control.

Level 1 (semataxy/structure.py) looks inside one matrix and is destroyed by shuffling its entries.
Level 2 looks at how matrices relate through the residual stream and is destroyed by shuffling
which layer's matrix follows which, while leaving every matrix intact. Same permutation criterion,
one level up.

Mechanics (Llama/Qwen/Mistral/Gemma/OLMo-HF naming):
  writers to the residual stream   self_attn.o_proj (d x a), mlp.down_proj (d x f)
  readers from the residual stream self_attn.{q,k,v}_proj (. x d), mlp.{gate,up}_proj (f x d)
For a writer W at layer l take the top-k left singular vectors U (d x k): the residual directions it
writes. For a reader R at layer m take the top-k right singular vectors V (d x k): the directions it
reads. Overlap = ||U^T V||_F^2 / k, in [0, 1]: the fraction of what l writes that m reads, for the
dominant k directions. Random k-subspaces of R^d overlap k/d.

Control: the same overlaps with reader layers permuted (so writer l is paired with a reader from a
random layer at the same lag distribution), averaged over several permutations.

Output CSV: one row per (writer kind, reader kind, lag) with actual and shuffled overlap; plus a
summary line. Streams one matrix at a time; CPU only.

    python tools/structure_level2.py --model Qwen/Qwen2.5-7B-Instruct --out results/structure/l2_qwen2.5-7b.csv
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

WRITERS = {"attn_out": "self_attn.o_proj", "mlp_out": "mlp.down_proj"}
READERS = {"q": "self_attn.q_proj", "k": "self_attn.k_proj", "v": "self_attn.v_proj",
           "gate": "mlp.gate_proj", "up": "mlp.up_proj"}
PAT = re.compile(r"model\.layers\.(\d+)\.(self_attn\.[qkvo]_proj|mlp\.(?:gate|up|down)_proj)\.weight$")


def topk_subspace(W: torch.Tensor, k: int, side: str) -> np.ndarray:
    """Top-k left ('rows' of residual, writers) or right (readers) singular subspace, d x k."""
    q = min(k + 32, min(W.shape))
    U, S, V = torch.svd_lowrank(W, q=q, niter=4)
    M = U[:, :k] if side == "left" else V[:, :k]
    return M.contiguous().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--max-lag", type=int, default=8)
    ap.add_argument("--n-perm", type=int, default=5)
    ap.add_argument("--random-init", action="store_true", help="size control: N(0, 0.02) matrices of the same shapes")
    a = ap.parse_args()
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    local = snapshot_download(a.model, revision=a.revision, allow_patterns=["*.safetensors", "*.json"])
    files = sorted(glob.glob(os.path.join(local, "*.safetensors")))
    # collect subspaces: layer -> kind -> (d x k)
    sub: dict[int, dict[str, np.ndarray]] = {}
    d_model = None
    for f in files:
        with safe_open(f, framework="pt") as sf:
            for name in sf.keys():
                m = PAT.match(name)
                if not m:
                    continue
                layer, kind = int(m.group(1)), m.group(2)
                W = sf.get_tensor(name).float()
                if a.random_init:
                    W = torch.randn_like(W) * 0.02
                if kind in WRITERS.values():        # (d, inner): residual directions are rows -> left vectors
                    M = topk_subspace(W, a.k, "left")
                    d_model = W.shape[0]
                else:                               # (inner, d): residual directions are columns -> right vectors
                    M = topk_subspace(W, a.k, "right")
                    d_model = W.shape[1]
                sub.setdefault(layer, {})[kind] = M
                print(f"layer {layer} {kind} done", flush=True)
    layers = sorted(sub)
    L = len(layers)

    def overlap(U, V):
        return float(np.linalg.norm(U.T @ V) ** 2 / a.k)

    rows = []
    for wname, wkey in WRITERS.items():
        for rname, rkey in READERS.items():
            for lag in range(1, a.max_lag + 1):
                actual, shuffled = [], []
                for l in layers:
                    m = l + lag
                    if m not in sub or wkey not in sub[l] or rkey not in sub[m]:
                        continue
                    actual.append(overlap(sub[l][wkey], sub[m][rkey]))
                for _ in range(a.n_perm):
                    perm = rng.permutation(layers)
                    pmap = dict(zip(layers, perm))
                    for l in layers:
                        m = l + lag
                        if m not in sub or wkey not in sub[l]:
                            continue
                        m2 = pmap[m]
                        if rkey not in sub[m2] or m2 == m:
                            continue
                        shuffled.append(overlap(sub[l][wkey], sub[m2][rkey]))
                if actual:
                    rows.append({"model": a.model, "revision": a.revision, "random_init": a.random_init,
                                 "writer": wname, "reader": rname, "lag": lag, "n_pairs": len(actual),
                                 "overlap": float(np.mean(actual)),
                                 "overlap_shuffled": float(np.mean(shuffled)) if shuffled else np.nan,
                                 "chance": a.k / d_model})
    df = pd.DataFrame(rows)
    df["excess"] = df["overlap"] - df["overlap_shuffled"]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    df.to_csv(a.out, index=False)
    lag1 = df[df.lag == 1]
    print(f"\n{a.model}{'@' + a.revision if a.revision else ''}: {L} layers, k={a.k}, chance {a.k / d_model:.3f}")
    print(f"  lag-1 overlap actual {lag1.overlap.mean():.3f}  shuffled {lag1.overlap_shuffled.mean():.3f}  "
          f"excess {lag1.excess.mean():.3f}")
    print(f"  all-lag excess {df.excess.mean():.3f}; by writer: "
          + ", ".join(f"{w} {g.excess.mean():.3f}" for w, g in df.groupby('writer')))


if __name__ == "__main__":
    main()
