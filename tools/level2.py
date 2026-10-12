"""Level-2 structure: organisation ACROSS matrices (virtual weights), not within them.

Per layer of a GPT-NeoX model, the composed linear maps (Elhage et al. 2021 "virtual weights"):
  OV   = W_o @ W_v                 (attention value path, d x d)
  MLP  = W_out @ W_in              (MLP path, d x d)
  XATT = W_qkv[l+1] @ W_o[l]       (attention output of layer l feeding q/k/v of layer l+1, residual path)
  XMLP = W_in[l+1] @ W_out[l]      (MLP output of layer l feeding MLP input of layer l+1)
LayerNorm and nonlinearities are ignored: these are the linear skeletons of the paths.

For two checkpoints (e.g. base vs forgotten), report for factors and for products:
  rel_size_change   = sum|s1 - s0| / sum s0
  retained          = mean squared overlap of singular directions, by band (big / rest)
  Sigma_struct and mass_above (semataxy.structure) for each.

    python tools/level2.py --a EleutherAI/pythia-160m --b runs/live_learning_v2/naive.pt --out results/level2_forgetting.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import semataxy  # noqa: E402,F401
from semataxy.structure import analyze_matrix  # noqa: E402


def mats(model):
    sd = {n: p.detach().double().numpy() for n, p in model.named_parameters() if p.ndim == 2}
    L = max(int(n.split(".")[2]) for n in sd if n.startswith("gpt_neox.layers.")) + 1
    d = sd["gpt_neox.layers.0.attention.dense.weight"].shape[0]
    out = {}
    for l in range(L):
        p = f"gpt_neox.layers.{l}."
        qkv, o = sd[p + "attention.query_key_value.weight"], sd[p + "attention.dense.weight"]
        win, wout = sd[p + "mlp.dense_h_to_4h.weight"], sd[p + "mlp.dense_4h_to_h.weight"]
        v = qkv[2 * d:3 * d]  # NeoX packs q,k,v per head; approximate the v block as the last third
        out[f"L{l}.W_o"] = o; out[f"L{l}.W_v"] = v; out[f"L{l}.W_in"] = win; out[f"L{l}.W_out"] = wout
        out[f"L{l}.OV"] = o @ v; out[f"L{l}.MLP"] = wout @ win
        if l + 1 < L:
            q = f"gpt_neox.layers.{l+1}."
            out[f"L{l}>{l+1}.XATT"] = sd[q + "attention.query_key_value.weight"] @ o
            out[f"L{l}>{l+1}.XMLP"] = sd[q + "mlp.dense_h_to_4h.weight"] @ wout
    return out


def compare(A, B, rng):
    rows = []
    for name in A:
        W0, W1 = A[name], B[name]
        if W0.shape[0] < W0.shape[1]:
            W0, W1 = W0.T, W1.T
        U0, s0, _ = np.linalg.svd(W0, full_matrices=False); U1, s1, _ = np.linalg.svd(W1, full_matrices=False)
        st0 = analyze_matrix(name, W0, rng); st1 = analyze_matrix(name, W1, rng)
        k = max(st0.n_above, 1)
        def ret(idx):
            P = U1[:, idx].T @ U0[:, idx]
            return float(np.mean(np.linalg.norm(P, axis=0) ** 2))
        m = len(s0)
        rows.append({"name": name, "kind": "product" if any(t in name for t in ("OV", "MLP", "XATT", "XMLP")) else "factor",
                     "rel_size_change": float(np.abs(s1 - s0).sum() / s0.sum()),
                     "retained_big": ret(np.arange(0, k)), "retained_rest": ret(np.arange(k, m)),
                     "sigma0": st0.sigma_struct, "sigma1": st1.sigma_struct, "mass0": st0.mass_above, "mass1": st1.mass_above})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="EleutherAI/pythia-160m")
    ap.add_argument("--b", required=True, help="state dict path (or HF id) for the second checkpoint")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(1); rng = np.random.default_rng(0)
    ma = AutoModelForCausalLM.from_pretrained(args.a, dtype=torch.float32)
    if os.path.exists(args.b):
        mb = AutoModelForCausalLM.from_pretrained(args.a, dtype=torch.float32); mb.load_state_dict(torch.load(args.b))
    else:
        mb = AutoModelForCausalLM.from_pretrained(args.b, dtype=torch.float32)
    df = compare(mats(ma), mats(mb), rng)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True); df.to_csv(args.out, index=False)
    print(df.groupby("kind")[["rel_size_change", "retained_big", "retained_rest", "sigma0", "sigma1", "mass0", "mass1"]].mean().round(4).to_string())
    print(); print(df.sort_values("retained_rest").head(6)[["name", "rel_size_change", "retained_big", "retained_rest", "sigma0", "sigma1"]].to_string(index=False))


if __name__ == "__main__":
    main()
