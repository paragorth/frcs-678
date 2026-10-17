"""Organisation of the *representation* (activations on the task's inputs), from saved checkpoints.

The weight-only measures (tools/grokking.py, tools/pregrok_level2.py) are world-blind: a permutation
null on a weight matrix cannot tell "organised about the task" from "organised at all". Semataxy is
organisation relative to the environment, so here the environment does the pointing: every
checkpoint sees a fixed sample of the task's inputs, and the same shuffle-null spectral measures are
applied to the activation matrices (inputs x units) they produce at the prediction position:

    attn_out   what attention writes into the residual stream        (N x d)
    resid_mid  residual stream after attention                        (N x d)
    mlp_hid    MLP hidden layer after ReLU                            (N x 4d)
    mlp_out    what the MLP writes                                    (N x d)
    resid_post residual stream before the unembedding                 (N x d)
    logits     the output                                             (N x p)

Null: shuffle the entries of the activation matrix (same values, arrangement destroyed). Reported
per matrix: mass above the edge, fraction above, concentration, tail exponent, span, effective rank,
Frobenius norm, top singular values; plus the Nanda-style task-aware comparator, the Gini of
per-frequency norms of the mean logit vector by (a+b) mod p. Output has the shape of
tools/grokking.py's log so tools/pregrok_analysis.py runs on it unchanged.

    python tools/pregrok_acts.py runs/grokking/s0_wd1.0_f0.3 [--n-inputs 4096] [--n-shuffles 4] [--device cuda]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from semataxy.structure import analyze_matrix  # noqa: E402
from tools.grokking import Model  # noqa: E402
from tools.pregrok_level2 import gini  # noqa: E402


def activations(model: Model, ab: torch.Tensor):
    """Forward pass with the intermediate values at the prediction position (the '=' token)."""
    blk = model.block
    B = ab.shape[0]
    eq = torch.full((B, 1), model.tok.num_embeddings - 1, dtype=torch.long, device=ab.device)
    x = torch.cat([ab, eq], 1)
    x = model.tok(x) + model.pos(torch.arange(3, device=ab.device))[None]
    T, D, h = 3, x.shape[-1], blk.heads
    q = blk.q(x).view(B, T, h, D // h).transpose(1, 2)
    k = blk.k(x).view(B, T, h, D // h).transpose(1, 2)
    v = blk.v(x).view(B, T, h, D // h).transpose(1, 2)
    att = (q @ k.transpose(-1, -2)) / (D // h) ** 0.5
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
    att = att.masked_fill(mask, float("-inf")).softmax(-1)
    attn_out = blk.o((att @ v).transpose(1, 2).reshape(B, T, D))
    resid_mid = x + attn_out
    hid = F.relu(blk.up(resid_mid))
    mlp_out = blk.down(hid)
    resid_post = resid_mid + mlp_out
    logits = model.unembed(resid_post[:, -1])
    last = lambda t: t[:, -1].detach().cpu().double().numpy()
    return {"attn_out": last(attn_out), "resid_mid": last(resid_mid), "mlp_hid": last(hid),
            "mlp_out": last(mlp_out), "resid_post": last(resid_post), "logits": logits.detach().cpu().double().numpy()}


def stats(name, A, rng, n_shuffles):
    real = analyze_matrix(name, A, rng, n_shuffles=n_shuffles)
    sv = np.linalg.svd(A, compute_uv=False)
    p_ = sv / sv.sum()
    return {"mass_above": real.mass_above, "frac_above": real.frac_above, "alpha": real.alpha, "sigma": real.sigma_struct,
            "octaves_total": real.octaves_total, "conc": (real.mass_above / real.frac_above) if real.frac_above else float("nan"),
            "erank": float(np.exp(-(p_ * np.log(p_ + 1e-12)).sum())), "fro": float(np.linalg.norm(A)),
            "top_sv": [float(x) for x in sv[:16]]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--p", type=int, default=113)
    ap.add_argument("--n-inputs", type=int, default=4096, help="fixed random sample of (a, b) pairs, same for every checkpoint")
    ap.add_argument("--n-shuffles", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    dev = a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu"
    pairs = np.array([(i, j) for i in range(a.p) for j in range(a.p)])
    sel = np.random.default_rng(12345).choice(len(pairs), size=min(a.n_inputs, len(pairs)), replace=False)
    ab = torch.tensor(pairs[sel], device=dev)
    answer = (pairs[sel].sum(1)) % a.p
    ckpts = sorted(glob.glob(os.path.join(a.run_dir, "ckpt", "ep*.pt")))
    ckpts = [c for c in ckpts if not c.endswith(".opt")]
    if not ckpts:
        sys.exit(f"no checkpoints under {a.run_dir}/ckpt")
    beh = {}
    logp = os.path.join(a.run_dir, "log.jsonl")
    if os.path.exists(logp):
        for l in open(logp):
            r = json.loads(l)
            if "test_acc" in r:
                beh[int(r["epoch"])] = r
    model = Model(a.p).to(dev)
    out = a.out or os.path.join(a.run_dir, "acts_log.jsonl")
    with open(out, "w") as f:
        for ck in ckpts:
            ep = int(os.path.basename(ck)[2:8])
            model.load_state_dict(torch.load(ck, map_location=dev))
            with torch.no_grad():
                acts = activations(model, ab)
            layers = {n: stats(n, A, rng, a.n_shuffles) for n, A in acts.items()}
            # task-aware comparator: mean logit vector per answer class, DFT over the class axis
            M = np.stack([acts["logits"][answer == c].mean(0) for c in range(a.p)])  # (p, p)
            Fm = np.abs(np.fft.rfft(M, axis=0)).sum(1)
            layers["fourier"] = {"mass_above": float("nan"), "frac_above": float("nan"), "alpha": float("nan"), "sigma": float("nan"),
                                 "gini_logits_by_answer": gini(Fm), "fro": float(np.linalg.norm(M)), "top_sv": []}
            rec = {"epoch": ep, "layers": layers,
                   "test_loss": beh.get(ep, {}).get("test_loss"), "test_acc": beh.get(ep, {}).get("test_acc"),
                   "train_acc": beh.get(ep, {}).get("train_acc"), "train_loss": beh.get(ep, {}).get("train_loss")}
            f.write(json.dumps(rec) + "\n"); f.flush()
            L = layers
            print(f"ep {ep:6d} acc {rec['test_acc']} | resid_post mass {L['resid_post']['mass_above']:.3f} frac {L['resid_post']['frac_above']:.3f} erank {L['resid_post']['erank']:.1f} | mlp_hid mass {L['mlp_hid']['mass_above']:.3f} erank {L['mlp_hid']['erank']:.1f} | logits erank {L['logits']['erank']:.1f} gini {layers['fourier']['gini_logits_by_answer']:.3f}", flush=True)
    print(out)


if __name__ == "__main__":
    main()
