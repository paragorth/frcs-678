"""Grokking on modular addition with Sigma_struct tracked through training.

Setup follows Nanda et al. (2023) / Power et al. (2022): a one-layer transformer
trained with full-batch AdamW and strong weight decay on (a + b) mod p from a
fraction of all pairs. Train accuracy hits 100% early (memorisation); test
accuracy stays at chance for thousands of epochs and then jumps (grokking).

Prediction being tested: Sigma_struct (semataxy.structure) of the weight
matrices is low / rank-collapsed during memorisation and rises at or before
the test-accuracy jump. If it does not move at the grok, the measure is not
seeing the transition from lookup table to circuit.

    python tools/grokking.py --out runs/grokking --epochs 30000
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import semataxy  # noqa: E402,F401
from semataxy.structure import analyze_matrix  # noqa: E402


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.up = nn.Linear(d, 4 * d, bias=False)
        self.down = nn.Linear(4 * d, d, bias=False)
        self.heads = heads

    def forward(self, x):
        B, T, D = x.shape
        h = self.heads
        q = self.q(x).view(B, T, h, D // h).transpose(1, 2)
        k = self.k(x).view(B, T, h, D // h).transpose(1, 2)
        v = self.v(x).view(B, T, h, D // h).transpose(1, 2)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(D // h)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
        att = att.masked_fill(mask, float("-inf")).softmax(-1)
        x = x + self.o((att @ v).transpose(1, 2).reshape(B, T, D))
        x = x + self.down(F.relu(self.up(x)))
        return x


class Model(nn.Module):
    def __init__(self, p, d=128, heads=4):
        super().__init__()
        self.tok = nn.Embedding(p + 1, d)  # tokens 0..p-1 and '='
        self.pos = nn.Embedding(3, d)
        self.block = Block(d, heads)
        self.unembed = nn.Linear(d, p, bias=False)

    def forward(self, ab):
        B = ab.shape[0]
        eq = torch.full((B, 1), self.tok.num_embeddings - 1, dtype=torch.long)
        x = torch.cat([ab, eq], 1)
        x = self.tok(x) + self.pos(torch.arange(3))[None]
        x = self.block(x)
        return self.unembed(x[:, -1])

    def matrices(self):
        return {n: p.detach().cpu().numpy() for n, p in self.named_parameters()
                if p.ndim == 2 and "tok" not in n and "pos" not in n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=113)
    ap.add_argument("--frac", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=30000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--every", type=int, default=200)
    ap.add_argument("--out", default="runs/grokking")
    ap.add_argument("--resume", default=None, help="state dict to continue from")
    ap.add_argument("--start-epoch", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    os.makedirs(a.out, exist_ok=True)

    p = a.p
    pairs = torch.tensor([(i, j) for i in range(p) for j in range(p)])
    labels = (pairs[:, 0] + pairs[:, 1]) % p
    perm = torch.randperm(len(pairs))
    n_train = int(a.frac * len(pairs))
    tr, te = perm[:n_train], perm[n_train:]

    model = Model(p)
    if a.resume:
        model.load_state_dict(torch.load(a.resume))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd, betas=(0.9, 0.98))
    if a.resume and os.path.exists(a.resume + ".opt"):
        opt.load_state_dict(torch.load(a.resume + ".opt"))
    log = open(os.path.join(a.out, "log.jsonl"), "a" if a.resume else "w")
    t0 = time.time()
    for epoch in range(a.start_epoch, a.epochs + 1):
        if a.resume and epoch == a.start_epoch:
            continue  # already logged by the previous run
        model.train()
        logits = model(pairs[tr])
        loss = F.cross_entropy(logits, labels[tr])
        opt.zero_grad(); loss.backward(); opt.step()
        if epoch % a.every == 0:
            model.eval()
            with torch.no_grad():
                tr_acc = (model(pairs[tr]).argmax(-1) == labels[tr]).float().mean().item()
                te_logits = model(pairs[te])
                te_acc = (te_logits.argmax(-1) == labels[te]).float().mean().item()
                te_loss = F.cross_entropy(te_logits, labels[te]).item()
            structs = {}
            for name, W in model.matrices().items():
                st = analyze_matrix(name, W, rng)
                structs[name] = {"sigma": st.sigma_struct, "frac_above": st.frac_above,
                                 "mass_above": st.mass_above, "alpha": st.alpha,
                                 "octaves_total": st.octaves_total}
            rec = {"epoch": epoch, "train_loss": loss.item(), "train_acc": tr_acc,
                   "test_loss": te_loss, "test_acc": te_acc,
                   "sigma_mean": float(np.mean([s["sigma"] for s in structs.values()])),
                   "mass_mean": float(np.mean([s["mass_above"] for s in structs.values()])),
                   "layers": structs, "wall_s": time.time() - t0}
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(f"ep {epoch:6d} train {tr_acc:.3f} test {te_acc:.3f} loss {loss.item():.3f} "
                  f"sigma {rec['sigma_mean']:.2f} mass {rec['mass_mean']:.3f} ({rec['wall_s']:.0f}s)", flush=True)
    torch.save(model.state_dict(), os.path.join(a.out, "final.pt"))
    torch.save(opt.state_dict(), os.path.join(a.out, "final.pt.opt"))


if __name__ == "__main__":
    main()
