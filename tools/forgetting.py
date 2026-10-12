"""T7: catastrophic forgetting as the grokking operation run without the world in view.

Fine-tune a pretrained LM on a narrow corpus and track, every few steps:
  - held-out loss on general text (forgetting),
  - narrow-corpus loss (learning),
  - Sigma_struct span and concentration of every weight matrix.

Two arms: plain narrow fine-tuning, and the same with interleaved replay of
general text (the complementary-learning-systems safeguard).

Predictions (docs/STRUCTURAL_TESTS.md, T7):
  narrow:  concentration up, span DOWN, held-out loss UP
  replay:  concentration up, span kept, held-out loss kept

    python tools/forgetting.py --arm narrow --steps 400 --out runs/forgetting_narrow
    python tools/forgetting.py --arm replay --steps 400 --out runs/forgetting_replay
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import semataxy  # noqa: E402,F401
from semataxy.structure import analyze_matrix  # noqa: E402


def chunks(tok, text, seq, limit=None):
    ids = tok(text, return_tensors="pt").input_ids[0]
    n = (len(ids) // seq) * seq
    out = ids[:n].view(-1, seq)
    return out[:limit] if limit else out


def load_corpora(tok, seq):
    """Narrow: tiny Shakespeare (Karpathy's char-rnn copy). General: WikiText-2 raw via the HF hub parquet."""
    import pandas as pd
    import requests
    from huggingface_hub import hf_hub_download

    narrow = requests.get(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", timeout=60
    ).text
    tr = pd.read_parquet(hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/train-00000-of-00001.parquet",
                                         repo_type="dataset"), engine="fastparquet")
    te = pd.read_parquet(hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet",
                                         repo_type="dataset"), engine="fastparquet")
    general_train = "\n".join(t for t in tr["text"] if t.strip())
    general_test = "\n".join(t for t in te["text"] if t.strip())
    return (chunks(tok, narrow, seq), chunks(tok, general_train, seq, 2000), chunks(tok, general_test, seq, 120))


@torch.no_grad()
def eval_loss(model, data, bs=8):
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, len(data), bs):
        x = data[i:i + bs]
        tot += model(x, labels=x).loss.item() * len(x)
        n += len(x)
    model.train()
    return tot / n


def structure(model, rng):
    out = {}
    for name, p in model.named_parameters():
        if p.ndim == 2 and min(p.shape) >= 64 and "embed" not in name:
            st = analyze_matrix(name, p.detach().cpu().float().numpy(), rng)
            out[name] = {"sigma": st.sigma_struct, "mass": st.mass_above, "frac": st.frac_above, "alpha": st.alpha}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-160m")
    ap.add_argument("--arm", choices=["narrow", "replay"], required=True)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    os.makedirs(a.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32)
    narrow, gen_train, gen_test = load_corpora(tok, a.seq)
    print(f"narrow {len(narrow)} chunks, general train {len(gen_train)}, held-out {len(gen_test)}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    log = open(os.path.join(a.out, "log.jsonl"), "w")
    t0 = time.time()

    def record(step):
        s = structure(model, rng)
        rec = {"step": step, "arm": a.arm,
               "heldout_loss": eval_loss(model, gen_test), "narrow_loss": eval_loss(model, narrow[:120]),
               "sigma_mean": float(np.mean([v["sigma"] for v in s.values()])),
               "mass_mean": float(np.mean([v["mass"] for v in s.values()])),
               "frac_mean": float(np.mean([v["frac"] for v in s.values()])),
               "alpha_median": float(np.nanmedian([v["alpha"] for v in s.values()])),
               "layers": s, "wall_s": time.time() - t0}
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(f"step {step:4d} heldout {rec['heldout_loss']:.3f} narrow {rec['narrow_loss']:.3f} "
              f"sigma {rec['sigma_mean']:.2f} mass {rec['mass_mean']:.3f} frac {rec['frac_mean']:.3f} "
              f"alpha {rec['alpha_median']:.1f} ({rec['wall_s']:.0f}s)", flush=True)

    # Keep the pretrained weights so the update dW = W - W0 can be scored for
    # "intrusion": how much of it lands on W0's existing above-edge directions
    # (damage spreading into old structure) versus orthogonal to them
    # (compartmentalised). Computed in analysis from final.pt and the HF base.
    record(0)
    model.train()
    for step in range(1, a.steps + 1):
        idx = torch.randint(0, len(narrow), (a.bs,))
        x = narrow[idx]
        if a.arm == "replay":  # interleave: half the batch is general text
            gidx = torch.randint(0, len(gen_train), (a.bs // 2,))
            x = torch.cat([narrow[idx[: a.bs // 2]], gen_train[gidx]])
        loss = model(x, labels=x).loss
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % a.every == 0:
            record(step)
    torch.save(model.state_dict(), os.path.join(a.out, "final.pt"))


def intrusion(W0: np.ndarray, W1: np.ndarray, rng) -> dict:
    """Fraction of the update's energy that lies in W0's above-edge left/right singular subspaces."""
    from semataxy.structure import shuffled
    W0 = np.asarray(W0, dtype=np.float64); W1 = np.asarray(W1, dtype=np.float64)
    dW = W1 - W0
    U, s, Vt = np.linalg.svd(W0, full_matrices=False)
    n = max(W0.shape)
    lam = s**2 / n
    edge = 1.02 * (np.linalg.svd(shuffled(W0, rng), compute_uv=False) ** 2 / n).max()
    k = int((lam > edge).sum())
    if k == 0:
        return {"k": 0, "intrusion": 0.0}
    P = U[:, :k] @ (U[:, :k].T @ dW) @ Vt[:k].T @ Vt[:k]
    return {"k": k, "intrusion": float((P**2).sum() / max((dW**2).sum(), 1e-12))}


if __name__ == "__main__":
    main()
