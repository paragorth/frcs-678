"""Catastrophic forgetting in a real pretrained model, with the above-edge structure measured.

The toy results (docs/AUTHOR_NOTES.md, 20 Oct) say permanent loss of a capability requires the
interfering task to consolidate a circuit, and that the eviction is visible as the above-edge
direction count collapsing to the new task's natural size. Both were measured in a one-layer
transformer on modular arithmetic. This asks whether any of it survives contact with a real model.

Setup. Pythia-160m, pretrained, 12 layers. Task A is the capability it already has -- general
language modelling, scored as perplexity on held-out wikitext. Task B is a narrow synthetic rule
presented as text, so that its acquisition has a clean accuracy curve and a detectable point of
consolidation. Fine-tuning on B degrades A; the question is whether the degradation becomes
permanent, and whether the weights show the same signature.

Three phases:
  1. baseline      -- A's perplexity and the pretrained above-edge subspace, as the reference
  2. interference  -- fine-tune on B, logging A's perplexity, B's accuracy and the structure
  3. recovery      -- train on wikitext again, and see whether A's perplexity returns

What would support the toy finding: A's perplexity recovering fully when B has not consolidated, and
failing to recover once it has, with the direction count falling during B's consolidation. What
would refute it: recovery behaving the same either way, or the structure not moving at all, which
would mean the toy result does not describe real models.

    python tools/real_model_forgetting.py --steps 1500 --recover 1500
"""
from __future__ import annotations
import argparse, csv, json, math, os, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/Users/pg/frcs-678")
from semataxy.structure import shuffled, spectrum  # noqa: E402

MODEL = "EleutherAI/pythia-160m"


def device_of(name):
    if name == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    return name


def tracked_matrices(model, min_dim=256):
    """Every 2-D weight of reasonable size, excluding embeddings (whose rows are tokens, so a
    shuffled null over all entries is not the right comparison for them)."""
    out = {}
    for n, p in model.named_parameters():
        if p.ndim == 2 and min(p.shape) >= min_dim and "embed" not in n:
            out[n] = p
    return out


def _eigs(W64):
    """Eigenvalues lam = s^2/n, descending, and the right singular vectors as rows.

    LAPACK's gesdd driver, which numpy uses, failed to converge on a Pythia matrix at the first
    interference measurement in both part-B cells (runs/chain.log, 21 Oct); the weights were finite,
    so this is driver non-convergence and not divergence. eigh on the Gram matrix returns the same
    two quantities and does converge, so it serves as the fallback."""
    n = W64.shape[0]
    try:
        _, s, Vt = np.linalg.svd(W64, full_matrices=False)
        return s ** 2 / n, Vt
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(W64.T @ W64)
        order = np.argsort(w)[::-1]
        return np.clip(w[order], 0.0, None) / n, V[:, order].T


def above_basis(W, seed, n_shuffles=2, tol=1.02, max_dim=1536):
    """Above-edge row basis and summary stats. n_shuffles is 2 rather than 4 because each shuffle
    costs an SVD of a large matrix and the sweep in docs/AUTHOR_NOTES.md showed 1-16 shuffles change
    the selected count by at most two directions."""
    W64 = np.asarray(W, np.float64)
    if not np.isfinite(W64).all():
        return None, 0, float("nan"), float("nan"), float("nan")
    if W64.shape[0] < W64.shape[1]:
        W64 = W64.T
    if W64.shape[0] > max_dim:          # keep the SVD affordable on a laptop
        W64 = W64[:max_dim]
    n = W64.shape[0]
    lam, Vt = _eigs(W64)
    rng = np.random.default_rng(seed)
    edge = max(_eigs(shuffled(W64, rng))[0].max() for _ in range(n_shuffles)) * tol
    keep = lam > edge
    above = lam[keep]
    tot = lam.sum()
    if len(above) >= 2 and edge > 0:
        sl = float(np.sum(np.log(above / edge)))
        alpha = 1.0 + len(above) / sl if sl > 1e-12 else float("nan")
        pr = above / above.sum()
        ent = float(-np.sum(pr * np.log(pr)) / np.log(len(above)))
    else:
        alpha = ent = float("nan")
    return Vt[keep], int(keep.sum()), float(above.sum() / tot) if tot > 0 else 0.0, alpha, ent


def overlap(A, B):
    if A is None or B is None or len(A) == 0 or len(B) == 0:
        return float("nan")
    k = min(len(A), len(B))
    return float(np.mean(np.clip(np.linalg.svd(A[:k] @ B[:k].T, compute_uv=False), 0.0, 1.0)))


def rule_batch(tok, p, n, device, gen, table=None):
    """Task B: 'a + b = c' mod p, as text. A narrow rule with a clean accuracy readout, so that its
    consolidation can be located the way the grok can be in the toy setting. With `table`, c is a
    fixed random answer per pair instead -- same format and gradient scale, nothing to grok."""
    a = torch.randint(0, p, (n,), generator=gen)
    b = torch.randint(0, p, (n,), generator=gen)
    c = table[a, b] if table is not None else (a + b) % p
    texts = [f"{int(x)} + {int(y)} = {int(z)}" for x, y, z in zip(a, b, c)]
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=24)
    return {k: v.to(device) for k, v in enc.items()}


def rule_accuracy(model, tok, p, device, n=96, seed=1234):
    """Greedy-decode the answer and check it. Held-out by construction: fresh random pairs."""
    gen = torch.Generator().manual_seed(seed)
    a = torch.randint(0, p, (n,), generator=gen)
    b = torch.randint(0, p, (n,), generator=gen)
    c = (a + b) % p
    prompts = [f"{int(x)} + {int(y)} =" for x, y in zip(a, b)]
    tok.padding_side = "left"
    enc = tok(prompts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=4, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    gen_txt = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    tok.padding_side = "right"
    hits = 0
    for t, z in zip(gen_txt, c):
        digits = "".join(ch for ch in t.strip().split()[0] if ch.isdigit()) if t.strip() else ""
        hits += digits == str(int(z))
    return hits / n


def perplexity(model, batches):
    tot, ntok = 0.0, 0
    with torch.no_grad():
        for b in batches:
            out = model(**b, labels=b["input_ids"])
            n = int(b["attention_mask"].sum())
            tot += float(out.loss) * n
            ntok += n
    return math.exp(tot / max(ntok, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--p", type=int, default=50, help="modulus for the synthetic rule")
    ap.add_argument("--steps", type=int, default=1500); ap.add_argument("--recover", type=int, default=1500)
    ap.add_argument("--every", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5); ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-eval-batches", type=int, default=8)
    ap.add_argument("--out", default="results/live_learning/real_model.csv")
    # The change measure (k_dW) grows with interference steps, so it may track "amount of fine-tuning"
    # rather than forgetting. This control fine-tunes on wikitext itself in the interference phase:
    # same steps, rate and optimiser, nothing forgotten.
    ap.add_argument("--control-wikitext", action="store_true",
                    help="interference phase trains on wikitext instead of the rule")
    # pg (23 Oct): the forgetting update is compact and off the pretrained directions, which is the
    # shape of a grokked circuit forming. The toy's random-label control, ported: same text format,
    # a fixed random answer per pair, so the rule can only be memorised. Prediction: the update
    # stays diffuse, and the damage to wikitext is milder.
    ap.add_argument("--random-labels", action="store_true",
                    help="interference rule answers are a fixed random table, not (a + b) mod p")
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import glob as _glob
    import pandas as pd

    dev = device_of(a.device)
    print(f"device={dev}  model={a.model}")
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.pad_token = tok.eos_token
    # Pythia checkpoints are stored in float16. Loading at the stored dtype and then training with
    # AdamW diverges on the first optimiser step -- loss 5.16 then NaN, on both CPU and MPS, which
    # is what crashed the first two attempts inside an SVD. Force float32.
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).to(dev)

    # read the cached parquet directly: the `datasets` library is not installed and there is no pip
    # in this venv, but pandas reads the file that `datasets` would have downloaded anyway.
    # Honour HF_HOME rather than hardcoding ~/.cache/huggingface: the model is loaded through the
    # transformers cache, which does respect it, so a hardcoded dataset path sends the two halves of
    # the input to different places as soon as the cache moves (as it does on the cluster).
    hf_home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    cand = sorted(_glob.glob(os.path.join(
        hf_home, "hub/datasets--Salesforce--wikitext/snapshots/*/wikitext-2-raw-v1/test-*.parquet")))
    if not cand:
        sys.exit(f"wikitext parquet not found under {hf_home}")
    texts = [t for t in pd.read_parquet(cand[0])["text"].tolist() if len(t.strip()) > 200]
    print(f"wikitext: {len(texts)} passages over 200 chars")
    def wk_batches(idx):
        out = []
        for i in range(0, len(idx), a.batch):
            chunk = [texts[j] for j in idx[i:i + a.batch]]
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=128)
            out.append({k: v.to(dev) for k, v in enc.items()})
        return out
    rng = np.random.default_rng(a.seed)
    order = rng.permutation(len(texts))
    eval_idx = list(order[:a.batch * a.n_eval_batches])
    train_idx = list(order[a.batch * a.n_eval_batches:])
    eval_batches = wk_batches(eval_idx)

    mats = list(tracked_matrices(model))
    print(f"tracking {len(mats)} weight matrices\n")

    ref, rows, W0 = {}, [], {}
    gen = torch.Generator().manual_seed(a.seed)

    def measure(tag, step, b_acc):
        sd = model.state_dict()
        ks, ms, al, en, ov = [], [], [], [], []
        # relative weight displacement per matrix, and which matrix moved most. A count that does not
        # move is uninformative if the weights did not move either; this separates the two.
        dW, dW_name, dW_top = [], "", 0.0
        for n in mats:
            if tag == "baseline":
                W0[n] = sd[n].detach().float().clone()
            else:
                d = float((sd[n].detach().float() - W0[n]).norm() / W0[n].norm())
                dW.append(d)
                if d > dW_top:
                    dW_top, dW_name = d, n
        # pg (22 Oct): k on W cannot move when W moves 1-3% -- the pretrained structure swamps the
        # update. So apply the same shuffle null to the CHANGE, D = W - W0: if the update is
        # organised it clears its own shuffled spectrum however small it is next to W. ov_dW_pre is
        # whether the update lands in the pretrained above-edge directions or beside them.
        kd, md, ed, ovd = [], [], [], []
        for n in mats:
            V, k, mass, alpha, ent = above_basis(sd[n].detach().float().cpu().numpy(), a.seed)
            ks.append(k); ms.append(mass); al.append(alpha); en.append(ent)
            if tag == "baseline":
                ref[n] = V
            else:
                ov.append(overlap(V, ref.get(n)))
                D = (sd[n].detach().float() - W0[n]).cpu().numpy()
                Vd, k2, m2, _, e2 = above_basis(D, a.seed)
                kd.append(k2); md.append(m2); ed.append(e2); ovd.append(overlap(Vd, ref.get(n)))
        nm = lambda v: float(np.nanmean(v)) if len(v) else float("nan")
        ppl = perplexity(model, eval_batches)
        row = dict(phase=tag, step=step, wikitext_ppl=ppl, rule_acc=b_acc, k=nm(ks),
                   mass=nm(ms), alpha_above=nm(al), entropy_above=nm(en), ov_pretrained=nm(ov),
                   dW_rel=nm(dW), dW_top=dW_top, dW_top_matrix=dW_name,
                   k_dW=nm(kd), mass_dW=nm(md), entropy_dW=nm(ed), ov_dW_pre=nm(ovd))
        rows.append(row)
        print(f"{tag:12}{step:>6}  ppl={ppl:>9.2f}  rule={b_acc:>6.3f}  k={row['k']:>6.1f}  "
              f"mass={row['mass']:.3f}  ov={row['ov_pretrained']:.4f}  "
              f"dW={row['dW_rel']:.5f}  worst={dW_name.replace('gpt_neox.layers.','L')[:22]} "
              f"{dW_top:.4f}  | change: k={row['k_dW']:.1f} mass={row['mass_dW']:.3f} "
              f"ent={row['entropy_dW']:.3f} ov_pre={row['ov_dW_pre']:.3f}", flush=True)
        return row

    model.eval()
    base = measure("baseline", 0, rule_accuracy(model, tok, a.p, dev))
    print(f"\nbaseline wikitext perplexity {base['wikitext_ppl']:.2f}, "
          f"rule accuracy {base['rule_acc']:.3f}\n")

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    model.train()
    wk_train = wk_batches(train_idx[:a.batch * 200])
    table = (torch.randint(0, a.p, (a.p, a.p), generator=torch.Generator().manual_seed(12345))
             if a.random_labels else None)
    for step in range(1, a.steps + 1):
        b = (wk_train[step % len(wk_train)] if a.control_wikitext
             else rule_batch(tok, a.p, a.batch, dev, gen, table))
        out = model(**b, labels=b["input_ids"])
        opt.zero_grad(); out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if not torch.isfinite(out.loss):
            sys.exit(f"loss became non-finite at interference step {step}; aborting before the SVD")
        if step % a.every == 0:
            model.eval(); measure("interference", step, rule_accuracy(model, tok, a.p, dev)); model.train()

    print()
    for step in range(1, a.recover + 1):
        b = wk_train[step % len(wk_train)]
        out = model(**b, labels=b["input_ids"])
        opt.zero_grad(); out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % a.every == 0:
            model.eval(); measure("recovery", a.steps + step, rule_accuracy(model, tok, a.p, dev)); model.train()

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        w = csv.DictWriter(fh, fieldnames=keys, restval=""); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {a.out}")
    worst = max(r["wikitext_ppl"] for r in rows if r["phase"] == "interference")
    final = rows[-1]["wikitext_ppl"]
    print(f"baseline ppl {base['wikitext_ppl']:.2f} -> worst during interference {worst:.2f} "
          f"-> after recovery {final:.2f}")
    print(f"rule accuracy reached {max(r['rule_acc'] for r in rows):.3f}")


if __name__ == "__main__":
    main()
