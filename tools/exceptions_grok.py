"""Does grokking itself evict what it replaces? One function, with planted exceptions.

Parag's claim is that grokking *is* the mechanism of catastrophic forgetting: a new circuit
consolidating into the organised subspace evicts the previous occupant. The two-task experiments
support the permanence half of it (`docs/AUTHOR_NOTES.md`, 20 Oct) but they confound the mechanism
with a change of data distribution. This removes that confound entirely.

The task is (a + b) mod p, except for a small set of planted pairs whose labels are wrong. The
training split contains both. Early in training the network memorises everything, exceptions
included -- memorisation has no reason to treat them differently. Then it groks the rule. The
generalising circuit *cannot represent the exceptions*: a single algorithm that computes a + b for
every input is incapable of also returning something else for a handful of them.

So the prediction, recorded before running: exception accuracy rises to 1.0 during memorisation and
collapses at the grok, with no change in the data, no second task, and no distribution shift. If it
holds, the eviction is caused purely by internal reorganisation, which is the claim in its strongest
form. If the exceptions survive the grok, the network can hold both a rule and its exceptions at
once, and grokking is not eviction.

Reported at each logged step: accuracy on the rule-following training pairs, on the planted
exceptions, and on held-out pairs (the usual grokking curve), alongside the above-edge structure.

    python tools/exceptions_grok.py --d 128 --frac 0.4 --n-exceptions 20 --epochs 30000
"""
from __future__ import annotations
import argparse, csv, os, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/Users/pg/frcs-678")
from semataxy.structure import shuffled, spectrum  # noqa: E402
from tools.grokking import Model  # noqa: E402

MATS = ["block.q.weight", "block.k.weight", "block.v.weight", "block.o.weight",
        "block.up.weight", "block.down.weight", "unembed.weight"]


def above_stats(W, seed, n_shuffles=4, tol=1.02):
    W64 = np.asarray(W, np.float64)
    if W64.shape[0] < W64.shape[1]:
        W64 = W64.T
    n = W64.shape[0]
    lam = np.linalg.svd(W64, compute_uv=False) ** 2 / n
    rng = np.random.default_rng(seed)
    edge = max(spectrum(shuffled(W64, rng))[0] for _ in range(n_shuffles)) * tol
    above = lam[lam > edge]
    tot = lam.sum()
    if len(above) < 2 or edge <= 0:
        return len(above), float(above.sum() / tot) if tot > 0 else 0.0, float("nan")
    sl = float(np.sum(np.log(above / edge)))
    alpha = 1.0 + len(above) / sl if sl > 1e-12 else float("nan")
    return len(above), float(above.sum() / tot), alpha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=113); ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--frac", type=float, default=0.4)
    ap.add_argument("--n-exceptions", type=int, default=20,
                    help="planted pairs whose label breaks the rule. Kept small so the rule is "
                         "still the efficient solution and grokking still happens.")
    ap.add_argument("--epochs", type=int, default=30000)
    ap.add_argument("--every", type=int, default=250)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--wd", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/live_learning/exceptions.csv")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    p = a.p
    pairs = torch.tensor([(i, j) for i in range(p) for j in range(p)])
    true_labels = (pairs[:, 0] + pairs[:, 1]) % p
    labels = true_labels.clone()

    g = torch.Generator().manual_seed(a.seed)
    perm = torch.randperm(len(pairs), generator=g)
    n_tr = int(a.frac * len(pairs))
    tr, te = perm[:n_tr], perm[n_tr:]

    # plant exceptions inside the TRAINING split, with labels guaranteed different from the rule
    exc = tr[torch.randperm(len(tr), generator=g)[:a.n_exceptions]]
    offs = torch.randint(1, p, (len(exc),), generator=g)
    labels[exc] = (true_labels[exc] + offs) % p
    normal = torch.tensor([i for i in tr.tolist() if i not in set(exc.tolist())])

    print(f"d={a.d} p={p} frac={a.frac}: {len(tr)} training pairs, of which "
          f"{len(exc)} are planted exceptions ({100*len(exc)/len(tr):.2f}%)")
    print("prediction, recorded before running: exception accuracy rises to 1.0 during memorisation")
    print("and collapses at the grok, with no change to the data.\n")

    model = Model(p, d=a.d)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    acc = lambda idx: float((model(pairs[idx]).argmax(-1) == labels[idx]).float().mean())
    rows = []
    print(f"{'epoch':>7}{'rule_train':>11}{'EXCEPTIONS':>12}{'held_out':>10}{'k':>6}{'mass':>7}")
    for ep in range(a.epochs + 1):
        if ep % a.every == 0:
            model.eval()
            with torch.no_grad():
                an, ae, at = acc(normal), acc(exc), acc(te)
            sd = model.state_dict()
            st = [above_stats(sd[m].detach().numpy(), a.seed) for m in MATS]
            k = float(np.mean([s[0] for s in st])); mass = float(np.mean([s[1] for s in st]))
            alpha = float(np.nanmean([s[2] for s in st]))
            rows.append(dict(epoch=ep, rule_train=an, exceptions=ae, held_out=at,
                             k=k, mass=mass, alpha_above=alpha))
            if ep % (a.every * 4) == 0:
                print(f"{ep:>7}{an:>11.4f}{ae:>12.4f}{at:>10.4f}{k:>6.1f}{mass:>7.3f}", flush=True)
            model.train()
        if ep == a.epochs:
            break
        loss = F.cross_entropy(model(pairs[tr]), labels[tr])
        opt.zero_grad(); loss.backward(); opt.step()

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {a.out}")

    grok = next((r for r in rows if r["held_out"] >= 0.5), None)
    if grok:
        pre = [r for r in rows if r["epoch"] < grok["epoch"]]
        peak = max(r["exceptions"] for r in pre) if pre else float("nan")
        end = rows[-1]["exceptions"]
        print(f"grok at epoch {grok['epoch']}; exception accuracy peaked at {peak:.4f} before it "
              f"and ended at {end:.4f}")
        print("-> exceptions were evicted by the grok" if peak > 0.8 and end < 0.3 else
              "-> exceptions were NOT evicted; the network held rule and exceptions together")
    else:
        print("the model never grokked -- the exceptions may have prevented it; "
              "retry with fewer, or more epochs")


if __name__ == "__main__":
    main()
