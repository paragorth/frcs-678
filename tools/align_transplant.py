"""Align two models' hidden units, then transplant the above-edge component.

Two independently trained models of the same task have near-orthogonal above-edge subspaces (mean
principal-angle cosine 0.238 against a random floor of 0.197, measured 20 Oct). So the donor's
circuit is written in the donor's own coordinates, and implanting it raw into a recipient gives the
function (accuracy 0.96) but not the identity (donor agreement 0.09).

Parag's proposal was to "enter it at the same position". A first attempt rotated each matrix's input
basis independently and destroyed the model (accuracy 0.144): a matrix's inputs arrive from the
previous layer in *that* layer's coordinates, so a per-matrix rotation breaks the wiring. Alignment
has to be one transformation of the shared hidden space, applied consistently to every matrix that
reads from it or writes to it -- which is what Git Re-Basin (Ainsworth et al.) does with neuron
permutations before interpolating two networks.

Here the shared space is the residual stream of width d. A permutation P of it must be applied as:

    writes into the stream   (tok, pos, block.o, block.down)   ->   P @ W
    reads from the stream    (block.q/k/v, block.up, unembed)  ->   W @ P.T

P is chosen to maximise the alignment of donor and recipient weights across all of those at once,
which is a linear assignment problem and so is solved exactly by the Hungarian algorithm rather
than approximately.

    python tools/align_transplant.py --donor <ckpt_dir> --recipient <ckpt_dir> --epoch 10000
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, "/Users/pg/frcs-678")
from semataxy.structure import shuffled, spectrum  # noqa: E402
from tools.grokking import Model  # noqa: E402

WRITES = ["tok.weight", "pos.weight", "block.o.weight", "block.down.weight"]   # rows indexed by d
READS = ["block.q.weight", "block.k.weight", "block.v.weight", "block.up.weight",
         "unembed.weight"]                                                      # cols indexed by d
SPLIT = ["block.q.weight", "block.k.weight", "block.v.weight", "block.o.weight",
         "block.up.weight", "block.down.weight", "unembed.weight"]


def residual_permutation(don, rec, d):
    """One permutation of the residual stream maximising donor-recipient weight alignment.

    Cost is the total inner product achieved by assigning donor unit j to recipient unit i, summed
    over every matrix that writes into or reads from the stream. Maximising a sum of inner products
    over permutations is a linear assignment problem.
    """
    C = np.zeros((d, d))
    for n in WRITES:
        A = rec[n].numpy().astype(np.float64)
        B = don[n].numpy().astype(np.float64)
        if A.shape[0] != d:
            A, B = A.T, B.T
        C += A @ B.T                      # rows are stream units
    for n in READS:
        A = rec[n].numpy().astype(np.float64)
        B = don[n].numpy().astype(np.float64)
        if A.shape[1] != d:
            A, B = A.T, B.T
        C += A.T @ B                      # columns are stream units
    r, c = linear_sum_assignment(-C)      # maximise
    P = np.zeros((d, d))
    P[r, c] = 1.0
    frac_moved = float(np.mean(c != np.arange(d)))
    return P, frac_moved


def permute_state(sd, P, d):
    out = {}
    for n, t in sd.items():
        W = t.numpy().astype(np.float64)
        if n in WRITES:
            out[n] = torch.tensor(P @ W if W.shape[0] == d else (P @ W.T).T, dtype=t.dtype)
        elif n in READS:
            out[n] = torch.tensor(W @ P.T if W.shape[1] == d else (W.T @ P.T).T, dtype=t.dtype)
        else:
            out[n] = t.clone()
    return out


def split_by_edge(W, rng, n_shuffles=4, tol=1.02):
    W64 = np.asarray(W, np.float64)
    U, s, Vt = np.linalg.svd(W64, full_matrices=False)
    lam = s ** 2 / max(W64.shape)
    edge = max(spectrum(shuffled(W64, rng))[0] for _ in range(n_shuffles)) * tol
    keep = lam > edge
    return (U[:, keep] * s[keep]) @ Vt[keep], int(keep.sum())


def load(ckpt_dir, epoch):
    cs = sorted(c for c in glob.glob(os.path.join(ckpt_dir, "ep*.pt")) if not c.endswith(".opt"))
    eps = [int(os.path.basename(c)[2:8]) for c in cs]
    i = min(range(len(eps)), key=lambda j: abs(eps[j] - epoch))
    return torch.load(cs[i], map_location="cpu"), eps[i]


def data(p, frac, seed):
    pairs = torch.tensor([(i, j) for i in range(p) for j in range(p)])
    labels = (pairs[:, 0] + pairs[:, 1]) % p
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(pairs), generator=g)
    n = int(frac * len(pairs))
    return pairs, labels, perm[:n], perm[n:]


def lg(sd, p, pairs, idx):
    m = Model(p, d=sd["block.q.weight"].shape[0]); m.load_state_dict(sd); m.eval()
    with torch.no_grad():
        return m(pairs[idx])


def runner_up(x):
    return x.scatter(-1, x.argmax(-1, keepdim=True), float("-inf")).argmax(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--donor", required=True); ap.add_argument("--recipient", required=True)
    ap.add_argument("--epoch", type=int, default=10000)
    ap.add_argument("--p", type=int, default=113); ap.add_argument("--frac", type=float, default=0.4)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--wd", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/grokking/align_transplant.json")
    a = ap.parse_args()

    don, de = load(a.donor, a.epoch); rec, re_ = load(a.recipient, a.epoch)
    d = rec["block.q.weight"].shape[0]
    pairs, labels, tr, te = data(a.p, a.frac, a.seed)
    acc = lambda sd: float((lg(sd, a.p, pairs, te).argmax(-1) == labels[te]).float().mean())

    P, moved = residual_permutation(don, rec, d)
    don_p = permute_state(don, P, d)
    print(f"donor ep{de} acc {acc(don):.4f} | recipient ep{re_} acc {acc(rec):.4f}")
    print(f"residual permutation found: {100*moved:.1f}% of {d} units reassigned")
    print(f"permuted donor acc {acc(don_p):.4f}  "
          f"({'PRESERVED -- permutation is a symmetry, as it must be' if acc(don_p) > 0.95 else 'BROKEN -- permutation is not being applied consistently'})")

    L_don, L_rec = lg(don_p, a.p, pairs, te), lg(rec, a.p, pairs, te)
    ru_don, ru_rec = runner_up(L_don), runner_up(L_rec)
    agree = lambda x, y: float((x == y).float().mean())
    print(f"\nparents after alignment: argmax {agree(L_don.argmax(-1), L_rec.argmax(-1)):.4f}, "
          f"runner-up {agree(ru_don, ru_rec):.4f}")

    rows = []
    print(f"\n{'mode':16}{'acc':>8}{'ru_donor':>10}{'ru_recip':>10}")
    for mode in ("none", "above_raw", "above_aligned"):
        sd = dict(rec)
        if mode != "none":
            src = don if mode == "above_raw" else don_p
            rng = np.random.default_rng(a.seed)
            for n in SPLIT:
                R = rec[n].numpy().astype(np.float64)
                dA, _ = split_by_edge(src[n].numpy(), rng)
                rA, _ = split_by_edge(R, rng)
                M = dA + (R - rA)
                nm = np.linalg.norm(M)
                if nm > 0:
                    M = M * (np.linalg.norm(R) / nm)
                sd[n] = torch.tensor(M, dtype=rec[n].dtype)
        m = Model(a.p, d=d); m.load_state_dict(sd); m.train()
        opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=a.wd)
        for _ in range(a.steps):
            loss = F.cross_entropy(m(pairs[tr]), labels[tr])
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        out = m.state_dict()
        L = lg(out, a.p, pairs, te)
        rd, rr = agree(runner_up(L), ru_don), agree(runner_up(L), ru_rec)
        print(f"{mode:16}{acc(out):>8.4f}{rd:>10.4f}{rr:>10.4f}")
        rows.append(dict(mode=mode, acc=acc(out), ru_donor=rd, ru_recipient=rr))

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(dict(donor=a.donor, recipient=a.recipient, epoch=[de, re_],
                       frac_units_reassigned=moved, permuted_donor_acc=acc(don_p),
                       results=rows), fh, indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
