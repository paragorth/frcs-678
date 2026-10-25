"""Null-split chimera: which part of a weight direction carries the circuit's identity?

Khanh 2026 (arXiv 2607.06628) decomposes each weight vector into magnitude and direction and
transplants them between independently trained seeds. Implanting a donor's *direction* into a
recipient drives continued training toward the donor's circuit in 40/40 recombinations, with an
angle-matched random control producing no shift; the norm carries no identity information. They
transplant the whole direction and never ask which part of it carries the identity.

This splits the direction at the permutation-null edge and transplants only part of it:

    full        donor's whole matrix direction            (reproduces their result)
    above       donor's above-edge component only, recipient keeps its own bulk
    below       donor's below-edge bulk only, recipient keeps its own above-edge part
    shuffled    donor's above-edge component, entries shuffled within the component
                (angle-matched-style control: same magnitude and rank, no donor content)

If identity transfers under `above` but not `below`, the above-edge subspace *is* the circuit
identity and the null isolates it. If it transfers under both, or neither, the edge cuts in the
wrong place -- which is equally worth knowing, and is the reason this is run as a comparison rather
than as a demonstration of `above` alone.

Identity is read out the way the quartet test already reads it: continue training the chimera
briefly, then score it. Two seeds trained on the same task reach *different* circuits but the same
accuracy, so accuracy alone cannot say whose circuit won. The readout is therefore agreement with
each parent's predictions on the held-out pairs -- which parent does the chimera answer like?

    python tools/chimera_null.py --donor runs/chimera/d128_f0.4_s1/ckpt \
        --recipient runs/chimera/d128_f0.4_s0/ckpt --epoch 20000 --steps 200
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from semataxy.structure import shuffled, spectrum  # noqa: E402
from tools.grokking import Model  # noqa: E402

MATS = ["block.q.weight", "block.k.weight", "block.v.weight", "block.o.weight",
        "block.up.weight", "block.down.weight", "unembed.weight"]


def split_by_edge(W, rng, n_shuffles=4, tol=1.02):
    """Same split as tools/pregrok_quartet.py, so counts are comparable with the quartet test."""
    W64 = np.asarray(W, np.float64)
    U, s, Vt = np.linalg.svd(W64, full_matrices=False)
    n = max(W64.shape)
    lam = s ** 2 / n
    edge = max(spectrum(shuffled(W64, rng))[0] for _ in range(n_shuffles)) * tol
    keep = lam > edge
    above = (U[:, keep] * s[keep]) @ Vt[keep]
    return above, W64 - above, int(keep.sum())


def load(ckpt_dir, epoch):
    cands = sorted(c for c in glob.glob(os.path.join(ckpt_dir, "ep*.pt")) if not c.endswith(".opt"))
    if not cands:
        sys.exit(f"no checkpoints in {ckpt_dir}")
    eps = [int(os.path.basename(c)[2:8]) for c in cands]
    i = min(range(len(eps)), key=lambda j: abs(eps[j] - epoch))
    return torch.load(cands[i], map_location="cpu"), eps[i]


def make_chimera(rec, don, mode, rng, keep_norm=True, tol=1.02, n_shuffles=4):
    """Recipient state dict with some part of the donor's direction implanted.

    Khanh's protocol keeps the recipient's magnitude and takes the donor's direction, so each
    implanted matrix is rescaled to the recipient's Frobenius norm. That is what `keep_norm` does,
    and it matters: they find the recipient's norm predicts whether the transplant takes.
    """
    out = dict(rec)
    stats = []
    for n in MATS:
        R = rec[n].numpy().astype(np.float64)
        D = don[n].numpy().astype(np.float64)
        rA, rB, rk = split_by_edge(R, rng, n_shuffles, tol)
        dA, dB, dk = split_by_edge(D, rng, n_shuffles, tol)
        if mode == "full":
            M = D
        elif mode == "above":
            M = dA + rB
        elif mode == "below":
            M = rA + dB
        elif mode == "randsub":
            # pg's question (21 Oct): the matched control here is `shuffled`, which keeps the
            # implant's magnitude and rank but destroys its structure. The control that killed the
            # protection claim was different -- a *random subspace* of matched rank, carrying real
            # weight structure in the wrong directions. That one was never run on the transplant, and
            # after what it did to the protection result, reasoning that it would hold is not enough.
            # Donor's component inside a random rank-k subspace, recipient's component outside it.
            Q, _ = np.linalg.qr(rng.standard_normal((D.shape[1], max(dk, 1))))
            P = Q[:, :max(dk, 1)] @ Q[:, :max(dk, 1)].T
            M = D @ P + R @ (np.eye(D.shape[1]) - P)
        elif mode == "shuffled":
            flat = dA.ravel().copy(); rng.shuffle(flat)
            M = flat.reshape(dA.shape) + rB
        elif mode == "none":
            M = R
        elif mode in ("aligned", "aligned_full"):
            # Parag's point (20 Oct): the donor's directions are expressed in the donor's own basis,
            # and two independently trained models of the same task have near-orthogonal above-edge
            # subspaces (mean principal-angle cosine 0.238 against a random floor of 0.197). So
            # implanting the donor's component raw is inserting it in the wrong coordinates. Align
            # first by orthogonal Procrustes -- the rotation best mapping the donor's above-edge row
            # space onto the recipient's -- then implant. Same idea as permutation alignment in
            # Git Re-Basin, applied to the above-edge subspace rather than to neurons.
            src = D if mode == "aligned_full" else dA
            Ud, sd_, Vtd = np.linalg.svd(src, full_matrices=False)
            kd = max(1, dk if mode != "aligned_full" else min(src.shape))
            Ur, sr, Vtr = np.linalg.svd(R, full_matrices=False)
            kr = max(1, rk)
            k = min(kd, kr)
            # rotation aligning donor's top-k right-singular directions to the recipient's
            M = Vtd[:k].T @ Vtd[:k] @ Vtr[:k].T @ Vtr[:k]
            Uo, _, Vto = np.linalg.svd(Vtd[:k] @ Vtr[:k].T, full_matrices=False)
            Rot = Uo @ Vto                      # k x k orthogonal, donor-basis -> recipient-basis
            aligned_V = Rot.T @ Vtd[:k]
            M = (Ud[:, :k] * sd_[:k]) @ aligned_V
            M = M + (R - (Ur[:, :kr] * sr[:kr]) @ Vtr[:kr]) if mode != "aligned_full" else M
        elif mode.startswith("drop:"):
            # Leave-one-out: transplant the donor's above-edge component for every matrix EXCEPT
            # the named one, which keeps the recipient's. The accuracy drop relative to the full
            # above-edge transplant is that matrix's contribution, and the question is whether the
            # contributions track the null's per-matrix allocation (q:1 k:4 v:10 o:16 up:16 down:9
            # unembed:9). If they do, the allocation is doing real work; if not, the null gives a
            # budget without knowing where to spend it.
            held = mode.split(":", 1)[1]
            short = n.split(".")[-2] if "block" in n else "unembed"
            M = R if short == held else dA + rB
        elif mode.startswith("topk:"):
            # The control that matters: a fixed number of top directions per matrix, ignoring the
            # null. If some fixed k matches what the edge selects, the permutation null is adding
            # nothing beyond "take the strongest few" -- which is exactly how the fixed-k top-8
            # energy-share control defeated the null as a grokking detector on 17 Oct.
            k = int(mode.split(":")[1])
            U, s, Vt = np.linalg.svd(D, full_matrices=False)
            k = min(k, len(s))
            topk = (U[:, :k] * s[:k]) @ Vt[:k]
            Ur, sr, Vtr = np.linalg.svd(R, full_matrices=False)
            kr = min(k, len(sr))
            rest = R - (Ur[:, :kr] * sr[:kr]) @ Vtr[:kr]
            M = topk + rest
            dk = k
        else:
            sys.exit(f"unknown mode {mode}")
        if keep_norm:
            nm = np.linalg.norm(M)
            if nm > 0:
                M = M * (np.linalg.norm(R) / nm)
        out[n] = torch.tensor(M, dtype=rec[n].dtype)
        stats.append((n.split(".")[-2] if "block" in n else "unembed", rk, dk))
    return out, stats


def data(p, frac, seed):
    torch.manual_seed(seed)
    pairs = torch.tensor([(i, j) for i in range(p) for j in range(p)])
    labels = (pairs[:, 0] + pairs[:, 1]) % p
    perm = torch.randperm(len(pairs))
    n_tr = int(frac * len(pairs))
    return pairs, labels, perm[:n_tr], perm[n_tr:]


def logits(sd, p, pairs, idx):
    m = Model(p, d=sd["block.q.weight"].shape[0]); m.load_state_dict(sd); m.eval()
    with torch.no_grad():
        return m(pairs[idx])


def preds(sd, p, pairs, idx):
    return logits(sd, p, pairs, idx).argmax(-1)


def runner_up(lg):
    """Second-choice prediction. Two circuits that both compute the task correctly agree on the
    argmax by construction, so the argmax cannot distinguish them -- this is what made the first
    cross-seed run uninformative (both parents at 1.0000, agreement 1.0000, every mode ambiguous).
    The runner-up and the logit geometry still differ between circuits that agree on the answer."""
    top = lg.argmax(-1, keepdim=True)
    masked = lg.scatter(-1, top, float("-inf"))
    return masked.argmax(-1)


def logit_sim(a, b):
    """Mean cosine similarity between centred logit vectors, row by row. Insensitive to overall
    logit scale, sensitive to which alternatives each circuit considers plausible."""
    a = a - a.mean(-1, keepdim=True)
    b = b - b.mean(-1, keepdim=True)
    return float(F.cosine_similarity(a, b, dim=-1).mean())


def agreement(a, b):
    return float((a == b).float().mean())


def train_on(sd, p, pairs, labels, tr, steps, lr, wd):
    m = Model(p, d=sd["block.q.weight"].shape[0]); m.load_state_dict(sd); m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    for _ in range(steps):
        loss = F.cross_entropy(m(pairs[tr]), labels[tr])
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    return m.state_dict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--donor", required=True); ap.add_argument("--recipient", required=True)
    ap.add_argument("--epoch", type=int, default=20000)
    ap.add_argument("--donor-epoch", type=int, default=None,
                    help="donor checkpoint epoch if different from --epoch (temporal transplant: "
                         "implant a post-grok component into a still-memorising recipient)")
    ap.add_argument("--steps", type=int, default=200, help="continued-training steps after implant")
    ap.add_argument("--p", type=int, default=113); ap.add_argument("--frac", type=float, default=0.4)
    ap.add_argument("--data-seed", type=int, default=0,
                    help="both parents must share the data split for agreement to be meaningful")
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--wd", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/grokking/chimera_null.json")
    ap.add_argument("--modes", default="", help="comma-separated; e.g. none,above,topk:9,topk:20")
    ap.add_argument("--tol", type=float, default=1.02, help="edge tolerance in split_by_edge")
    ap.add_argument("--shuffles", type=int, default=4, help="shuffled draws the edge maximises over")
    ap.add_argument("--no-keep-norm", action="store_true",
                    help="implant the donor component at its OWN scale instead of rescaling to the "
                         "recipient Frobenius norm. Khanh rescales deliberately, to separate "
                         "magnitude from direction; for restoring a function the donor scale is "
                         "what you want.")
    a = ap.parse_args()

    rec, rec_ep = load(a.recipient, a.epoch)
    don, don_ep = load(a.donor, a.donor_epoch if a.donor_epoch is not None else a.epoch)
    p = a.p
    pairs, labels, tr, te = data(p, a.frac, a.data_seed)
    lg_rec, lg_don = logits(rec, p, pairs, te), logits(don, p, pairs, te)
    pr_rec, pr_don = lg_rec.argmax(-1), lg_don.argmax(-1)
    ru_rec, ru_don = runner_up(lg_rec), runner_up(lg_don)
    base = agreement(pr_rec, pr_don)
    base_ru = agreement(ru_rec, ru_don)
    base_lg = logit_sim(lg_rec, lg_don)
    acc = lambda sd: agreement(preds(sd, p, pairs, te), labels[te])
    print(f"recipient ep{rec_ep} test acc {acc(rec):.4f} | donor ep{don_ep} test acc {acc(don):.4f}")
    print(f"parents: argmax agreement {base:.4f}, runner-up agreement {base_ru:.4f}, "
          f"logit cosine {base_lg:.4f}")
    if base > 0.95 and base_ru > 0.95 and base_lg > 0.98:
        print("  parents indistinguishable on all three readouts -- this pair cannot test transfer")
    elif base > 0.95:
        print("  argmax cannot separate these parents (both correct); using runner-up and logit "
              "geometry, on which they differ")
    print(f"\n{'mode':10}{'acc@0':>10}{'acc@N':>8}{'agree_don':>11}{'agree_rec':>11}{'verdict':>16}")
    rows = []
    modes = a.modes.split(",") if a.modes else ["none", "full", "above", "below", "shuffled"]
    for mode in modes:
        rng = np.random.default_rng(a.seed)
        ch, stats = make_chimera(rec, don, mode, rng, keep_norm=not a.no_keep_norm,
                                 tol=a.tol, n_shuffles=a.shuffles)
        ch_pre = {k: v.clone() for k, v in ch.items()}
        ch = train_on(ch, p, pairs, labels, tr, a.steps, a.lr, a.wd)
        # Accuracy BEFORE any continued training. Parag's point: the leave-one-out arms allowed
        # 200 steps after the implant, during which the remaining matrices can compensate for the
        # withheld one, so a small drop cannot distinguish "this matrix did not matter" from "the
        # others covered for it". This is the uncompensated contribution.
        acc_immediate = agreement(preds(ch_pre, p, pairs, te), labels[te])
        lg_ch = logits(ch, p, pairs, te)
        pc, ruc = lg_ch.argmax(-1), runner_up(lg_ch)
        ad, ar = agreement(pc, pr_don), agreement(pc, pr_rec)
        rd, rr = agreement(ruc, ru_don), agreement(ruc, ru_rec)
        ld, lr = logit_sim(lg_ch, lg_don), logit_sim(lg_ch, lg_rec)
        # pick the readout that actually separates the parents
        if base > 0.95 and base_ru <= 0.95:
            dd, rrv, tag = rd, rr, "runner-up"
        elif base > 0.95:
            dd, rrv, tag = ld, lr, "logit-cos"
        else:
            dd, rrv, tag = ad, ar, "argmax"
        v = "donor-like" if dd > rrv + 0.02 else ("recipient-like" if rrv > dd + 0.02 else "ambiguous")
        print(f"{mode:10}{acc_immediate:>10.4f}{acc(ch):>8.4f}{dd:>11.4f}{rrv:>11.4f}{v:>16}")
        rows.append({"mode": mode, "acc_immediate": acc_immediate, "acc": acc(ch),
                     "agree_donor": ad, "agree_recipient": ar,
                     "ru_donor": rd, "ru_recipient": rr, "logit_donor": ld, "logit_recipient": lr,
                     "readout": tag,
                     "verdict": v, "kept_above_edge": [{"mat": m, "recipient": r, "donor": d}
                                                       for m, r, d in stats]})
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump({"donor": a.donor, "recipient": a.recipient, "epoch_used": [rec_ep, don_ep],
                   "steps": a.steps, "parent_agreement": base, "results": rows}, fh, indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
