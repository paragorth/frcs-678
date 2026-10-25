"""Causal test: block the eviction and the old capability should survive.

The observational result (docs/AUTHOR_NOTES.md, 20 Oct) is that a capability is permanently lost
only when the interfering task consolidates a circuit into the same above-edge subspace, and that
the eviction shows up as the direction count collapsing to the new task's natural size. That is a
correlation between a structural event and an outcome. This intervenes on the structural event.

During interference, each gradient is projected to remove the component that acts on the old
capability's above-edge subspace. Concretely, for a tracked matrix W with the old model's above-edge
right-singular basis V (k x n), the update is replaced by

    dW  <-  dW - (dW V^T) V

so that dW v = 0 for every v in span(V): the new task cannot change how W acts on the directions the
old circuit reads. It is free to consolidate anywhere else. This is the null-space idea used by NESS
and Subspace Tuning, with one difference that matters for the present argument -- their subspace
comes from the previous task's input covariance, which requires its data, and this one comes from the
weights.

Three arms, because a projection removes degrees of freedom and that alone could preserve the old
capability for uninteresting reasons:

    none        no projection -- the baseline, in which the old capability ends capped at ~0.30
    protect     project out the old capability's above-edge subspace
    random      project out a RANDOM subspace of the same per-matrix dimension

If `protect` preserves the old capability and `random` does not, the specific subspace is the causal
mediator. If both preserve it, the result is about reduced capacity and says nothing about which
directions matter. If neither does, the eviction is not mediated by these directions at all.

Also reported: whether the interfering task still learns under projection. A method that protects the
old capability by preventing the new one from being acquired is not useful.

    python tools/protect_subspace.py --steps 8000 --recover 3000
"""
from __future__ import annotations
import argparse, csv, os, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/Users/pg/frcs-678")
from semataxy.structure import shuffled, spectrum  # noqa: E402
from tools.sequential_tasks import task_data  # noqa: E402
from tools.grokking import Model  # noqa: E402

MATS = ["block.q.weight", "block.k.weight", "block.v.weight", "block.o.weight",
        "block.up.weight", "block.down.weight", "unembed.weight"]


def above_basis(W, seed, n_shuffles=4, tol=1.02):
    """Left and right above-edge singular bases of W.

    The right basis Vt[keep] (k x n) spans the input directions A's circuit reads; projecting a
    gradient against it preserves W's action on those directions exactly. That alone is the weak
    form of the intervention: it does not stop the new task writing *into* the output channels A
    uses, so a later layer's reading of those channels still changes. The left basis U[:, keep].T
    (k x m) spans those output directions, and removing both sides -- (I - P_U) G (I - P_V) -- is
    the strict form.
    """
    W64 = np.asarray(W, np.float64)
    U, s, Vt = np.linalg.svd(W64, full_matrices=False)
    lam = s ** 2 / max(W64.shape)
    rng = np.random.default_rng(seed)
    edge = max(spectrum(shuffled(W64, rng))[0] for _ in range(n_shuffles)) * tol
    keep = lam > edge
    return Vt[keep], U[:, keep].T, int(keep.sum())


def rand_basis(k, n, rng):
    if k <= 0:
        return np.zeros((0, n))
    Q, _ = np.linalg.qr(rng.standard_normal((n, k)))
    return Q.T[:k]


def task(p, kind, frac, seed):
    pairs = torch.tensor([(i, j) for i in range(p) for j in range(p)])
    labels = ((pairs[:, 0] + pairs[:, 1]) if kind == "add" else (pairs[:, 0] * pairs[:, 1])) % p
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(pairs), generator=g)
    n = int(frac * len(pairs))
    return pairs, labels, perm[:n], perm[n:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1", default="/Users/pg/frcs-678/runs/grok_ckpt/ckpt")
    ap.add_argument("--phase1-epoch", type=int, default=72000)
    ap.add_argument("--p", type=int, default=113)
    ap.add_argument("--frac", type=float, default=0.3)
    ap.add_argument("--frac-b", type=float, default=0.30)
    ap.add_argument("--steps", type=int, default=8000); ap.add_argument("--recover", type=int, default=3000)
    ap.add_argument("--every", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--wd", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    # The interfering task was hardcoded to multiplication. The random-subspace control refuted
    # specificity for that one task; whether it does so for every task is a separate question, and
    # multiplication is the one task we know needs genuinely different machinery from addition.
    ap.add_argument("--task-a", default="add")
    ap.add_argument("--task-b", default="mul")
    ap.add_argument("--arms", default="none,protect_right,protect_strict,random_strict")
    ap.add_argument("--out", default="results/live_learning/protect.csv")
    a = ap.parse_args()

    import glob
    cs = sorted(c for c in glob.glob(os.path.join(a.phase1, "ep*.pt")) if not c.endswith(".opt"))
    eps = [int(os.path.basename(c)[2:8]) for c in cs]
    i = min(range(len(eps)), key=lambda j: abs(eps[j] - a.phase1_epoch))
    anc = torch.load(cs[i], map_location="cpu")
    d = anc["block.q.weight"].shape[0]

    pa, la, tra, tea = task_data(a.p, a.task_a, a.frac, a.seed)
    pb, lb, trb, teb = task_data(a.p, a.task_b, a.frac_b, a.seed + 1)

    bases, lbases, ks = {}, {}, {}
    for n in MATS:
        V, Ul, k = above_basis(anc[n].numpy(), a.seed)
        bases[n], lbases[n], ks[n] = V, Ul, k
    print(f"ancestor epoch {eps[i]}, protecting {sum(ks.values())} directions: "
          + " ".join(f"{n.split('.')[-2] if 'block' in n else 'unemb'}:{ks[n]}" for n in MATS))

    rng = np.random.default_rng(a.seed)
    rbases = {n: rand_basis(ks[n], anc[n].shape[1], rng) for n in MATS}
    rlbases = {n: rand_basis(ks[n], anc[n].shape[0], rng) for n in MATS}

    rows = []
    for arm in a.arms.split(","):
        model = Model(a.p, d=d); model.load_state_dict(anc); model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
        T = lambda x: torch.tensor(x, dtype=torch.float32)
        if arm == "none":
            PV = PU = None
        elif arm == "protect_right":          # weak form: input side only
            PV, PU = {n: T(bases[n]) for n in MATS}, None
        elif arm == "protect_strict":         # strict form: (I - P_U) G (I - P_V)
            PV, PU = {n: T(bases[n]) for n in MATS}, {n: T(lbases[n]) for n in MATS}
        elif arm == "random_right":           # the matched control for protect_right: random
            # subspace of the same per-matrix rank, applied on the input side only. Without this,
            # protect_right's 0.947 cannot be separated from the generic effect of constraining the
            # gradient at all -- random_strict already gives 0.739 but is two-sided and so not
            # matched.
            PV, PU = {n: T(rbases[n]) for n in MATS}, None
        elif arm == "random_strict":          # rank-matched random subspaces, both sides
            PV, PU = {n: T(rbases[n]) for n in MATS}, {n: T(rlbases[n]) for n in MATS}
        else:
            sys.exit(f"unknown arm {arm}")
        removed_frac = []
        acc = lambda pr, lb_, idx: float((model(pr[idx]).argmax(-1) == lb_[idx]).float().mean())
        print(f"\n--- arm: {arm} ---")
        # k and the overlap with the ancestor, logged DURING interference, not only at the end.
        # pg's question (21 Oct): both constrained arms take B to 1.0000 on held-out data, so B
        # generalises -- but does it consolidate? If B groks behaviourally in every arm while k
        # collapses only in the unconstrained one, the constraint lets the new task generalise
        # without evicting the old one's arrangement, and that is the intervention's mechanism. If k
        # collapses everywhere, then B's consolidation does not explain why A's recoverability
        # differs by 0.32 against 0.95, and the re-tenanting account does not cover the intervention.
        # Either way it is the measurement that decides, and it was missing.
        print(f"{'step':>6}{'A_test':>9}{'B_test':>9}{'k':>7}{'ov_anc':>9}{'phase':>11}")
        for step in range(a.steps + 1):
            if step % a.every == 0:
                model.eval()
                with torch.no_grad():
                    cur = model.state_dict()
                    kk, ovv = [], []
                    for n in MATS:
                        V2, _, k2 = above_basis(cur[n].detach().numpy(), a.seed)
                        kk.append(k2)
                        m = min(len(V2), len(bases[n]))
                        ovv.append(float(np.mean(np.clip(np.linalg.svd(
                            V2[:m] @ bases[n][:m].T, compute_uv=False), 0, 1))) if m else float("nan"))
                    rows.append(dict(arm=arm, phase="interfere", step=step,
                                     acc_a=acc(pa, la, tea), acc_b=acc(pb, lb, teb),
                                     k_mean=float(np.mean(kk)), ov_anc=float(np.nanmean(ovv))))
                    print(f"{step:>6}{rows[-1]['acc_a']:>9.4f}{rows[-1]['acc_b']:>9.4f}"
                          f"{rows[-1]['k_mean']:>7.2f}{rows[-1]['ov_anc']:>9.4f}"
                          f"{'interfere':>11}", flush=True)
                model.train()
            if step == a.steps:
                break
            loss = F.cross_entropy(model(pb[trb]), lb[trb])
            opt.zero_grad(); loss.backward()
            if PV is not None:
                with torch.no_grad():
                    for n, prm in model.named_parameters():
                        if n not in PV or prm.grad is None or len(PV[n]) == 0:
                            continue
                        g0 = prm.grad.norm().item()
                        V = PV[n]
                        prm.grad -= (prm.grad @ V.T) @ V          # kill action on A's input dirs
                        if PU is not None and len(PU[n]) > 0:
                            Ul = PU[n]
                            prm.grad -= Ul.T @ (Ul @ prm.grad)    # and writes into A's output dirs
                        if g0 > 0:
                            removed_frac.append(1.0 - prm.grad.norm().item() / g0)
            opt.step()
        model.eval()
        with torch.no_grad():
            a_after, b_after = acc(pa, la, tea), acc(pb, lb, teb)
        # recovery: retrain on task A, no projection
        model.train()
        o2 = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
        best = a_after
        for step in range(1, a.recover + 1):
            loss = F.cross_entropy(model(pa[tra]), la[tra])
            o2.zero_grad(); loss.backward(); o2.step()
            if step % a.every == 0:
                model.eval()
                with torch.no_grad():
                    av = acc(pa, la, tea)
                best = max(best, av)
                print(f"{a.steps+step:>6}{av:>9.4f}{acc(pb, lb, teb):>9.4f}{'recover':>11}",
                      flush=True)
                model.train()
        sd_end = model.state_dict()
        kk, ovv = [], []
        for n in MATS:
            V2, _, k2 = above_basis(sd_end[n].detach().numpy(), a.seed)
            kk.append(k2)
            if ks[n]:
                m = min(len(V2), len(bases[n]))
                if m:
                    ovv.append(float(np.mean(np.clip(np.linalg.svd(
                        V2[:m] @ bases[n][:m].T, compute_uv=False), 0, 1))))
        k_end = float(np.mean(kk)); ov_end = float(np.nanmean(ovv)) if ovv else float("nan")
        rem = float(np.mean(removed_frac)) if removed_frac else 0.0
        rows.append(dict(arm=arm, phase="summary", a_after_interference=a_after, b_learned=b_after,
                         a_recovery_ceiling=best, k_end=k_end, ov_end=ov_end,
                         grad_frac_removed=rem))
        print(f"  => {arm}: A after {a_after:.4f}, B learned {b_after:.4f}, A recovers {best:.4f}, "
              f"k={k_end:.1f}, ov={ov_end:.3f}, grad removed {100*rem:.1f}%")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        # Union of keys, not rows[0]: the file now holds per-step trajectory rows as well as the
        # one summary row per arm, and they do not share a schema.
        keys = list(dict.fromkeys(k for r in rows for k in r))
        w = csv.DictWriter(fh, fieldnames=keys, restval="")
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {a.out}")
    print(f"\n{'arm':16}{'B learned':>11}{'A recovers':>12}{'k_end':>8}{'ov_end':>8}{'grad cut':>10}")
    for r in (x for x in rows if "b_learned" in x):   # rows now also hold per-step trajectories
        print(f"{r['arm']:16}{r['b_learned']:>11.4f}{r['a_recovery_ceiling']:>12.4f}"
              f"{r['k_end']:>8.1f}{r['ov_end']:>8.3f}{100*r['grad_frac_removed']:>9.1f}%")


if __name__ == "__main__":
    main()
