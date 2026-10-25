"""Does forgetting show up as rotation of the old task's above-edge subspace?

Train task A (a+b mod p) to grokking, then train task B (a*b mod p) in the same weights with no task
indicator, and watch A decay. At every logging step, measure:

  * A's held-out accuracy  -- the forgetting curve, which needs task A's data
  * the principal angles between the current above-edge subspace and A's above-edge subspace at the
    end of phase 1 -- which needs only the weights

The question is whether the second predicts the first. If it does, forgetting of A is measurable
*without access to A*, which is the thing the existing literature cannot do: Varma and Nanda need
the task's Fourier basis, NESS needs the previous task's input covariance, and Natural Ungrokking
needs corpus support counts. All three are unavailable once a model is deployed.

Why rotation is the hypothesis. Natural Ungrokking (arXiv 2606.26050) finds the mechanism of
within-run forgetting is *displacement*: a competing pattern out-competes the rule and the
log-probability margin crosses zero within 100 steps of behavioural collapse. Khanh (arXiv
2607.06628) finds the weight *direction* carries circuit identity while the norm governs how
susceptible that identity is to being overwritten. Taken together, displacement of a circuit should
appear as its identity-carrying directions being rotated away -- and our null-split chimera result
(docs/AUTHOR_NOTES.md, 20 Oct) says the identity-carrying directions are the ones above the edge.

Controls that make the answer meaningful rather than decorative:
  * `--control none` trains phase 2 on task A again, so nothing should be forgotten and the overlap
    should stay high. Without this, any decay of overlap could just be continued training.
  * the below-edge subspace overlap is reported alongside, since the above-edge claim requires that
    the bulk does *not* track forgetting.
  * a random-subspace overlap of matched dimension, as a floor.

    python tools/sequential_tasks.py --phase1 runs/chimera/d128_f0.4_s0/ckpt --out results/live_learning/seq_add_mul.csv
"""
from __future__ import annotations
import argparse, csv, glob, math, os, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from semataxy.structure import shuffled, spectrum  # noqa: E402
from tools.grokking import Model  # noqa: E402

MATS = ["block.q.weight", "block.k.weight", "block.v.weight", "block.o.weight",
        "block.up.weight", "block.down.weight", "unembed.weight"]


def above_basis(W, rng, n_shuffles=4, tol=1.02):
    """(above-edge row basis, below-edge row basis, k). Row space, so comparable across checkpoints
    without worrying about left-singular-vector sign or permutation conventions."""
    W64 = np.asarray(W, np.float64)
    if W64.shape[0] < W64.shape[1]:
        W64 = W64.T
    n = W64.shape[0]
    U, s, Vt = np.linalg.svd(W64, full_matrices=False)
    lam = s ** 2 / n
    edge = max(spectrum(shuffled(W64, rng))[0] for _ in range(n_shuffles)) * tol
    keep = lam > edge
    return Vt[keep], Vt[~keep], int(keep.sum())


def overlap(A, B):
    """Mean cosine of principal angles between two row spaces. 1 = same span, 0 = orthogonal."""
    if A is None or B is None or len(A) == 0 or len(B) == 0:
        return float("nan")
    sv = np.clip(np.linalg.svd(A @ B.T, compute_uv=False), 0.0, 1.0)
    return float(np.mean(sv))


def random_basis(k, d, rng):
    if k <= 0:
        return None
    M = rng.standard_normal((k, d))
    Q, _ = np.linalg.qr(M.T)
    return Q.T[:k]


def task_data(p, kind, frac, seed):
    pairs = torch.tensor([(i, j) for i in range(p) for j in range(p)])
    if kind == "add":
        labels = (pairs[:, 0] + pairs[:, 1]) % p
    elif kind == "mul":
        labels = (pairs[:, 0] * pairs[:, 1]) % p
    elif kind == "sub":
        labels = (pairs[:, 0] - pairs[:, 1]) % p
    elif kind.startswith("mod"):
        # (a + b) mod q for a different modulus q. Same operation, same input space, comparable
        # difficulty -- but the circuit uses the characters of a different cyclic group, so there is
        # no shared Fourier structure for the new task to reuse. Parag's point: addition,
        # subtraction and multiplication mod 113 are all highly related, so "the capability always
        # recovers" may be a fact about related tasks. His prediction, recorded before running:
        # recovery here will be near zero.
        q = int(kind[3:])
        labels = (pairs[:, 0] + pairs[:, 1]) % q
    elif kind == "sq_sum":
        # (a^2 + b^2) mod p. Needed because every task this tool previously offered except mul is
        # additive -- aff<c> and mod<q> are addition with a scaled argument or a different modulus,
        # sub is addition with a sign. So "interference" has almost always meant a near-relative of
        # addition interfering with addition, and mul was the only task that caused any damage. A
        # squared form needs a different frequency structure, so it can serve either as a harder
        # capability to protect or as an interfering task that is not an addition variant.
        labels = (pairs[:, 0] ** 2 + pairs[:, 1] ** 2) % p
    elif kind in ("maxab", "minab", "gtab", "absdiff"):
        # Ordering tasks. pg's point (22 Oct): every task used so far is modular arithmetic, so they
        # all run on the characters of the same cyclic group -- which is why mul is the only one that
        # causes damage and the others reuse addition's circuit. Comparison needs ORDER, not cyclic
        # structure, and no Fourier circuit over Z_p expresses it. So these are the first tasks here
        # that require genuinely different machinery, which is what "English and multiplication"
        # means inside a model we can still measure.
        a_, b_ = pairs[:, 0], pairs[:, 1]
        if kind == "maxab":   labels = torch.maximum(a_, b_)
        elif kind == "minab": labels = torch.minimum(a_, b_)
        elif kind == "gtab":  labels = (a_ > b_).long()        # 2 classes, the rest unused
        else:                 labels = (a_ - b_).abs()
    elif kind == "quad":
        # Power et al. (arXiv 2201.02177) Appendix A.1.1 operation 6: x^2 + xy + y^2 mod p. Their
        # twelve operations are where the field's grokking results come from, and nine of them fit
        # this tool's (a, b) -> class format. Using their set rather than ones we invented means the
        # tasks are known to grok and the choice is not ours to defend.
        labels = (pairs[:, 0] ** 2 + pairs[:, 0] * pairs[:, 1] + pairs[:, 1] ** 2) % p
    elif kind == "quad_x":          # operation 7: x^2 + xy + y^2 + x
        labels = (pairs[:, 0] ** 2 + pairs[:, 0] * pairs[:, 1] + pairs[:, 1] ** 2 + pairs[:, 0]) % p
    elif kind == "cubic":           # operation 8: x^3 + xy
        labels = (pairs[:, 0] ** 3 + pairs[:, 0] * pairs[:, 1]) % p
    elif kind == "cubic2":          # operation 9: x^3 + xy^2 + y
        labels = (pairs[:, 0] ** 3 + pairs[:, 0] * pairs[:, 1] ** 2 + pairs[:, 1]) % p
    elif kind == "div":
        # operation 3: x / y mod p, which needs y invertible. y = 0 has no inverse, so those rows are
        # given label 0 and excluded from both splits by the caller's mask below.
        inv = torch.zeros(p, dtype=torch.long)
        for y in range(1, p):
            inv[y] = pow(y, p - 2, p)       # Fermat: y^(p-2) = y^-1 mod p for prime p
        labels = (pairs[:, 0] * inv[pairs[:, 1]]) % p
    elif kind == "sq_a_b":
        labels = (pairs[:, 0] ** 2 + pairs[:, 1]) % p
    elif kind.startswith("aff"):
        # (a + c*b) mod p. Same modulus, same operation count, same difficulty, and it groks -- but
        # it agrees with (a + b) mod p only where (c-1)*b == 0 mod p, i.e. on b == 0, which is 1/p of
        # the input. This is the task the mod<q> arm was supposed to be. mod 97 turned out to agree
        # with mod 113 on every pair with a + b < 97, which is 4753/12769 = 0.3722 of the input --
        # exactly the residual task-A accuracy observed after B consolidated. So mod<q> is a
        # *high*-coincidence task, not a low-coincidence one, and cannot test whether recovery
        # depends on the two rules being related. This can.
        c = int(kind[3:])
        labels = (pairs[:, 0] + c * pairs[:, 1]) % p
    elif kind == "random":
        # Random labels: the same number of examples and the same gradient scale as a real task,
        # but no rule to find, so no circuit can consolidate. This separates "the interfering task
        # formed a circuit" from "the interfering task simply had more data". Without it, the
        # low-data conditions are confounded -- they interfere less as well as consolidating less.
        g2 = torch.Generator().manual_seed(12345)
        labels = torch.randint(0, p, (len(pairs),), generator=g2)
    else:
        sys.exit(f"unknown task {kind}")
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(pairs), generator=g)
    n_tr = int(frac * len(pairs))
    return pairs, labels, perm[:n_tr], perm[n_tr:]


def acc(model, pairs, labels, idx):
    with torch.no_grad():
        return float((model(pairs[idx]).argmax(-1) == labels[idx]).float().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1", required=True, help="checkpoint dir from a grokked task-A run")
    ap.add_argument("--phase1-epoch", type=int, default=20000)
    ap.add_argument("--task-a", default="add"); ap.add_argument("--task-b", default="mul")
    ap.add_argument("--control", default="", choices=["", "none"],
                    help="'none' trains phase 2 on task A again -- nothing should be forgotten")
    ap.add_argument("--p", type=int, default=113); ap.add_argument("--frac", type=float, default=0.4)
    ap.add_argument("--frac-b", type=float, default=0.0,
                    help="separate data fraction for task B. Set below the critical threshold "
                         "(~0.25 for these tasks) so B can only memorise and never groks. Tests "
                         "whether permanent loss of A requires B to consolidate a circuit.")
    ap.add_argument("--steps", type=int, default=6000); ap.add_argument("--every", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--wd", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/live_learning/seq.csv")
    ap.add_argument("--readout-only", action="store_true",
                    help="during recovery, freeze everything except the unembedding. This is the "
                         "operation Trivedi and Melwani use (freeze backbone, retrain the linear "
                         "head), under which they report recoverability staying near 0.70 even at "
                         "65 degrees of subspace drift. If task A returns this way but not under "
                         "full-model retraining, the ceiling is a retraining-dynamics effect and "
                         "not information loss.")
    ap.add_argument("--recover", type=int, default=0,
                    help="after phase 2, retrain on task A for this many steps and log recovery. "
                         "Fast recovery means the circuit survived and forgetting was loss of "
                         "access; slow recovery means it was deleted.")
    # Recovery was previously logged every recover//20 steps, so a 20,000-step budget gave a first
    # reading at step 1,000 and a 40,000-step budget a first reading at 2,000. Every recovery cost
    # quoted so far is therefore an upper bound at that granularity, not a measurement -- the mod97
    # arm reached 1.0000 by its first reading, which bounds its cost only by 1,000 steps. Recovery
    # cost is the dependent variable of the whole cost law, so it needs its own resolution.
    # The cost of the cheap conditions is 200-400 steps against a 20 000-step budget, so ~98% of
    # those cells is spent training after the quantity being measured has already been determined.
    # Stopping a fixed margin past recovery keeps the post-recovery trajectory readable (the margin
    # is logged like any other stretch) while cutting the wasted part. 0 disables it, which is the
    # default, so every run already completed remains reproducible.
    ap.add_argument("--stop-after", type=int, default=0,
                    help="stop this many steps after task A first reaches --stop-acc (0 = never)")
    ap.add_argument("--stop-acc", type=float, default=0.99)
    ap.add_argument("--recover-every", type=int, default=0,
                    help="log recovery every this many steps (0 = the old recover//20 behaviour)")
    a = ap.parse_args()

    cks = sorted(c for c in glob.glob(os.path.join(a.phase1, "ep*.pt")) if not c.endswith(".opt"))
    if not cks:
        sys.exit(f"no checkpoints in {a.phase1}")
    eps = [int(os.path.basename(c)[2:8]) for c in cks]
    i = min(range(len(eps)), key=lambda j: abs(eps[j] - a.phase1_epoch))
    sd = torch.load(cks[i], map_location="cpu")
    d = sd["block.q.weight"].shape[0]
    print(f"phase 1 checkpoint: epoch {eps[i]} (d={d})")

    pa, la, tra, tea = task_data(a.p, a.task_a, a.frac, a.seed)
    task_b = a.task_a if a.control == "none" else a.task_b
    pb, lb, trb, teb = task_data(a.p, task_b, a.frac_b if a.frac_b else a.frac, a.seed + 1)

    model = Model(a.p, d=d); model.load_state_dict(sd); model.eval()
    print(f"task A ({a.task_a}) held-out acc at start: {acc(model, pa, la, tea):.4f}")
    print(f"task B ({task_b}) held-out acc at start: {acc(model, pb, lb, teb):.4f}"
          + ("   [control: phase 2 retrains task A]" if a.control == "none" else ""))

    rng = np.random.default_rng(a.seed)
    ref_above, ref_below, ref_k = {}, {}, {}
    for n in MATS:
        A_, B_, k = above_basis(sd[n].numpy(), np.random.default_rng(a.seed))
        ref_above[n], ref_below[n], ref_k[n] = A_, B_, k
    print(f"reference above-edge directions per matrix: "
          + " ".join(f"{n.split('.')[-2] if 'block' in n else 'unemb'}:{ref_k[n]}" for n in MATS))

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    rows = []
    print(f"\n{'step':>6}{'A_test':>8}{'A_TRAIN':>9}{'accB':>8}{'ov_above':>10}{'k_now':>7}")
    for step in range(a.steps + 1):
        if step % a.every == 0:
            model.eval()
            aa, ab = acc(model, pa, la, tea), acc(model, pb, lb, teb)
            aa_tr = acc(model, pa, la, tra)   # task A on its OWN TRAINING examples
            cur = model.state_dict()
            ovA, ovB, ovR, ks = [], [], [], []
            for n in MATS:
                Anow, Bnow, k = above_basis(cur[n].detach().numpy(),
                                            np.random.default_rng(a.seed))
                ovA.append(overlap(Anow, ref_above[n]))
                ovB.append(overlap(Bnow, ref_below[n]))
                ovR.append(overlap(Anow, random_basis(ref_k[n], Anow.shape[1], rng)
                                   if ref_k[n] else None))
                ks.append(k)
            f = lambda v: float(np.nanmean(v))
            row = dict(step=step, acc_a=aa, acc_a_train=aa_tr, acc_b=ab, ov_above=f(ovA),
                       ov_below=f(ovB), ov_random=f(ovR), k_mean=float(np.mean(ks)))
            rows.append(row)
            print(f"{step:>6}{aa:>8.4f}{aa_tr:>9.4f}{ab:>8.4f}{row['ov_above']:>10.4f}"
                  f"{row['k_mean']:>7.1f}", flush=True)
            model.train()
        if step == a.steps:
            break
        loss = F.cross_entropy(model(pb[trb]), lb[trb])
        opt.zero_grad(); loss.backward(); opt.step()

    if a.recover:
        if a.readout_only:
            for nm, prm in model.named_parameters():
                prm.requires_grad = nm.startswith("unembed")
            opt = torch.optim.AdamW([prm for prm in model.parameters() if prm.requires_grad],
                                    lr=a.lr, weight_decay=a.wd)
            live = [nm for nm, prm in model.named_parameters() if prm.requires_grad]
            print(f"\n--- phase 3: readout-only recovery, trainable = {live} ---")
        # Path and displacement, Parag's "energy spent" (21 Oct). Two models can sit at the same k
        # and the same accuracy 600 steps into recovery and still differ by 1 600 steps in what they
        # cost from there, so k and accuracy do not pin the state down. Path is the total distance
        # the weights travel, sum over steps of ||theta_t - theta_{t-1}||; displacement is the net
        # distance from where recovery started, ||theta_t - theta_0||. Their ratio is wasted motion:
        # a model that dismantles an occupying circuit before rebuilding should travel a long way to
        # end up not far, which a model that only builds should not. No SVD, so it is nearly free.
        flat = lambda: torch.cat([prm.detach().reshape(-1) for prm in model.parameters()]).clone()
        theta0 = flat(); prev = theta0.clone(); path = 0.0; recovered_at = None
        print(f"\n--- phase 3: retraining task A for {a.recover} steps ---")
        print(f"{'step':>6}{'accA':>8}{'accB':>8}{'ov_above':>10}{'k':>7}")
        for step in range(a.recover + 1):
            if step % max(1, a.recover_every or a.recover // 20) == 0:
                model.eval()
                cur = model.state_dict()
                ov, ks = [], []
                for n in MATS:
                    An, Bn, k = above_basis(cur[n].detach().numpy(), np.random.default_rng(a.seed))
                    ov.append(overlap(An, ref_above[n])); ks.append(k)
                now = flat()
                disp = float((now - theta0).norm())
                rows.append(dict(step=a.steps + step, acc_a=acc(model, pa, la, tea),
                                 acc_b=acc(model, pb, lb, teb), ov_above=float(np.nanmean(ov)),
                                 ov_below=float("nan"), ov_random=float("nan"),
                                 k_mean=float(np.mean(ks)), path=path, disp=disp,
                                 wasted=path / disp if disp > 0 else float("nan")))
                print(f"{a.steps + step:>6}{rows[-1]['acc_a']:>8.4f}{rows[-1]['acc_b']:>8.4f}"
                      f"{rows[-1]['ov_above']:>10.4f}{rows[-1]['k_mean']:>7.1f}", flush=True)
                model.train()
                if recovered_at is None and rows[-1]["acc_a"] >= a.stop_acc:
                    recovered_at = step
            if step == a.recover:
                break
            if a.stop_after and recovered_at is not None and step >= recovered_at + a.stop_after:
                print(f"  stopping at {a.steps + step}: reached {a.stop_acc} at {a.steps + recovered_at}, "
                      f"plus a {a.stop_after}-step margin", flush=True)
                break
            loss = F.cross_entropy(model(pa[tra]), la[tra])
            opt.zero_grad(); loss.backward(); opt.step()
            cur_flat = flat(); path += float((cur_flat - prev).norm()); prev = cur_flat

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        keys = list(dict.fromkeys(k for r in rows for k in r))   # union, in first-seen order:
        # the interference rows have no path/disp, so taking fieldnames from rows[0] silently
        # dropped the new columns and then raised on the recovery rows.
        w = csv.DictWriter(fh, fieldnames=keys, restval=""); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {a.out}")
    # the headline: does the weights-only overlap track the accuracy that needs task A's data?
    import scipy.stats as st
    A_ = [r["acc_a"] for r in rows]
    for k in ("ov_above", "ov_below", "ov_random"):
        v = [r[k] for r in rows]
        if len(set(v)) > 1 and len(set(A_)) > 1:
            rho, p = st.spearmanr(v, A_)
            print(f"  Spearman(task-A accuracy, {k}) = {rho:+.3f}  p={p:.2e}")


if __name__ == "__main__":
    main()
