"""Can you transplant from a fork back into its parent?

Parag's scheme (21 Oct): freeze the real model, fork it, let the fork learn the new task, read the
fork from its weights alone, and transplant a good fork back onto the parent. Its payoff rests on a
transplant step we have never measured.

What we have measured is the two ends of a range:
  * ancestor -> its own later self, one run: the above-edge implant reaches 0.9012 after 200 steps
  * an independently trained model of the same task: 0.0084, below the 0.0167 of doing nothing
A fork is neither. It shares the parent's entire history up to the fork point and then diverges onto
a different task, so whether it behaves like an ancestor or like a stranger is an open question and
it is the one the scheme depends on.

Design. Parent = the grokked task-A model, frozen. Fork = a copy trained on task B for N steps, so
it has B and has lost A. Then implant parts of the fork back into the parent and ask two things:
does B arrive, and does A survive? Controls are the same five as the temporal transplant, including
the rank-matched random subspace that made the earlier result interpretable.

    python tools/fork_transplant.py --task-b mul --fork-steps 8000 --out out.csv
"""
from __future__ import annotations
import argparse, csv, glob, os, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.sequential_tasks import task_data  # noqa: E402
from tools.protect_subspace import above_basis, rand_basis, MATS  # noqa: E402
from tools.grokking import Model  # noqa: E402


def split(W, V):
    """W's component inside the row space V, and outside it."""
    P = V.T @ V
    return W @ P, W @ (np.eye(W.shape[1]) - P)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1", default="/Users/pg/frcs-678/runs/grok_ckpt/ckpt")
    ap.add_argument("--phase1-epoch", type=int, default=72000)
    ap.add_argument("--task-a", default="add"); ap.add_argument("--task-b", default="mul")
    ap.add_argument("--p", type=int, default=113)
    ap.add_argument("--frac", type=float, default=0.3); ap.add_argument("--frac-b", type=float, default=0.25)
    # pg's design (22 Oct), and the one my first attempt got wrong. I forked from a parent that had
    # never trained on task B and implanted B's circuit into it; nothing took, because a transplant
    # only lands if the recipient is building the same circuit. Measurement ruled out the explanation
    # I had offered: the embeddings move 1.0692 over the 52 000 same-task steps that DO transplant
    # and 1.2112 over the 8 000 different-task steps that do not, so distance is not the difference.
    # --pre-steps trains the PARENT on B first, to the point where k starts to fall and the alarm
    # would have fired, then freezes it. The fork continues from there to full consolidation. The
    # parent is then mid-construction of the very circuit being implanted, which is structurally the
    # temporal case, and it never pays the consolidation that evicts A.
    ap.add_argument("--pre-steps", type=int, default=0,
                    help="steps of task B the PARENT takes before being frozen and forked")
    ap.add_argument("--fork-steps", type=int, default=8000)
    ap.add_argument("--settle", type=int, default=200,
                    help="steps of task-B training after the implant, matching the temporal "
                         "transplant's 200. 0 measures the implant with no training at all.")
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--wd", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    # MATS is the seven weight matrices and EXCLUDES tok.weight and pos.weight. In the temporal
    # transplant that was harmless -- donor and recipient were the same run 52 000 epochs apart, so
    # their embeddings were compatible. A fork that trained 8 000 steps on another task has diverged
    # embeddings, and leaving them behind made even the `full` arm transfer nothing (mul 0.0082 when
    # the fork itself has 1.0000). Without the embeddings the fork result says nothing about forks.
    ap.add_argument("--with-embeddings", action="store_true",
                    help="carry tok/pos embeddings across with the implant")
    ap.add_argument("--modes", default="none,above,below,randsub,full")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cks = sorted(c for c in glob.glob(os.path.join(a.phase1, "ep*.pt")) if not c.endswith(".opt"))
    eps = [int(os.path.basename(c)[2:8]) for c in cks]
    i = min(range(len(eps)), key=lambda j: abs(eps[j] - a.phase1_epoch))
    parent = torch.load(cks[i], map_location="cpu")
    d = parent["block.q.weight"].shape[0]

    pa, la, tra, tea = task_data(a.p, a.task_a, a.frac, a.seed)
    pb, lb, trb, teb = task_data(a.p, a.task_b, a.frac_b, a.seed + 1)
    acc = lambda sd, pr, lb_, idx: (
        lambda m: float((m(pr[idx]).argmax(-1) == lb_[idx]).float().mean()))(
            _load(sd, a.p, d))

    print(f"ancestor: epoch {eps[i]}, A={acc(parent, pa, la, tea):.4f}, B={acc(parent, pb, lb, teb):.4f}")
    if a.pre_steps:
        pm = Model(a.p, d=d); pm.load_state_dict(parent); pm.train()
        po = torch.optim.AdamW(pm.parameters(), lr=a.lr, weight_decay=a.wd)
        for _ in range(a.pre_steps):
            loss = F.cross_entropy(pm(pb[trb]), lb[trb])
            po.zero_grad(); loss.backward(); po.step()
        parent = {k: v.detach().clone() for k, v in pm.state_dict().items()}
        kk = float(np.mean([above_basis(parent[n].numpy(), a.seed)[2] for n in MATS]))
        print(f"parent frozen after {a.pre_steps} steps on {a.task_b}: "
              f"A={acc(parent, pa, la, tea):.4f}, B={acc(parent, pb, lb, teb):.4f}, k={kk:.2f}")

    # the fork: a copy of the parent trained on task B
    fork_m = Model(a.p, d=d); fork_m.load_state_dict(parent); fork_m.train()   # fork FROM the frozen parent
    opt = torch.optim.AdamW(fork_m.parameters(), lr=a.lr, weight_decay=a.wd)
    for _ in range(a.fork_steps):
        loss = F.cross_entropy(fork_m(pb[trb]), lb[trb])
        opt.zero_grad(); loss.backward(); opt.step()
    fork = {k: v.detach().clone() for k, v in fork_m.state_dict().items()}
    print(f"fork after {a.fork_steps} steps on {a.task_b}: "
          f"A={acc(fork, pa, la, tea):.4f}, B={acc(fork, pb, lb, teb):.4f}")

    rng = np.random.default_rng(a.seed)
    bases = {n: above_basis(fork[n].numpy(), a.seed)[0] for n in MATS}
    rbases = {n: rand_basis(len(bases[n]), fork[n].shape[1], rng) for n in MATS}
    print(f"fork's above-edge directions: {sum(len(bases[n]) for n in MATS)}\n")

    rows = []
    print(f"{'implant':10}{'B at 0':>9}{'A at 0':>9}{'B after':>9}{'A after':>9}")
    for mode in a.modes.split(","):
        ch = {k: v.clone() for k, v in parent.items()}
        if a.with_embeddings and mode != "none":
            for n in parent:
                if "tok" in n or "pos" in n:
                    ch[n] = fork[n].clone()
        for n in MATS:
            P, Fk = parent[n].numpy().astype(np.float64), fork[n].numpy().astype(np.float64)
            if mode == "none":
                M = P
            elif mode == "full":
                M = Fk
            elif mode == "above":
                M = split(Fk, bases[n])[0] + split(P, bases[n])[1]
            elif mode == "below":
                M = split(Fk, bases[n])[1] + split(P, bases[n])[0]
            elif mode == "randsub":
                M = split(Fk, rbases[n])[0] + split(P, rbases[n])[1]
            else:
                sys.exit(f"unknown mode {mode}")
            ch[n] = torch.tensor(M, dtype=parent[n].dtype)
        b0, a0 = acc(ch, pb, lb, teb), acc(ch, pa, la, tea)
        if a.settle:
            m = _load(ch, a.p, d); m.train()
            o = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=a.wd)
            for _ in range(a.settle):
                loss = F.cross_entropy(m(pb[trb]), lb[trb])
                o.zero_grad(); loss.backward(); o.step()
            ch = {k: v.detach().clone() for k, v in m.state_dict().items()}
        b1, a1 = acc(ch, pb, lb, teb), acc(ch, pa, la, tea)
        rows.append(dict(implant=mode, b_at_0=b0, a_at_0=a0, b_after=b1, a_after=a1,
                         fork_steps=a.fork_steps, settle=a.settle))
        print(f"{mode:10}{b0:>9.4f}{a0:>9.4f}{b1:>9.4f}{a1:>9.4f}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {a.out}")


def _load(sd, p, d):
    m = Model(p, d=d); m.load_state_dict(sd); m.eval(); return m


if __name__ == "__main__":
    main()
