"""Two tasks that a single model can actually hold at once, via an operator token.

Everything in this programme so far used `f(a, b)` with a fixed '=' token, so add and mul presented
identical inputs with labels differing on 99.12% of pairs. One set of weights cannot satisfy both,
which made "retaining A while learning B" arithmetically impossible rather than merely hard, and
made the five-task sequence and the replay experiments unanswerable (AUTHOR_NOTES, 21 Oct 15:55).

Here the operator is an input token: f(a, op, b), vocabulary p + 2. Now f(3,+,5)=8 and f(3,x,5)=15
are both representable, so two questions become real:

  --mode joint       can one model grok both tasks together? (does the target of a search exist?)
  --mode sequential  grok A, then train B, and ask what happens to A -- the forgetting experiment as
                     it should have been run, with tasks that share weights but do not contradict
                     each other on a single input

The second is a different regime from everything measured today. With a shared `f(a,b)` the
interference is maximally adversarial: every example of B is a counterexample to A. With the operator
token, B's examples say nothing about A's answers. Whether the cliff survives that is unknown, and it
bounds the scope of every cost result we have.

    python tools/multitask.py --mode joint --epochs 60000 --out results/live_learning/joint.csv
"""
from __future__ import annotations
import argparse, copy, csv, os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.grokking import Block  # noqa: E402
from tools.sequential_tasks import above_basis, overlap  # noqa: E402

OPS = {"add": lambda a, b, p: (a + b) % p, "mul": lambda a, b, p: (a * b) % p,
       "sub": lambda a, b, p: (a - b) % p,
       # Ordering, not cyclic structure: no Fourier circuit over Z_p expresses max(a, b), so it is
       # the first task here that needs genuinely different machinery.
       # builtin max/min: build() passes Python ints, which torch.maximum rejects (killed cells
       # 108, 109 and 113 on 22 Oct)
       "maxab": lambda a, b, p: max(a, b),
       "minab": lambda a, b, p: min(a, b)}
MATS = ["block.q.weight", "block.k.weight", "block.v.weight", "block.o.weight",
        "block.up.weight", "block.down.weight", "unembed.weight"]


class OpModel(nn.Module):
    """f(a, op, b) -> class. Same body as the single-task model; the operator is a token, so the
    three positions are (a, op, b) and the answer is read off the last."""
    def __init__(self, p, n_ops, d=128, heads=4):
        super().__init__()
        self.tok = nn.Embedding(p + n_ops, d)   # numbers 0..p-1, then one token per operator
        self.pos = nn.Embedding(3, d)
        self.block = Block(d, heads)
        self.unembed = nn.Linear(d, p, bias=False)

    def forward(self, x):
        h = self.tok(x) + self.pos(torch.arange(x.shape[1], device=x.device))[None]
        return self.unembed(self.block(h)[:, -1])


def build_disjoint(p, ops, frac, seed):
    """Each rule owns its own slice of the input space, so no input ever has two answers.

    pg's remedy (22 Oct) and the better one. With f(a, b) the two tasks collide: f(3,5) must be 8 for
    addition and 15 for multiplication, which is not a function. The operator token fixes that by
    changing the input. This fixes it by partitioning the input: rule i owns the pairs whose first
    argument falls in band i, and the model has to infer which rule applies from the input itself.

    That is closer to what a capability is in a real model. A language model is not told "now do
    French"; French input gets French handling, code gets code handling, and the two never contradict
    each other because they never share an input. So "train on B, does A survive" means here what it
    means in the continual-learning literature, which it could not in the f(a, b) setting.
    """
    n = len(ops)
    width = p // n
    xs, ys, tag = [], [], []
    for oi, name in enumerate(ops):
        lo = oi * width
        hi = (oi + 1) * width if oi < n - 1 else p
        for a in range(lo, hi):
            for b in range(p):
                xs.append((a, b)); ys.append(OPS[name](a, b, p)); tag.append(oi)
    x = torch.tensor(xs); y = torch.tensor(ys); t = torch.tensor(tag)
    g = torch.Generator().manual_seed(seed)
    tr, te = [], []
    for oi in range(n):
        idx = (t == oi).nonzero(as_tuple=True)[0]
        perm = idx[torch.randperm(len(idx), generator=g)]
        cut = int(frac * len(perm))
        tr.append(perm[:cut]); te.append(perm[cut:])
    return x, y, t, torch.cat(tr), te


def build(p, ops, frac, seed):
    """One (a, op, b) row per operator per pair, with a shared train/test split per operator."""
    xs, ys, tag = [], [], []
    for oi, name in enumerate(ops):
        for a in range(p):
            for b in range(p):
                xs.append((a, p + oi, b)); ys.append(OPS[name](a, b, p)); tag.append(oi)
    x = torch.tensor(xs); y = torch.tensor(ys); t = torch.tensor(tag)
    g = torch.Generator().manual_seed(seed)
    tr, te = [], []
    for oi in range(len(ops)):
        idx = (t == oi).nonzero(as_tuple=True)[0]
        perm = idx[torch.randperm(len(idx), generator=g)]
        cut = int(frac * len(perm))
        tr.append(perm[:cut]); te.append(perm[cut:])
    return x, y, t, torch.cat(tr), te


def main():
    ap = argparse.ArgumentParser()
    # pg's idea (22 Oct): transplant a finished circuit, then train on BOTH tasks' data. Joint
    # training from scratch is slow -- our joint run had add 0.18 and mul 0.23 after 60 000 epochs --
    # and the transplant is known to save ~21 000 steps on a single task. So seed the joint training
    # with one task already grokked and measure the saving. This is the first proposal in which the
    # measurement is used to ACCELERATE rather than to warn, which is the half that works: structure
    # leads behaviour during construction.
    ap.add_argument("--mode", choices=["joint", "sequential", "warmjoint"], required=True)
    ap.add_argument("--split", choices=["optoken", "disjoint"], default="optoken",
                    help="optoken: f(a, op, b), the rule is named in the input. disjoint: f(a, b) "
                         "with each rule owning its own band of first arguments, so the model must "
                         "infer which rule applies.")
    ap.add_argument("--ops", default="add,mul")
    ap.add_argument("--p", type=int, default=113); ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--frac", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=60000)
    ap.add_argument("--phase2", type=int, default=20000, help="sequential mode: steps on task B")
    # pg's question (22 Oct): in every f(a,b) run, recovering task A costs task B -- B falls from
    # 1.0000 to 0.0088 as A returns. That is forced there, because the two tasks contradict each
    # other on 99% of inputs. With an operator token they do not, so phase 3 asks the question that
    # could not be asked before: retrain A and see whether B SURVIVES.
    #   both end high  -> coexistence is reachable by sequence; the loss was a detour
    #   A returns, B dies -> the model holds one at a time even with no contradiction, which is a
    #                        far stronger statement about forgetting than anything in the draft
    ap.add_argument("--phase3", type=int, default=0, help="steps back on task A after phase 2")
    # Self-generated replay, retested where it CAN work (22 Oct). The 21 Oct test used f(a, b) with
    # no operator, so replaying A demanded contradictory answers on B's inputs and no replay could
    # have succeeded. Here A and B are different inputs. Before phase 2 the model is frozen as its
    # own teacher; during phase 2 it also matches the teacher's output distribution (KL, i.e.
    # distillation from its old self) on a small fixed set of A-questions drawn from A's training
    # pairs. Labels come from the teacher, never from the data. A is still scored only on its
    # held-out pairs, none of which is ever replayed.
    ap.add_argument("--replay-frac", type=float, default=0.0,
                    help="sequential mode: replay set size as a fraction of all p*p A-questions")
    ap.add_argument("--replay-labels", choices=["self", "true"], default="self")
    ap.add_argument("--replay-weight", type=float, default=1.0)
    ap.add_argument("--phase1-stop", action="store_true",
                    help="sequential mode: end phase 1 2 000 steps after A's test accuracy passes 0.99")
    ap.add_argument("--every", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--wd", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ops = a.ops.split(",")
    torch.manual_seed(a.seed)
    if a.split == "disjoint":
        x, y, t, tr, te = build_disjoint(a.p, ops, a.frac, a.seed)
        model = OpModel(a.p, 1, d=a.d)        # no operator tokens needed
    else:
        x, y, t, tr, te = build(a.p, ops, a.frac, a.seed)
        model = OpModel(a.p, len(ops), d=a.d)
    print(f"{a.mode}: ops {ops}, p={a.p}, d={a.d}, {len(tr)} train rows, "
          f"{[len(e) for e in te]} held out per op", flush=True)

    def acc(idx):
        with torch.no_grad():
            return float((model(x[idx]).argmax(-1) == y[idx]).float().mean())

    def struct():
        sd = model.state_dict()
        ks = [above_basis(sd[n].detach().numpy(), np.random.default_rng(a.seed))[2] for n in MATS]
        return float(np.mean(ks))

    rows = []
    def log(phase, step):
        r = dict(phase=phase, step=step, k=struct(),
                 **{f"test_{o}": acc(te[i]) for i, o in enumerate(ops)},
                 **{f"train_{o}": acc(tr[t[tr] == i]) for i, o in enumerate(ops)})
        rows.append(r)
        print(f"  {phase:10}{step:>7}  k={r['k']:>5.2f}  " +
              "  ".join(f"{o} test {r['test_'+o]:.4f}/train {r['train_'+o]:.4f}" for o in ops),
              flush=True)

    if a.mode == "warmjoint":
        # phase 1: task A alone to grokking. phase 2: both tasks together, from there.
        trA = tr[t[tr] == 0]
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
        for ep in range(a.epochs + 1):
            if ep % a.every == 0:
                model.eval(); log("warm", ep); model.train()
                if rows[-1][f"test_{ops[0]}"] > 0.99:
                    print(f"  task {ops[0]} grokked at {ep}; switching to joint", flush=True)
                    break
            loss = F.cross_entropy(model(x[trA]), y[trA])
            opt.zero_grad(); loss.backward(); opt.step()
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
        for ep in range(a.phase2 + 1):
            if ep % a.every == 0:
                model.eval(); log("joint", ep); model.train()
                if all(rows[-1][f"test_{o}"] > 0.95 for o in ops):
                    print(f"  ALL TASKS above 0.95 at joint step {ep}", flush=True)
            loss = F.cross_entropy(model(x[tr]), y[tr])
            opt.zero_grad(); loss.backward(); opt.step()
    elif a.mode == "joint":
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
        for ep in range(a.epochs + 1):
            if ep % a.every == 0:
                model.eval(); log("joint", ep); model.train()
            loss = F.cross_entropy(model(x[tr]), y[tr])
            opt.zero_grad(); loss.backward(); opt.step()
    else:
        # phase 1: task A alone, to grokking. phase 2: task B alone, watching A.
        trA = tr[t[tr] == 0]; trB = tr[t[tr] == 1]
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
        grokked_at = None
        for ep in range(a.epochs + 1):
            if ep % a.every == 0:
                model.eval(); log("phase1", ep); model.train()
                if grokked_at is None and rows[-1][f"test_{ops[0]}"] > 0.99:
                    grokked_at = ep
                if a.phase1_stop and grokked_at is not None and ep >= grokked_at + 2000:
                    print(f"  phase 1 stopped at {ep} (A passed 0.99 at {grokked_at})", flush=True)
                    break
            loss = F.cross_entropy(model(x[trA]), y[trA])
            opt.zero_grad(); loss.backward(); opt.step()
        rep = None
        if a.replay_frac > 0:
            teacher = copy.deepcopy(model).eval()
            for prm in teacher.parameters():
                prm.requires_grad = False
            g = torch.Generator().manual_seed(a.seed + 7)
            n_rep = min(len(trA), int(a.replay_frac * a.p * a.p))
            rep = trA[torch.randperm(len(trA), generator=g)[:n_rep]]
            with torch.no_grad():
                t_logits = teacher(x[rep])
            agree = float((t_logits.argmax(-1) == y[rep]).float().mean())
            print(f"  replay: {n_rep} A-questions, labels={a.replay_labels}, weight "
                  f"{a.replay_weight}; teacher agrees with truth on {agree:.4f} of them", flush=True)
        sd0 = {n: model.state_dict()[n].clone() for n in MATS}
        ref = {n: above_basis(sd0[n].numpy(), np.random.default_rng(a.seed))[0] for n in MATS}
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
        for ep in range(a.phase2 + 1):
            if ep % a.every == 0:
                model.eval()
                cur = model.state_dict()
                ov = float(np.nanmean([overlap(
                    above_basis(cur[n].detach().numpy(), np.random.default_rng(a.seed))[0], ref[n])
                    for n in MATS]))
                log("phase2", ep); rows[-1]["ov_above"] = ov
                print(f"             ov_above={ov:.4f}", flush=True)
                model.train()
            loss = F.cross_entropy(model(x[trB]), y[trB])
            if rep is not None:
                out = model(x[rep])
                if a.replay_labels == "self":
                    loss = loss + a.replay_weight * F.kl_div(F.log_softmax(out, -1),
                                                              F.softmax(t_logits, -1), reduction="batchmean")
                else:
                    loss = loss + a.replay_weight * F.cross_entropy(out, y[rep])
            opt.zero_grad(); loss.backward(); opt.step()

    if a.mode == "sequential" and a.phase3:
        trA = tr[t[tr] == 0]
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
        for ep in range(a.phase3 + 1):
            if ep % a.every == 0:
                model.eval(); log("phase3", ep); model.train()
            loss = F.cross_entropy(model(x[trA]), y[trA])
            opt.zero_grad(); loss.backward(); opt.step()

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, restval=""); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
