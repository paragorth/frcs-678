"""Does the circuit's Fourier structure survive the moment the capability stops working?

The claim under test (pg, 22 Oct): *organisation leads behaviour when a capability is being built,
and lags it when it is being destroyed.*

The leading half is measured: the above-edge directions of the unembedding put 0.497 of their power
in a single Fourier frequency at epoch 40 000, 6.4x a random subspace of matched rank, while test
accuracy is still 0.255 -- essentially complete before the behavioural transition.

The lagging half is measured with DIFFERENT quantities: the direction count sits at 8.86 at step 200
of interference (from 9.29) and the subspace overlap at 0.9096 at step 400, while accuracy has
already gone 1.0000 -> 0.0389. Two quantities on two sides is not a symmetry, and a referee will say
so. The mass measure does not settle it either, because it crosses 0.50 some 39 600 epochs before
behaviour and 0.90 one thousand epochs after -- the threshold picks the answer.

So this measures the SAME quantity on the destruction side: Fourier concentration of the above-edge
directions, step by step, as the capability is destroyed. Against a rank-matched random subspace at
every step, since the ratio is what the acquisition figures quote.

If concentration holds near its grokked value while accuracy collapses, the same quantity leads in
building and lags in destroying, and the claim is clean. If it collapses with the accuracy, only the
count lags and the claim is much weaker.

    python tools/fourier_decay.py --steps 3000 --out results/live_learning/fourier_decay.csv
"""
from __future__ import annotations
import argparse, csv, glob, os, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.fourier_check import split_sides, fourier_profile  # noqa: E402
from tools.sequential_tasks import task_data  # noqa: E402
from tools.grokking import Model  # noqa: E402


def concentration(W, seed, n_random=8):
    """Power in the strongest frequency for the above-edge directions, and the ratio to a
    rank-matched random subspace. The ratio is what the acquisition numbers quote."""
    A, B, k = split_sides(W, seed)
    if k < 1:
        return float("nan"), float("nan"), 0
    above = float(np.mean([fourier_profile(v)[0] for v in A]))
    rng = np.random.default_rng(seed)
    rnd = []
    for _ in range(n_random):
        Q, _ = np.linalg.qr(rng.standard_normal((W.shape[0], k)))
        rnd.append(np.mean([fourier_profile(Q[:, j])[0] for j in range(k)]))
    r = float(np.mean(rnd))
    return above, (above / r if r > 0 else float("nan")), k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1", default="/Users/pg/frcs-678/runs/grok_ckpt/ckpt")
    ap.add_argument("--phase1-epoch", type=int, default=72000)
    ap.add_argument("--p", type=int, default=113)
    ap.add_argument("--frac", type=float, default=0.3); ap.add_argument("--frac-b", type=float, default=0.25)
    ap.add_argument("--task-b", default="mul")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--wd", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    # pg (22 Oct): the thing that keeps relearning cheap for 4 000 steps after the unembedding's
    # circuit is gone may be in the EMBEDDINGS, which MATS excludes and which nothing in this
    # programme has measured. tok.weight is indexed by token like the unembedding, so the same
    # Fourier test applies. Carrying tok/pos across is also what made the fork transplant work
    # (multiplication arrived at 0.9998 immediately, against 0.0082 without them).
    ap.add_argument("--matrix", default="unembed.weight")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cs = sorted(c for c in glob.glob(os.path.join(a.phase1, "ep*.pt")) if not c.endswith(".opt"))
    eps = [int(os.path.basename(c)[2:8]) for c in cs]
    i = min(range(len(eps)), key=lambda j: abs(eps[j] - a.phase1_epoch))
    sd0 = torch.load(cs[i], map_location="cpu")
    d = sd0["block.q.weight"].shape[0]

    pa, la, tra, tea = task_data(a.p, "add", a.frac, a.seed)
    pb, lb, trb, teb = task_data(a.p, a.task_b, a.frac_b, a.seed + 1)
    m = Model(a.p, d=d); m.load_state_dict(sd0)
    acc = lambda pr, lb_, idx: float((m(pr[idx]).argmax(-1) == lb_[idx]).float().mean())

    # log densely where the capability dies (every 20 steps to 400) and sparsely after
    marks = sorted(set(list(range(0, 401, 20)) + list(range(500, a.steps + 1, 250))))
    rows = []
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=a.wd)
    print(f"{'step':>6}{'A':>9}{'B':>9}{'k':>5}{'1st freq':>10}{'x random':>10}")
    nxt = 0
    for step in range(a.steps + 1):
        if step == marks[nxt] if nxt < len(marks) else False:
            m.eval()
            with torch.no_grad():
                A_, B_ = acc(pa, la, tea), acc(pb, lb, teb)
            conc, ratio, k = concentration(m.state_dict()[a.matrix].detach().numpy(), a.seed)
            rows.append(dict(step=step, acc_a=A_, acc_b=B_, k=k, first_freq=conc, x_random=ratio))
            print(f"{step:>6}{A_:>9.4f}{B_:>9.4f}{k:>5d}{conc:>10.4f}{ratio:>10.2f}", flush=True)
            m.train(); nxt += 1
        if step == a.steps:
            break
        loss = F.cross_entropy(m(pb[trb]), lb[trb])
        opt.zero_grad(); loss.backward(); opt.step()

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {a.out}")
    print("Reference, same matrix at the grokked checkpoint: 0.530 in the strongest frequency, 6.8x")
    print("random. During acquisition it reached 6.4x at epoch 40 000 while test accuracy was 0.255.")


if __name__ == "__main__":
    main()
