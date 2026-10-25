"""Is the ~1/3 plateau a real basin, or an artefact of how we train?

Six of sixteen cells in the cliff sweep climb to 0.30-0.36 within ~1 200 recovery steps and then sit
there, unchanged, for another 18 000. pg's question (22 Oct): is not coming back the law, or a bug?

If the plateau is a genuine local optimum -- a partial circuit that is a real solution -- it should
hold whatever the optimiser does. If it is an artefact of weight decay 1.0, or of the learning rate,
or of Adam's accumulated moments, then changing only those should free it, and the "cliff in failure
rate" (0 of 6 cells below 5 400 interference steps, 6 of 10 above) would be a fact about our
hyperparameters rather than about the model.

The stuck state is reproducible because interference and recovery are both full-batch and
deterministic: train the grokked addition model on multiplication for 5 800 steps, then retrain on
addition until it plateaus. From there, each arm changes ONE thing and keeps training.

    python tools/escape_test.py --interfere 5800 --out results/live_learning/escape.csv
"""
from __future__ import annotations
import argparse, csv, glob, os, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.sequential_tasks import task_data, above_basis, MATS  # noqa: E402
from tools.grokking import Model  # noqa: E402

# One change each, against the setting the plateau was found under (lr 1e-3, weight decay 1.0).
ARMS = {
    "baseline  (lr 1e-3, wd 1.0)":      dict(lr=1e-3, wd=1.0, reset=False),
    "no weight decay":                  dict(lr=1e-3, wd=0.0, reset=False),
    "low lr (1e-4)":                    dict(lr=1e-4, wd=1.0, reset=False),
    "high lr (3e-3)":                   dict(lr=3e-3, wd=1.0, reset=False),
    "no decay + low lr":                dict(lr=1e-4, wd=0.0, reset=False),
    "fresh optimiser, same settings":   dict(lr=1e-3, wd=1.0, reset=True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1", default="/Users/pg/frcs-678/runs/grok_ckpt/ckpt")
    ap.add_argument("--phase1-epoch", type=int, default=72000)
    ap.add_argument("--p", type=int, default=113)
    ap.add_argument("--frac", type=float, default=0.3); ap.add_argument("--frac-b", type=float, default=0.25)
    ap.add_argument("--interfere", type=int, default=5800)
    ap.add_argument("--settle", type=int, default=3000, help="recovery steps to reach the plateau")
    ap.add_argument("--escape", type=int, default=5000, help="further steps per arm")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cks = sorted(c for c in glob.glob(os.path.join(a.phase1, "ep*.pt")) if not c.endswith(".opt"))
    eps = [int(os.path.basename(c)[2:8]) for c in cks]
    i = min(range(len(eps)), key=lambda j: abs(eps[j] - a.phase1_epoch))
    sd0 = torch.load(cks[i], map_location="cpu")
    d = sd0["block.q.weight"].shape[0]

    pa, la, tra, tea = task_data(a.p, "add", a.frac, a.seed)
    pb, lb, trb, teb = task_data(a.p, "mul", a.frac_b, a.seed + 1)
    acc = lambda m, pr, lb_, idx: float((m(pr[idx]).argmax(-1) == lb_[idx]).float().mean())
    kof = lambda sd: float(np.mean([above_basis(sd[n].detach().numpy(),
                                               np.random.default_rng(a.seed))[2] for n in MATS]))

    m = Model(a.p, d=d); m.load_state_dict(sd0); m.train()
    o = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1.0)
    for _ in range(a.interfere):
        loss = F.cross_entropy(m(pb[trb]), lb[trb]); o.zero_grad(); loss.backward(); o.step()
    m.eval(); print(f"after {a.interfere} steps of multiplication: A={acc(m, pa, la, tea):.4f}, "
                    f"B={acc(m, pb, lb, teb):.4f}, k={kof(m.state_dict()):.2f}", flush=True)

    m.train()
    o = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1.0)
    for _ in range(a.settle):
        loss = F.cross_entropy(m(pa[tra]), la[tra]); o.zero_grad(); loss.backward(); o.step()
    m.eval()
    stuck = {k: v.detach().clone() for k, v in m.state_dict().items()}
    opt_state = o.state_dict()
    base_a = acc(m, pa, la, tea)
    print(f"after {a.settle} recovery steps: A={base_a:.4f}, k={kof(stuck):.2f}", flush=True)
    if base_a > 0.8:
        print("\n  NOT STUCK -- this condition recovered, so there is no plateau to escape from.")
        print("  Pick an interference length that fails (frac-b 0.25: 5800 or 6200).")

    rows = []
    print(f"\n{'arm':34}{'A before':>10}{'A after':>10}{'k after':>9}   escaped?")
    for name, cfg in ARMS.items():
        mm = Model(a.p, d=d); mm.load_state_dict(stuck); mm.train()
        oo = torch.optim.AdamW(mm.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
        if not cfg["reset"]:
            try:
                oo.load_state_dict(opt_state)        # keep Adam's accumulated moments
                for g in oo.param_groups:
                    g["lr"], g["weight_decay"] = cfg["lr"], cfg["wd"]
            except Exception:
                pass
        for _ in range(a.escape):
            loss = F.cross_entropy(mm(pa[tra]), la[tra]); oo.zero_grad(); loss.backward(); oo.step()
        mm.eval()
        aft = acc(mm, pa, la, tea); kk = kof(mm.state_dict())
        esc = "ESCAPED" if aft > 0.8 else ("partly" if aft > base_a + 0.15 else "no")
        rows.append(dict(arm=name, lr=cfg["lr"], wd=cfg["wd"], reset=cfg["reset"],
                         a_before=base_a, a_after=aft, k_after=kk, escaped=esc))
        print(f"{name:34}{base_a:>10.4f}{aft:>10.4f}{kk:>9.2f}   {esc}", flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {a.out}")
    print("If no arm escapes, the plateau is a property of the model. If changing the optimiser")
    print("frees it, the cliff in failure rate is a fact about our hyperparameters.")


if __name__ == "__main__":
    main()
