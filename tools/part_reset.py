"""Put one part back. After add<-mul interference, restore one part of the model to its pre-
interference (grokked) values, clear that part's optimiser state, and measure the cost to relearn
addition. Recovery as in sequential_tasks.py: the interference optimiser carried over, full batch,
stop at 0.90, budget 20 000, logged every 100.

First run (laptop, 23 Oct, seeds 0-2 at 5 000 interference): restoring attention + MLP together
returns addition in 300 / 400 / 700 steps and everything except the embedding in 100 / 100 / 100,
while the embedding or the output layer alone is erratic. This is the cluster version, for more
seeds and for interference lengths before and after the cliff.

    python tools/part_reset.py --seed 3 --steps 7000 --out results/live_learning/part_reset_s3_7000.csv
"""
import argparse, copy, csv, os, sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.sequential_tasks import task_data, acc  # noqa: E402
from tools.grokking import Model  # noqa: E402

ATT = ["block.q.weight", "block.k.weight", "block.v.weight", "block.o.weight"]
MLP = ["block.up.weight", "block.down.weight"]
ARMS = [("control", []), ("embedding", ["tok.weight"]), ("positions", ["pos.weight"]),
        ("attention", ATT), ("MLP", MLP), ("whole block", ATT + MLP), ("output", ["unembed.weight"]),
        ("block + embedding", ["tok.weight"] + ATT + MLP),
        ("everything except embedding", ["pos.weight"] + ATT + MLP + ["unembed.weight"]),
        # the two ends of the first wires pulled (STATE 15): embedding->keys is read by k,
        # MLP->output is written by down. If these two alone match the whole block, they are the break.
        ("keys + MLP out", ["block.k.weight", "block.down.weight"]),
        ("keys only (k)", ["block.k.weight"]), ("MLP out only (down)", ["block.down.weight"])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/grok_ckpt/ckpt/ep072000.pt")
    ap.add_argument("--task-b", default="mul"); ap.add_argument("--frac", type=float, default=0.3)
    ap.add_argument("--frac-b", type=float, default=0.25)
    ap.add_argument("--steps", type=int, default=5000); ap.add_argument("--budget", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms", default="", help="|-separated arm names to run (default all)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    arms = [x for x in ARMS if not a.arms or x[0] in a.arms.split("|")]
    sd0 = torch.load(a.ckpt, map_location="cpu")
    d = sd0["block.q.weight"].shape[0]
    pa, la, tra, tea = task_data(113, "add", a.frac, a.seed)
    pb, lb, trb, teb = task_data(113, a.task_b, a.frac_b, a.seed + 1)
    m = Model(113, d=d); m.load_state_dict(sd0); m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1.0)
    for _ in range(a.steps):
        loss = F.cross_entropy(m(pb[trb]), lb[trb]); opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    print(f"seed {a.seed}, {a.steps} interference: A={acc(m, pa, la, tea):.4f} B={acc(m, pb, lb, teb):.4f}", flush=True)
    base_m, base_o = copy.deepcopy(m.state_dict()), copy.deepcopy(opt.state_dict())
    rows = []
    for name, parts in arms:
        m = Model(113, d=d); m.load_state_dict(base_m)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1.0); opt.load_state_dict(base_o)
        P = dict(m.named_parameters())
        with torch.no_grad():
            for n in parts:
                P[n].copy_(sd0[n]); opt.state[P[n]] = {}
        hit = None
        for s in range(a.budget + 1):
            if s % 100 == 0:
                m.eval(); A = acc(m, pa, la, tea); B = acc(m, pb, lb, teb); m.train()
                rows.append(dict(seed=a.seed, steps=a.steps, arm=name, step=s, acc_a=A, acc_b=B))
                if A >= 0.9:
                    hit = s; break
            loss = F.cross_entropy(m(pa[tra]), la[tra]); opt.zero_grad(); loss.backward(); opt.step()
        print(f"  {name:28} to 0.90: {hit if hit is not None else '>' + str(a.budget):>6}  end A={rows[-1]['acc_a']:.4f}", flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
