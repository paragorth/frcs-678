"""Paper 2, R1: attribution-guided rollback where the new task is learnable (toy, 24 Oct).

In Pythia the attribution's top parts restore copying monotonically while random parts do nothing,
but anti-induction is never learned, so there is nothing new to keep. Here it is: grokked addition,
then multiplication until learned (theta_1). Two exact path integrals on the same straight line
theta_0 -> theta_1, per part:
  cost_A    the rise in a DATA-FREE loss for the old skill: KL(old model || current) on random (a, b)
            probes — the model's own pre-forgetting answers, no stored addition data
  benefit_B the fall in the new task's training loss (we have the new data; we are learning it)
Parts: per head q / k / v / o, MLP neurons in groups of 32 (in and out rows together), embedding,
positions, unembedding. Rollback: return selected parts to theta_0. Rules compared at matched
counts: by cost_A alone; by cost_A among parts with small benefit_B ("surgical"); random.
Reports addition and multiplication test accuracy for each.

    python tools/toy_rollback.py --seed 0 --steps 5000 --out results/live_learning/toy_rollback_s0.csv
"""
import argparse, csv, os, random, sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.sequential_tasks import task_data, acc  # noqa: E402
from tools.grokking import Model  # noqa: E402


def parts_of(d=128, heads=4, group=32):
    """(name, param, index-builder) — each part is a slice of one parameter tensor."""
    hd = d // heads; P = []
    for h in range(heads):
        for m in "qkv":
            P.append((f"H{h}.{m}", f"block.{m}.weight", (slice(h * hd, (h + 1) * hd), slice(None))))
        P.append((f"H{h}.o", "block.o.weight", (slice(None), slice(h * hd, (h + 1) * hd))))
    for g in range(4 * d // group):
        sl = slice(g * group, (g + 1) * group)
        P.append((f"mlp{g}.in", "block.up.weight", (sl, slice(None))))
        P.append((f"mlp{g}.out", "block.down.weight", (slice(None), sl)))
    P += [("embed", "tok.weight", (slice(None), slice(None))), ("pos", "pos.weight", (slice(None), slice(None))),
          ("unembed", "unembed.weight", (slice(None), slice(None)))]
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/grok_ckpt/ckpt/ep072000.pt")
    ap.add_argument("--task-b", default="mul"); ap.add_argument("--frac-b", type=float, default=0.25)
    ap.add_argument("--steps", type=int, default=5000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--K", type=int, default=24); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    sd0 = torch.load(a.ckpt, map_location="cpu")
    pa, la, tra, tea = task_data(113, "add", 0.3, a.seed)
    pb, lb, trb, teb = task_data(113, a.task_b, a.frac_b, a.seed + 1)
    m = Model(113, d=128); m.load_state_dict(sd0)
    probes = torch.randint(0, 113, (2048, 2), generator=torch.Generator().manual_seed(99))
    with torch.no_grad():
        teacher = F.log_softmax(m(probes), -1)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1.0)
    for _ in range(a.steps):
        loss = F.cross_entropy(m(pb[trb]), lb[trb]); opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    print(f"after {a.steps} steps of {a.task_b}: add {acc(m, pa, la, tea):.4f}  {a.task_b} {acc(m, pb, lb, teb):.4f}", flush=True)
    th0 = {n: sd0[n].clone().float() for n, _ in m.named_parameters()}
    th1 = {n: p.detach().clone() for n, p in m.named_parameters()}
    delta = {n: th1[n] - th0[n] for n in th0}
    params = dict(m.named_parameters())

    def lossA():
        return F.kl_div(F.log_softmax(m(probes), -1), teacher, log_target=True, reduction="batchmean")
    def lossB():
        return F.cross_entropy(m(pb[trb]), lb[trb])

    attrA = {n: torch.zeros_like(th0[n]) for n in th0}; attrB = {n: torch.zeros_like(th0[n]) for n in th0}
    for k in range(a.K):
        al = (k + 0.5) / a.K
        with torch.no_grad():
            for n in th0: params[n].copy_(th0[n] + al * delta[n])
        for fn, store in ((lossA, attrA), (lossB, attrB)):
            m.zero_grad(); fn().backward()
            for n in th0: store[n] += params[n].grad.detach() * delta[n] / a.K
    with torch.no_grad():
        for n in th0: params[n].copy_(th0[n]);
        LA0, LB0 = float(lossA()), float(lossB())
        for n in th0: params[n].copy_(th1[n])
        LA1, LB1 = float(lossA()), float(lossB())
    tA = sum(float(v.sum()) for v in attrA.values()); tB = sum(float(v.sum()) for v in attrB.values())
    print(f"old-skill KL {LA0:.3f} -> {LA1:.3f} (attributed {tA:.3f}); new-task loss {LB0:.3f} -> {LB1:.3f} (attributed {tB:.3f})", flush=True)

    P = parts_of()
    cost = {nm: float(attrA[pn][ix].sum()) for nm, pn, ix in P}
    benefit = {nm: -float(attrB[pn][ix].sum()) for nm, pn, ix in P}      # positive = helped the new task
    rows = []
    def evaluate(rule, names):
        with torch.no_grad():
            for n in th0: params[n].copy_(th1[n])
            spec = {nm: (pn, ix) for nm, pn, ix in P}
            for nm in names:
                pn, ix = spec[nm]; params[pn][ix] = th0[pn][ix]
        r = dict(rule=rule, n=len(names), add=acc(m, pa, la, tea), new=acc(m, pb, lb, teb),
                 cost_share=sum(cost[x] for x in names) / tA, benefit_share=sum(benefit[x] for x in names) / -tB)
        rows.append(r)
        print(f"{rule:34} n={len(names):>3}  add {r['add']:.3f}  {a.task_b} {r['new']:.3f}  "
              f"cost covered {100 * r['cost_share']:5.1f}%  benefit lost {100 * r['benefit_share']:5.1f}%", flush=True)
    names = [nm for nm, _, _ in P]
    by_cost = sorted(names, key=lambda x: -cost[x])
    med_b = sorted(benefit.values())[len(benefit) // 2]
    surgical = [x for x in by_cost if benefit[x] <= med_b]
    rnd = random.Random(a.seed)
    evaluate("none (forgotten)", []); evaluate("all (original)", names)
    for n in (4, 8, 16, 32):
        evaluate(f"top-{n} by old-skill cost", by_cost[:n])
        evaluate(f"top-{n} surgical (low new-task use)", surgical[:n])
        evaluate(f"random {n}", rnd.sample(names, n))
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k_ for r_ in rows for k_ in r_)), restval=""); w.writeheader(); w.writerows(rows)
    top = sorted(names, key=lambda x: -cost[x])[:12]
    print("largest old-skill costs:", [(x, round(100 * cost[x] / tA, 1), "benefit", round(100 * benefit[x] / -tB, 1)) for x in top])
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
