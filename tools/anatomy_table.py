"""Paper 1 §7.4-7.5 across models (25 Oct): the exact attribution of the copy-loss rise and the restore
tests, side by side, from results/live_learning/pythia{tag}_attribution.csv and _kind_restore.csv.

Per-weight blame is a kind's share of the rise divided by its share of the attention + MLP weights (the
restore CSV's frac_weights for "restore all <kind>"): above 1, a kind carries more of the damage than
its size. Induction-head layers come from pyinduction_damage50_{tag}.csv (pyinduction_damage50.csv for
the original 160m run).

    python tools/anatomy_table.py 160m 160m_s1 160m_s2 410m 1.4b qwen0.5b qwen1.5b
"""
import csv, os, re, sys
from collections import defaultdict

R = "results/live_learning"
KINDS = ("attn_q", "attn_k", "attn_v", "attn_out", "mlp_in", "mlp_out")
NAMES = dict(attn_q="queries", attn_k="keys", attn_v="values", attn_out="attn out", mlp_in="MLP in", mlp_out="MLP out")


def load(tag):
    attr = list(csv.DictReader(open(f"{R}/pythia{tag}_attribution.csv")))
    rest = {r["arm"]: r for r in csv.DictReader(open(f"{R}/pythia{tag}_kind_restore.csv"))}
    dmg = f"{R}/pyinduction_damage50_{tag}.csv" if tag != "160m" else f"{R}/pyinduction_damage50.csv"
    heads = []
    if os.path.exists(dmg):
        cols = next(csv.reader(open(dmg)))
        heads = sorted({int(m.group(1)) for c in cols for m in [re.match(r"indscore_L(\d+)H\d+", c)] if m})
    return attr, rest, heads


def main():
    tags = sys.argv[1:]
    print("| model | rise (nats) | " + " | ".join(NAMES[k] for k in KINDS) + " | embed + unembed | largest part |")
    print("|---|---|" + "---|" * (len(KINDS) + 2))
    dens = {}
    for t in tags:
        attr, rest, _ = load(t)
        sh = defaultdict(float)
        for r in attr:
            sh[r["kind"]] += float(r["share"]) * 100
        rise = sum(float(r["attribution"]) for r in attr)
        top = max(attr, key=lambda r: float(r["share"]))
        fw = {k: float(rest[f"restore all {k}"]["frac_weights"]) * 100 for k in KINDS
              if "frac_weights" in rest.get(f"restore all {k}", {})}
        dens[t] = {k: sh[k] / fw[k] for k in fw if fw[k] > 0}
        print(f"| {t} | {rise:.2f} | " + " | ".join(f"{sh[k]:.0f}%" for k in KINDS)
              + f" | {sh['embed'] + sh['unembed']:+.1f}% | {float(top['share']) * 100:.1f}% ({top['group'].replace('gpt_neox.', '').replace('model.', '')}) |")
    if any(dens.values()):
        print("\nper-weight blame (share of the rise / share of the weights)")
        print("| model | " + " | ".join(NAMES[k] for k in KINDS) + " |")
        print("|---|" + "---|" * len(KINDS))
        for t in tags:
            if dens[t]:
                print(f"| {t} | " + " | ".join(f"{dens[t].get(k, float('nan')):.2f}" for k in KINDS) + " |")
    print("\nby layer: induction-head layers, the three largest layers, first and last layer")
    for t in tags:
        attr, _, heads = load(t)
        by = defaultdict(float)
        for r in attr:
            if int(r["layer"]) >= 0:
                by[int(r["layer"])] += float(r["share"]) * 100
        L = max(by) + 1
        big = sorted(by, key=lambda l: -by[l])[:3]
        print(f"  {t:9} L = {L:2}  induction heads in layers {heads}  largest " + ", ".join(f"L{l} {by[l]:.1f}%" for l in sorted(big))
              + f"  first {by[0]:.1f}%  last {by[L - 1]:.1f}%")
    print("\nrestores (copying; % of attention + MLP weights restored)")
    for t in tags:
        _, rest, _ = load(t)
        g = lambda arm: rest.get(arm)
        fmt = lambda r: (f"{float(r['induction_acc']):.3f}" + (f" ({100 * float(r['frac_weights']):.0f}%)" if r.get("frac_weights") else "")) if r else "—"
        print(f"  {t:9} damaged {fmt(g('damaged (nothing restored)'))}  best single kind "
              + max((f"{float(g(f'restore all {k}')['induction_acc']):.3f} {NAMES[k]}" for k in KINDS if g(f'restore all {k}')), key=lambda s: float(s.split()[0]))
              + f"  keys+MLP out {fmt(g('restore keys + MLP out'))}  k+v+MLP {fmt(g('restore keys + values + MLP in + MLP out'))}  all {fmt(g('restore all attention + all MLP'))}")
        for n in (10, 20, 40, 80):
            top = next((r for a, r in rest.items() if a.startswith(f"restore top-{n} attributed")), None)
            pr = [fmt(g(f"restore {n} random parts #{i}")) for i in range(2)]
            mr = [fmt(g(f"restore {n} matched random parts #{i}")) for i in range(2)]
            print(f"      top-{n:<2} {top['arm'].split('(')[1].split(' of')[0] if top else '':>4} of rise: {fmt(top)}   random {' / '.join(pr)}   matched random {' / '.join(mr)}")


if __name__ == "__main__":
    main()
