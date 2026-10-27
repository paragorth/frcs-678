"""Is the attribution's ranking specific? Controls for the restore test (27 Oct; programme review, Paper 1 §7.5).

The first restore test (`pythia_kind_restore.py`) set the top-N attributed parts against N random parts of
the same kinds and sizes. The review found that those controls share about half their weights with the top
set (56% at 160m), themselves carry 44-50% of the attributed rise, are scored on half of the attribution's
own probes, and are two draws. Here:
  * fresh probes (seed 4321; the attribution used 1234);
  * controls DISJOINT from the top set, with the same kind x size composition, 20 draws per N;
  * a bottom-N arm (the parts the attribution ranks last);
  * copy loss as well as accuracy, and the fraction of the loss recovered, (L_damaged - L) / (L_damaged - L_healthy);
  * a curve over random subsets of loss recovered against summed attributed share: if the attribution is
    causal, recovery follows the share, with a slope near 1, whichever parts carry it.
Parts follow the attribution CSV, including MLP neuron groups ("L4mlp3.in") from `pythia_attribution.py
--mlp-groups`, which the disjoint draws need (whole MLP matrices are too few to draw from).

    python tools/restore_controls.py --damaged runs/pythia_damaged/anti50.pt \
        --attribution results/live_learning/pythia160m_attribution_g16.csv --out results/live_learning/pythia160m_restore_controls.csv
"""
import argparse, csv, os, random, re, sys
from collections import Counter
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.pythia_induction import repeated  # noqa: E402

HEAD = re.compile(r"L(\d+)H(\d+)\.(\w+)$")
MLPG = re.compile(r"L(\d+)mlp(\d+)\.(in|out)$")
KINDS = ("attn_q", "attn_k", "attn_v", "attn_out", "mlp_in", "mlp_out")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-160m")
    ap.add_argument("--damaged", required=True); ap.add_argument("--attribution", required=True)
    ap.add_argument("--T", type=int, default=40); ap.add_argument("--n-probes", type=int, default=64)
    ap.add_argument("--probe-seed", type=int, default=4321)
    ap.add_argument("--Ns", default="10,20,40,80"); ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--curve", type=int, default=50, help="random subsets for the recovered-vs-attributed curve")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()
    L, H, d = model.config.num_hidden_layers, model.config.num_attention_heads, model.config.hidden_size
    hd = d // H
    healthy = {k: v.clone() for k, v in model.state_dict().items()}
    damaged = {k: v.float() for k, v in torch.load(a.damaged, map_location="cpu").items()}
    seqs = repeated(a.n_probes, a.T, 1000, 20000, torch.Generator().manual_seed(a.probe_seed))
    attr = [r for r in csv.DictReader(open(a.attribution))]
    G = 1 + max([int(m.group(2)) for r in attr for m in [MLPG.match(r["group"])] if m] or [0])

    def restore(parts):
        sd = {k: v.clone() for k, v in damaged.items()}
        for g in parts:
            m, mg = HEAD.match(g), MLPG.match(g)
            if m:
                l, h, comp = int(m.group(1)), int(m.group(2)), m.group(3)
                pre = f"gpt_neox.layers.{l}.attention."
                if comp == "out":
                    sd[pre + "dense.weight"].view(d, H, hd)[:, h] = healthy[pre + "dense.weight"].view(d, H, hd)[:, h]
                else:
                    c = "qkv".index(comp[0]); isb = comp.endswith("_b")
                    key = pre + "query_key_value." + ("bias" if isb else "weight")
                    shp = (H, 3, hd) if isb else (H, 3, hd, d)
                    sd[key].view(*shp)[h, c] = healthy[key].view(*shp)[h, c]
            elif mg:
                l, gi, side = int(mg.group(1)), int(mg.group(2)), mg.group(3)
                pre = f"gpt_neox.layers.{l}.mlp."
                if side == "in":
                    W = pre + "dense_h_to_4h.weight"; gs = healthy[W].shape[0] // G
                    sd[W][gi * gs:(gi + 1) * gs] = healthy[W][gi * gs:(gi + 1) * gs]
                else:
                    W = pre + "dense_4h_to_h.weight"; gs = healthy[W].shape[1] // G
                    sd[W][:, gi * gs:(gi + 1) * gs] = healthy[W][:, gi * gs:(gi + 1) * gs]
            else:
                sd[g] = healthy[g].clone()
        return sd

    def size(g):
        m, mg = HEAD.match(g), MLPG.match(g)
        if m:
            comp = m.group(3)
            return d * hd if comp == "out" else (hd if comp.endswith("_b") else hd * d)
        if mg:
            return d * (model.config.intermediate_size // G)
        return healthy[g].numel()

    @torch.no_grad()
    def metrics(sd):
        model.load_state_dict(sd)
        logits = model(seqs).logits[:, a.T - 1:2 * a.T - 1]; y = seqs[:, a.T:2 * a.T]
        acc = float((logits.argmax(-1) == y).float().mean())
        loss = float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1)))
        return acc, loss

    share = {r["group"]: float(r["share"]) for r in attr}
    kind_of = {r["group"]: r["kind"] for r in attr}
    pool = [r["group"] for r in attr if r["kind"] in KINDS]
    total = sum(size(g) for g in pool)
    ranked = sorted(pool, key=lambda g: -share[g])
    acc_h, Lh = metrics(healthy); acc_d, Ld = metrics(damaged)
    print(f"fresh probes (seed {a.probe_seed}): healthy copying {acc_h:.3f}, loss {Lh:.3f}; damaged {acc_d:.3f}, loss {Ld:.3f}", flush=True)
    rows = [dict(arm="healthy", copy_acc=acc_h, copy_loss=Lh, loss_recovered=1.0),
            dict(arm="damaged", copy_acc=acc_d, copy_loss=Ld, loss_recovered=0.0)]

    def score(arm, parts, n=None, draw=None, top=None):
        acc, loss = metrics(restore(parts))
        w = sum(size(g) for g in parts)
        r = dict(arm=arm, N=n, draw=draw, n_parts=len(parts), frac_weights=w / total,
                 attr_share=sum(share[g] for g in parts), copy_acc=acc, copy_loss=loss, loss_recovered=(Ld - loss) / (Ld - Lh),
                 weights_shared_with_top=(sum(size(g) for g in parts if g in top) / max(w, 1)) if top else None)
        rows.append(r)
        print(f"{arm:26} N={n!s:>4} draw={draw!s:>3}  parts {len(parts):>4} ({100 * r['frac_weights']:5.1f}% of weights, "
              f"{100 * r['attr_share']:5.1f}% of the attributed rise)  copying {acc:.3f}  loss recovered {r['loss_recovered']:+.3f}", flush=True)
        with open(a.out, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for rr in rows for k in rr)), restval=""); wr.writeheader(); wr.writerows(rows)

    for n in (int(x) for x in a.Ns.split(",")):
        top = ranked[:n]; topset = set(top)
        score("top", top, n)
        score("bottom", ranked[-n:], n)
        need = Counter((kind_of[g], size(g)) for g in top)
        rnd = random.Random(1000 + n)
        for i in range(a.draws):
            pick = []
            for (k, sz), c in need.items():
                cand = [g for g in pool if kind_of[g] == k and size(g) == sz and g not in topset]
                pick += rnd.sample(cand, min(c, len(cand)))
            score("disjoint matched random", pick, n, i, topset)
    rc = random.Random(7)
    for i in range(a.curve):
        m = rc.randint(5, min(200, len(pool)))
        score("random subset (curve)", rc.sample(pool, m), m, i)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
