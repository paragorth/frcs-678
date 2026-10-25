"""Which weight changes carry the forgetting? An exact decomposition by path integration (24 Oct).

Head-by-head restoration showed the induction heads' own q/k are not where copying was lost:
restoring all four leaves induction at 1.4% (damaged 0.9%, healthy 84%), and breaking them in the
healthy model leaves 80%. The damage is spread thinly over the network (Weyl: every piece moves
~0.1%). So attribute it exactly.

Along the straight line theta(alpha) = theta_0 + alpha (theta_1 - theta_0) from healthy (theta_0) to
damaged (theta_1), the fundamental theorem of calculus gives
    L(theta_1) - L(theta_0) = sum_i  integral_0^1  dL/dtheta_i (theta(alpha)) d alpha  * (theta_1 - theta_0)_i
— exactly, for every parameter i (the completeness property integrated gradients rely on). Summing the
terms inside each group (each head's q, k, v; each attention output; each MLP in/out; LayerNorms;
embedding; unembedding) splits the rise in copying loss among the parts, with no remainder beyond the
quadrature error, which is reported. L is the copy loss on random repeated sequences — probes only.

    python tools/pythia_attribution.py --damaged runs/pythia_damaged/anti50.pt --out results/live_learning/pythia160m_attribution.csv
"""
import argparse, csv, os, sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.pythia_induction import repeated  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-160m")
    ap.add_argument("--damaged", required=True)
    ap.add_argument("--T", type=int, default=40); ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--K", type=int, default=16, help="midpoint quadrature points on the path")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32)
    L, H, d = model.config.num_hidden_layers, model.config.num_attention_heads, model.config.hidden_size
    hd = d // H
    names = [n for n, _ in model.named_parameters()]
    th0 = {n: p.detach().clone() for n, p in model.named_parameters()}
    dmg = torch.load(a.damaged, map_location="cpu")
    delta = {n: dmg[n].float() - th0[n] for n in names}
    seqs = repeated(a.n, a.T, 1000, 20000, torch.Generator().manual_seed(1234))

    def copy_loss():
        logits = model(seqs).logits[:, a.T - 1:2 * a.T - 1]
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), seqs[:, a.T:2 * a.T].reshape(-1))

    attr = {n: torch.zeros_like(th0[n]) for n in names}
    params = dict(model.named_parameters())
    for k in range(a.K):
        al = (k + 0.5) / a.K
        with torch.no_grad():
            for n in names:
                params[n].copy_(th0[n] + al * delta[n])
        model.zero_grad(); loss = copy_loss(); loss.backward()
        for n in names:
            attr[n] += params[n].grad.detach() * delta[n] / a.K
        print(f"  alpha {al:.3f}: copy loss {float(loss):.3f}", flush=True)
    with torch.no_grad():
        for n in names: params[n].copy_(th0[n])
        L0 = float(copy_loss())
        for n in names: params[n].copy_(th0[n] + delta[n])
        L1 = float(copy_loss())
    total = sum(float(v.sum()) for v in attr.values())
    print(f"copy loss healthy {L0:.3f} -> damaged {L1:.3f}; rise {L1 - L0:.3f}; attributed {total:.3f} "
          f"(quadrature error {100 * (total - (L1 - L0)) / (L1 - L0):+.1f}%)", flush=True)

    rows = []
    def add(group, kind, layer, head, val):
        rows.append(dict(group=group, kind=kind, layer=layer, head=head, attribution=float(val),
                         share=float(val) / (L1 - L0)))
    for n in names:
        v = attr[n]
        if "query_key_value.weight" in n or "query_key_value.bias" in n:
            l = int(n.split(".")[2]); shp = (H, 3, hd, d) if n.endswith("weight") else (H, 3, hd)
            vv = v.view(*shp)
            for h in range(H):
                for c, nm in enumerate("qkv"):
                    add(f"L{l}H{h}.{nm}{'_b' if n.endswith('bias') else ''}", f"attn_{nm}", l, h, vv[h, c].sum())
        elif "attention.dense.weight" in n:
            l = int(n.split(".")[2]); vv = v.view(d, H, hd)
            for h in range(H):
                add(f"L{l}H{h}.out", "attn_out", l, h, vv[:, h].sum())
        else:
            kind = ("mlp_in" if "dense_h_to_4h" in n else "mlp_out" if "dense_4h_to_h" in n else
                    "embed" if "embed_in" in n else "unembed" if ("embed_out" in n or "lm_head" in n) else
                    "layernorm" if "layernorm" in n or "layer_norm" in n else "attn_out_bias" if "dense.bias" in n else "other")
            l = int(n.split(".")[2]) if ".layers." in n else -1
            add(n, kind, l, -1, v.sum())
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    by_kind = {}
    for r in rows: by_kind[r["kind"]] = by_kind.get(r["kind"], 0) + r["share"]
    print("share of the rise in copy loss, by kind:", {k: round(100 * v, 1) for k, v in sorted(by_kind.items(), key=lambda x: -x[1])}, flush=True)
    by_layer = {}
    for r in rows:
        if r["layer"] >= 0: by_layer[r["layer"]] = by_layer.get(r["layer"], 0) + r["share"]
    print("by layer (%):", {l: round(100 * v, 1) for l, v in sorted(by_layer.items())}, flush=True)
    top = sorted(rows, key=lambda r: -r["attribution"])[:20]
    print("top 20 parts:", [(r["group"], round(100 * r["share"], 1)) for r in top], flush=True)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
