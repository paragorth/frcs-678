"""Which weight changes carry the forgetting? The exact path-integral decomposition of
`tools/pythia_attribution.py`, for llama-style models (Qwen2 / Llama / Mistral: separate q / k / v with
grouped key-value heads, o_proj, gate / up / down; 25 Oct, Paper 1 §10's second family). k and v are
attributed per key-value head ("L{l}H{g}.k" with g the key-value head); tied embeddings count as "embed".

Original note (24 Oct):

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

    python tools/llama_attribution.py --model Qwen/Qwen2.5-0.5B-Instruct --damaged runs/pythia_damaged/anti50_qwen0.5b.pt \
        --out results/live_learning/qwen0.5b-it_attribution.csv
"""
import argparse, csv, os, sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.pythia_induction import repeated  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--damaged", required=True)
    ap.add_argument("--T", type=int, default=40); ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--K", type=int, default=16, help="midpoint quadrature points on the path")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32)
    cfg = model.config
    L, H, d = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.hidden_size
    hd = getattr(cfg, "head_dim", None) or d // H; KV = getattr(cfg, "num_key_value_heads", H)
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
        l = int(n.split(".layers.")[1].split(".")[0]) if ".layers." in n else -1
        for proj, nh, kind in (("q_proj", H, "attn_q"), ("k_proj", KV, "attn_k"), ("v_proj", KV, "attn_v")):
            if f"self_attn.{proj}." in n:
                vv = v.view(nh, hd, -1) if n.endswith("weight") else v.view(nh, hd)
                for h in range(nh):
                    add(f"L{l}H{h}.{kind[-1]}{'_b' if n.endswith('bias') else ''}", kind, l, h, vv[h].sum())
                break
        else:
            if "self_attn.o_proj.weight" in n:
                vv = v.view(-1, H, hd)
                for h in range(H):
                    add(f"L{l}H{h}.out", "attn_out", l, h, vv[:, h].sum())
            else:
                kind = ("mlp_in" if ("gate_proj" in n or "up_proj" in n) else "mlp_out" if "down_proj" in n else
                        "embed" if "embed_tokens" in n else "unembed" if "lm_head" in n else
                        "layernorm" if "norm" in n else "attn_out_bias" if "o_proj.bias" in n else "other")
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
