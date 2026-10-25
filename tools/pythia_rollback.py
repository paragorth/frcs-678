"""Paper 2, R1 at real scale: attribution-guided rollback with a new task Pythia actually learns.

Old skill: induction (copy the token after the earlier occurrence). New, conflicting, learnable task:
copy-with-offset — on the second copy, predict the token TWO after the earlier occurrence. It
re-points the same circuit, so it conflicts with copying where the anti-induction task (never
learned) only suppressed it. Train from the pretrained model until the offset task is learned, then
two exact path integrals on the straight line healthy -> trained, accumulated per part:
  cost      rise in the copy loss on random repeated sequences (probes — no stored data)
  benefit   fall in the offset loss on a fixed training batch (the new task's own data)
Parts: per head q / k / v / o, MLP neuron groups (in rows + out columns), embedding, unembedding,
LayerNorms. Rollback rules at matched counts: by cost; "surgical" (cost-ranked among parts with
below-median benefit); random. Reports copying and offset accuracy. In the toy (rewiring regime)
old and new shared the same parts and no rollback kept both; does a large model separate them?

    python tools/pythia_rollback.py --lr 3e-6 --steps 400 --out results/live_learning/pythia160m_rollback_offset.csv
"""
import argparse, csv, os, random, sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.pythia_induction import repeated  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-160m")
    ap.add_argument("--lr", type=float, default=3e-6); ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=8); ap.add_argument("--T", type=int, default=40)
    ap.add_argument("--K", type=int, default=12); ap.add_argument("--mlp-groups", type=int, default=8)
    # --new-task rule (24 Oct): copy-with-offset re-points the induction circuit, so old and new share
    # parts and rollback only swaps them. The arithmetic rule ('a + b = c' as text) is a different
    # job: if it still damages copying, are the damaging parts separate from the ones it needs?
    ap.add_argument("--new-task", choices=["offset", "rule"], default="offset")
    ap.add_argument("--seed", type=int, default=0, help="training-batch seed (probes stay fixed)")
    # --no-rollback (25 Oct, Paper 1 §7.6 in a second family): only the training curve -- copying and the
    # new task every 50 steps, written to --out -- so any causal LM runs; the parts below are GPT-NeoX's.
    ap.add_argument("--no-rollback", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tools.real_model_forgetting import rule_batch, rule_accuracy
    tok = AutoTokenizer.from_pretrained(a.model); tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32)
    L, H, d = model.config.num_hidden_layers, model.config.num_attention_heads, model.config.hidden_size
    hd = d // H; T = a.T
    params = dict(model.named_parameters())
    th0 = {n: p.detach().clone() for n, p in params.items()}
    probe = repeated(32, T, 1000, 20000, torch.Generator().manual_seed(1234))
    newb = repeated(32, T, 1000, 20000, torch.Generator().manual_seed(4321))
    newb_rule = rule_batch(tok, 50, 32, "cpu", torch.Generator().manual_seed(4321))
    g = torch.Generator().manual_seed(a.seed)

    def losses(x):
        lg = model(x).logits[:, T:2 * T - 2]                       # positions with current token x_i
        copy = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), x[:, T + 1:2 * T - 1].reshape(-1))   # x_{i+1}
        if a.new_task == "rule":
            off = model(**newb_rule, labels=newb_rule["input_ids"]).loss if x is newb else copy * 0
        else:
            off = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), x[:, T + 2:2 * T].reshape(-1))    # x_{i+2}
        return copy, off

    @torch.no_grad()
    def accs(x):
        p = model(x).logits[:, T:2 * T - 2].argmax(-1)
        c = float((p == x[:, T + 1:2 * T - 1]).float().mean())
        if a.new_task == "rule":
            return c, rule_accuracy(model, tok, 50, "cpu")
        return c, float((p == x[:, T + 2:2 * T]).float().mean())

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    model.eval(); c, o = accs(probe); model.train()
    curve = [dict(step=0, copy=c, new_task=o)]
    print(f"step 0: copy {c:.3f} new-task {o:.3f}", flush=True)
    for step in range(1, a.steps + 1):
        if a.new_task == "rule":
            b = rule_batch(tok, 50, a.batch, "cpu", g)
            off = model(**b, labels=b["input_ids"]).loss
        else:
            x = repeated(a.batch, T, 1000, 20000, g)
            _, off = losses(x)
        opt.zero_grad(); off.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 50 == 0:
            model.eval(); c, o = accs(probe); model.train()
            curve.append(dict(step=step, copy=c, new_task=o))
            print(f"step {step}: copy {c:.3f} new-task {o:.3f}", flush=True)
    model.eval()
    if a.no_rollback:
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(curve[0])); w.writeheader(); w.writerows(curve)
        print(f"wrote {a.out}")
        return
    del opt                                                   # free AdamW state (~1.3 GB) before the path integrals
    th1 = {n: p.detach().clone() for n, p in params.items()}

    # parts: (name, param, index)
    P = []
    for l in range(L):
        pre = f"gpt_neox.layers.{l}."
        for h in range(H):
            for c, m in enumerate("qkv"):
                rows = slice(h * 3 * hd + c * hd, h * 3 * hd + (c + 1) * hd)   # fused layout [h][q|k|v]
                P.append((f"L{l}H{h}.{m}", pre + "attention.query_key_value.weight", (rows, slice(None))))
                P.append((f"L{l}H{h}.{m}_b", pre + "attention.query_key_value.bias", (rows,)))
            P.append((f"L{l}H{h}.o", pre + "attention.dense.weight", (slice(None), slice(h * hd, (h + 1) * hd))))
        gsz = 4 * d // a.mlp_groups
        for gi in range(a.mlp_groups):
            sl = slice(gi * gsz, (gi + 1) * gsz)
            P.append((f"L{l}mlp{gi}.in", pre + "mlp.dense_h_to_4h.weight", (sl, slice(None))))
            P.append((f"L{l}mlp{gi}.out", pre + "mlp.dense_4h_to_h.weight", (slice(None), sl)))
        for nm in ("input_layernorm.weight", "input_layernorm.bias", "post_attention_layernorm.weight", "post_attention_layernorm.bias"):
            P.append((f"L{l}.{nm}", pre + nm, (slice(None),)))
    P += [("embed", "gpt_neox.embed_in.weight", (slice(None), slice(None))),
          ("unembed", "lm_head.weight" if "lm_head.weight" in params else "embed_out.weight", (slice(None), slice(None))),
          ("final_ln_w", "gpt_neox.final_layer_norm.weight", (slice(None),)), ("final_ln_b", "gpt_neox.final_layer_norm.bias", (slice(None),))]
    cost = {nm: 0.0 for nm, _, _ in P}; benefit = {nm: 0.0 for nm, _, _ in P}
    for k in range(a.K):
        al = (k + 0.5) / a.K
        with torch.no_grad():
            for n in params: params[n].copy_(th0[n] + al * (th1[n] - th0[n]))
        for which, store in ((0, cost), (1, benefit)):
            model.zero_grad()
            copy_l, _ = losses(probe); _, off_l = losses(newb)
            (copy_l if which == 0 else off_l).backward()
            with torch.no_grad():
                for nm, pn, ix in P:
                    store[nm] += float((params[pn].grad[ix] * (th1[pn][ix] - th0[pn][ix])).sum()) / a.K
        print(f"  path point {k + 1}/{a.K}", flush=True)
    benefit = {nm: -v for nm, v in benefit.items()}                       # positive = lowered the offset loss
    with torch.no_grad():
        for n in params: params[n].copy_(th0[n])
        c0, _ = losses(probe); _, o0 = losses(newb)
        for n in params: params[n].copy_(th1[n])
        c1, _ = losses(probe); _, o1 = losses(newb)
    tc, tb = sum(cost.values()), sum(benefit.values())
    print(f"copy loss {float(c0):.3f} -> {float(c1):.3f} (attributed {tc:.3f}); {a.new_task} loss {float(o0):.3f} -> {float(o1):.3f} (attributed {-tb:.3f} fall)", flush=True)

    spec = {nm: (pn, ix) for nm, pn, ix in P}
    rows = []
    def evaluate(rule, names):
        with torch.no_grad():
            for n in params: params[n].copy_(th1[n])
            for nm in names:
                pn, ix = spec[nm]; params[pn][ix] = th0[pn][ix]
        c, o = accs(probe)
        r = dict(rule=rule, n=len(names), copy=c, offset=o, cost_covered=sum(cost[x] for x in names) / tc,
                 benefit_lost=sum(benefit[x] for x in names) / tb)
        rows.append(r)
        print(f"{rule:36} n={len(names):>4}  copy {c:.3f}  new-task {o:.3f}  cost covered {100 * r['cost_covered']:5.1f}%  benefit lost {100 * r['benefit_lost']:5.1f}%", flush=True)
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k_ for r_ in rows for k_ in r_)), restval=""); w.writeheader(); w.writerows(rows)
    names = [nm for nm, _, _ in P]
    by_cost = sorted(names, key=lambda x: -cost[x])
    med = sorted(benefit.values())[len(benefit) // 2]
    surgical = [x for x in by_cost if benefit[x] <= med]
    rnd = random.Random(0)
    evaluate("trained (nothing rolled back)", []); evaluate("all rolled back (= pretrained)", names)
    for n in (20, 40, 80, 160):
        evaluate(f"top-{n} by copy cost", by_cost[:n])
        evaluate(f"top-{n} surgical", surgical[:n])
        evaluate(f"random {n}", rnd.sample(names, n))
    print("largest copy costs (part, % of cost, % of offset benefit):",
          [(x, round(100 * cost[x] / tc, 1), round(100 * benefit[x] / tb, 1)) for x in by_cost[:15]], flush=True)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
