"""Which gate is responsible? Head-by-head restoration of a damaged Pythia (Paper 1, 24 Oct).

After 50 anti-induction steps copying is ~1% while the K-composition wiring is unchanged — a gate,
not a cut. Timing cannot say which head's gate carries the loss; interventions can. Using the
pretrained model (healthy) and the saved damaged model (`--damaged`), with random-token probes only:

  sufficiency  restore one induction head's query+key slice (weights and bias) in the damaged model
  necessity    put one induction head's damaged query+key slice into the healthy model
  both, for all induction heads together, and for size-matched random non-induction heads (control)
  query-only vs key-only for the top head; dose-response: interpolate the top head (and all four)
  between damaged and healthy at alpha = 0, 0.25, 0.5, 0.75, 1

    python tools/pythia_head_restore.py --damaged runs/pythia_damaged/anti50.pt --out results/live_learning/pythia160m_head_restore.csv
"""
import argparse, csv, os, random, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.pythia_induction import repeated, head_scores, induction_acc  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-160m")
    ap.add_argument("--damaged", required=True)
    ap.add_argument("--T", type=int, default=40); ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32, attn_implementation="eager").eval()
    L, H, d = model.config.num_hidden_layers, model.config.num_attention_heads, model.config.hidden_size
    hd = d // H
    healthy = {k: v.clone() for k, v in model.state_dict().items()}
    damaged = {k: v.float() for k, v in torch.load(a.damaged, map_location="cpu").items()}
    seqs = repeated(64, a.T, 1000, 20000, torch.Generator().manual_seed(1234))
    ind, prev = head_scores(model, seqs, a.T)
    flat = lambda s: [(int(i) // H, int(i) % H) for i in torch.argsort(s.flatten(), descending=True)]
    IH = flat(ind)[:a.n_heads]
    print("induction heads:", IH, flush=True)

    def qk_keys(l):
        p = f"gpt_neox.layers.{l}.attention.query_key_value."
        return p + "weight", p + "bias"

    def build(base, donor, heads, parts="qk", alpha=1.0):
        """base with the given heads' q/k slices moved alpha of the way to donor's."""
        sd = {k: v.clone() for k, v in base.items()}
        for l, h in heads:
            wk, bk = qk_keys(l)
            W = sd[wk].view(H, 3, hd, d); Wd = donor[wk].view(H, 3, hd, d)
            B = sd[bk].view(H, 3, hd); Bd = donor[bk].view(H, 3, hd)
            for c, name in ((0, "q"), (1, "k")):
                if name in parts:
                    W[h, c] = (1 - alpha) * W[h, c] + alpha * Wd[h, c]
                    B[h, c] = (1 - alpha) * B[h, c] + alpha * Bd[h, c]
        return sd

    rows = []
    def score(arm, sd):
        model.load_state_dict(sd)
        acc, _ = induction_acc(model, seqs, a.T)
        ind_t, _, sink_t, _, mar_t = head_scores(model, seqs, a.T, gate=True)
        r = dict(arm=arm, induction_acc=acc)
        for l, h in IH:
            r[f"indscore_L{l}H{h}"] = float(ind_t[l, h]); r[f"gate_margin_L{l}H{h}"] = float(mar_t[l, h])
            r[f"sink_L{l}H{h}"] = float(sink_t[l, h])
        rows.append(r)
        print(f"{arm:42} induction {acc:.3f}  margins " + " ".join(f"{r[f'gate_margin_L{l}H{h}']:+.1f}" for l, h in IH), flush=True)
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k_ for r_ in rows for k_ in r_)), restval=""); w.writeheader(); w.writerows(rows)

    score("healthy", healthy); score("damaged", damaged)
    for l, h in IH:
        score(f"restore L{l}H{h} q+k", build(damaged, healthy, [(l, h)]))
    score("restore all induction q+k", build(damaged, healthy, IH))
    top = IH[0]
    score(f"restore L{top[0]}H{top[1]} q only", build(damaged, healthy, [top], "q"))
    score(f"restore L{top[0]}H{top[1]} k only", build(damaged, healthy, [top], "k"))
    for l, h in IH:
        score(f"break L{l}H{h} q+k in healthy", build(healthy, damaged, [(l, h)]))
    score("break all induction q+k in healthy", build(healthy, damaged, IH))
    rnd = random.Random(0); others = [(l, h) for l in range(L) for h in range(H) if (l, h) not in IH]
    for i in range(3):
        pick = rnd.sample(others, a.n_heads)
        score(f"restore 4 random heads #{i} q+k", build(damaged, healthy, pick))
        score(f"break 4 random heads #{i} in healthy", build(healthy, damaged, pick))
    for al in (0.25, 0.5, 0.75):
        score(f"dose top head alpha={al}", build(damaged, healthy, [top], alpha=al))
        score(f"dose all induction alpha={al}", build(damaged, healthy, IH, alpha=al))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
