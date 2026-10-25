"""Does the attribution predict what restores the skill? Restore by kind, and by attributed rank (24 Oct).

The exact path-integral attribution (`pythia_attribution.py`) put the rise in copy loss mostly in
attention keys (34%) and MLP outputs (23%), spread over many heads and layers — which is why
restoring four induction heads did nothing. Test: in the damaged model, restore whole kinds (all
keys, all MLP outputs, both, ...) to their healthy values, and restore the top-N attributed parts
vs N random parts of the same kinds. The second is a first prototype of Paper 2's attribution-guided
rollback. Random-token probes only.

    python tools/pythia_kind_restore.py --damaged runs/pythia_damaged/anti50.pt \
        --attribution results/live_learning/pythia160m_attribution.csv --out results/live_learning/pythia160m_kind_restore.csv
"""
import argparse, csv, os, random, sys
from collections import Counter
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.pythia_induction import repeated, induction_acc, head_scores  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-160m")
    ap.add_argument("--damaged", required=True); ap.add_argument("--attribution", required=True)
    ap.add_argument("--T", type=int, default=40)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32, attn_implementation="eager").eval()
    L, H, d = model.config.num_hidden_layers, model.config.num_attention_heads, model.config.hidden_size
    hd = d // H
    healthy = {k: v.clone() for k, v in model.state_dict().items()}
    damaged = {k: v.float() for k, v in torch.load(a.damaged, map_location="cpu").items()}
    seqs = repeated(64, a.T, 1000, 20000, torch.Generator().manual_seed(1234))
    attr = list(csv.DictReader(open(a.attribution)))

    def restore(parts):
        """parts: list of group names as in the attribution CSV (L{l}H{h}.q/k/v[_b], L{l}H{h}.out, or a
        parameter name)."""
        sd = {k: v.clone() for k, v in damaged.items()}
        for g in parts:
            if g.startswith("L") and "H" in g and "." in g:
                lh, comp = g.split("."); l = int(lh[1:lh.index("H")]); h = int(lh[lh.index("H") + 1:])
                pre = f"gpt_neox.layers.{l}.attention."
                if comp == "out":
                    W = sd[pre + "dense.weight"].view(d, H, hd); W[:, h] = healthy[pre + "dense.weight"].view(d, H, hd)[:, h]
                else:
                    c = "qkv".index(comp[0]); isb = comp.endswith("_b")
                    key = pre + "query_key_value." + ("bias" if isb else "weight")
                    shp = (H, 3, hd) if isb else (H, 3, hd, d)
                    sd[key].view(*shp)[h, c] = healthy[key].view(*shp)[h, c]
            else:
                sd[g] = healthy[g].clone()
        return sd

    def size(g):
        """Weights in one part: a head's q / k / v rows (hd x d, or hd for a bias), its out columns (d x hd),
        or a whole parameter."""
        if g.startswith("L") and "H" in g and "." in g:
            comp = g.split(".")[1]
            return d * hd if comp == "out" else (hd if comp.endswith("_b") else hd * d)
        return healthy[g].numel()
    KINDS = ("attn_q", "attn_k", "attn_v", "attn_out", "mlp_in", "mlp_out")
    total = sum(size(r["group"]) for r in attr if r["kind"] in KINDS)

    rows = []
    def score(arm, parts):
        model.load_state_dict(restore(parts))
        acc, _ = induction_acc(model, seqs, a.T)
        ind_t, _ = head_scores(model, seqs, a.T)
        frac = sum(size(g) for g in parts) / total
        rows.append(dict(arm=arm, n_parts=len(parts), frac_weights=frac, induction_acc=acc, best_head_score=float(ind_t.max())))
        print(f"{arm:48} parts {len(parts):>4} ({100 * frac:5.1f}% of weights)  induction {acc:.3f}  best head score {float(ind_t.max()):.2f}", flush=True)
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k_ for r_ in rows for k_ in r_)), restval=""); w.writeheader(); w.writerows(rows)

    kinds = {}
    for r in attr:
        kinds.setdefault(r["kind"], []).append(r["group"])
    score("damaged (nothing restored)", [])
    for k in ("attn_q", "attn_k", "attn_v", "attn_out", "mlp_in", "mlp_out"):
        score(f"restore all {k}", kinds.get(k, []))
    score("restore keys + MLP out", kinds["attn_k"] + kinds["mlp_out"])
    score("restore keys + values + MLP in + MLP out", kinds["attn_k"] + kinds["attn_v"] + kinds["mlp_in"] + kinds["mlp_out"])
    score("restore all attention + all MLP", sum((kinds.get(k, []) for k in ("attn_q", "attn_k", "attn_v", "attn_out", "mlp_in", "mlp_out")), []))
    ranked = sorted(attr, key=lambda r: -float(r["attribution"]))
    pool = [r["group"] for r in attr if r["kind"] in ("attn_k", "attn_v", "attn_q", "attn_out", "mlp_in", "mlp_out")]
    rnd = random.Random(0)
    # matched draws on their own generator, so the plain draws stay the ones already reported
    rnd_m = random.Random(1)
    kind_of = {r["group"]: r["kind"] for r in attr}
    by_kind = {}
    for r in attr:
        by_kind.setdefault(r["kind"], []).append(r["group"])
    for n in (10, 20, 40, 80):
        top = [r["group"] for r in ranked[:n]]
        share = sum(float(r["share"]) for r in ranked[:n])
        score(f"restore top-{n} attributed ({100 * share:.0f}% of the rise)", top)
        for i in range(2):
            score(f"restore {n} random parts #{i}", rnd.sample(pool, n))
        need = Counter((kind_of[g], size(g)) for g in top)
        for i in range(2):
            pick = []
            for (k, sz), c in need.items():
                pick += rnd_m.sample([g for g in by_kind[k] if size(g) == sz], c)
            score(f"restore {n} matched random parts #{i}", pick)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
