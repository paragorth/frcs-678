"""192-fact depth gate. Replaces the 4-fact France->China gate, which saturated.

Per PREREG-sharper-gate.md. Primary measure is SOURCE SUPPRESSION
(`swapped_rank_src`), the one statistic that had headroom in the previous sweep.
`rank_dst` is recorded so its saturation is confirmed rather than assumed.
"""
from __future__ import annotations
import argparse, contextlib, json, math, os, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from common import CONFIGS, OUT, add_model_arg, depth_plan, depth_plan_at, lens_path, load_model
from lens_ops import intervene
from restricted import RestrictedLens
import tokens as toks

ap = argparse.ArgumentParser(); add_model_arg(ap)
ap.add_argument("--tag", default=""); ap.add_argument("--inject", type=int, default=None)
ap.add_argument("--gen-tokens", type=int, default=6)
args = ap.parse_args()

NL = CONFIGS[args.model]["n_layers"]
plan = depth_plan(NL) if args.inject is None else depth_plan_at(args.inject, NL)
BAND = plan["band"]
model, hf, tok, device, dtype = load_model(args.model)
lens = RestrictedLens.load(lens_path(args.model, args.tag))
tracked, seed_only = toks.build(tok, OUT)
IDS = {**tracked, **seed_only}

REPO = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "jacobian-lens")
cats = json.load(open(os.path.join(REPO, "data/experiments/flexible-generalization.json")))["categories"]

ALIASES = {"yuan": {"yuan","renminbi","rmb","cny"}, "euro": {"euro","euros","eur"},
           "beijing": {"beijing","peking"}, "chinese": {"chinese","mandarin"},
           "north": {"north","america"}, "dollar": {"dollar","dollars"},
           "pound": {"pound","pounds"}}
acc = lambda f: ALIASES.get(f.strip().lower(), {f.strip().lower()})
names = lambda t, f: bool(set(re.findall(r"[a-z]+", t.lower())) & acc(f))


def usable(w):
    v = IDS.get(w)
    return v is not None and v.get("single") and v["id"] in lens.index


def acc_ids(fact):
    out = []
    for s in acc(fact):
        for form in (" "+s, s, " "+s.capitalize(), s.capitalize()):
            e = tok.encode(form, add_special_tokens=False)
            if e: out.append(e[0])
    return sorted(set(out))


@torch.no_grad()
def ask(prompt, stack=None):
    ids = model.encode(prompt, max_length=128)
    with contextlib.ExitStack() as st:
        for l, fn in (stack or []): st.enter_context(intervene(model, l, fn))
        o = hf.generate(ids, max_new_tokens=args.gen_tokens, do_sample=False,
                        pad_token_id=tok.eos_token_id)
        lg = hf(ids).logits[0, -1].float()
    return tok.decode(o[0][ids.shape[1]:], skip_special_tokens=True), lg


def rank(lg, fact):
    ids = acc_ids(fact)
    if not ids: return None
    order = lg.argsort(descending=True).tolist()
    pos = {t: i for i, t in enumerate(order)}
    return min(pos.get(i, 10**9) for i in ids) + 1


def swapper(d_src, d_dst):
    def fn(h):
        c = (h * d_src).sum(-1, keepdim=True)
        return h - c * d_src + c * d_dst
    return fn


rows, t0, skipped = [], time.time(), 0
for c in cats:
    ents = [e for e in c["funcs"][0]["answers"] if usable(e)]
    for src in ents:
        for dst in ents:
            if src == dst: continue
            ds = {l: (lens.direction(l, IDS[src]["id"]), lens.direction(l, IDS[dst]["id"]))
                  for l in BAND}
            for f in c["funcs"]:
                a_src, a_dst = f["answers"][src], f["answers"][dst]
                if not (acc_ids(a_src) and acc_ids(a_dst)): skipped += 1; continue
                p = f["template"].format(arg=src)
                gb, lb = ask(p)
                gs, ls = ask(p, [(l, swapper(*ds[l])) for l in BAND])
                rows.append({"cat": c["name"], "func": f["name"], "src": src, "dst": dst,
                             "baseline_correct": names(gb, a_src),
                             "swapped_correct": names(gs, a_dst),
                             "swapped_still_src": names(gs, a_src),
                             "baseline_rank_dst": rank(lb, a_dst),
                             "swapped_rank_dst": rank(ls, a_dst),
                             "swapped_rank_src": rank(ls, a_src)})
val = [r for r in rows if r["baseline_correct"]]
lg10 = lambda xs: sum(math.log10(max(x, 1)) for x in xs) / max(len(xs), 1)
summary = {"model": CONFIGS[args.model]["name"], "tag": args.tag,
           "inject_layer": plan["inject_layer"], "band_width": len(BAND),
           "n_facts": len(rows), "n_baseline_valid": len(val), "n_skipped": skipped,
           "swap_rate": sum(r["swapped_correct"] for r in val) / max(len(val), 1),
           "still_src_rate": sum(r["swapped_still_src"] for r in val) / max(len(val), 1),
           "mean_log10_rank_src": lg10([r["swapped_rank_src"] for r in val
                                        if r["swapped_rank_src"]]),
           "mean_log10_rank_dst": lg10([r["swapped_rank_dst"] for r in val
                                        if r["swapped_rank_dst"]]),
           "rank_dst_eq1_frac": sum(1 for r in val if r["swapped_rank_dst"] == 1)
                                / max(len(val), 1),
           "wall": time.time() - t0, "rows": rows}
json.dump(summary, open(os.path.join(OUT, f"gate_multi_{args.model}{args.tag}.json"), "w"), indent=1)
print(f"inject L{plan['inject_layer']} w={len(BAND)} | facts {len(rows)} valid {len(val)} "
      f"skip {skipped} | swap {summary['swap_rate']:.1%} still_src "
      f"{summary['still_src_rate']:.1%} | log10 rank_src {summary['mean_log10_rank_src']:.3f} "
      f"| rank_dst==1 {summary['rank_dst_eq1_frac']:.1%}  ({summary['wall']/60:.1f} min)")
