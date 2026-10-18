"""POSITIVE CONTROL: does the SAME token-aligned patching pipeline transfer an
item edit?

France->China is the one result from the source paper that replicates cleanly
(Nanda's commentary). If activation patching at token-aligned shared positions
flips the answer to China's, the pipeline works and the skill's 0% isolates
item-vs-procedure. If this ALSO fails token-aligned, the pipeline has a further
problem and the skill result is not interpretable.

Deliberately the same alignment function, same hook shape, same extents as the
skill runs. The only change is the prompt pair: instead of (with-file, no-file)
it is (China-prompt, France-prompt). The shared token suffix is everything after
the country name.
"""
from __future__ import annotations
import argparse, json, os, re, sys, contextlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from common import CONFIGS, OUT, add_model_arg, depth_plan, load_model
from lens_ops import intervene
from jlens.hooks import ActivationRecorder

ap = argparse.ArgumentParser(); add_model_arg(ap)
ap.add_argument("--src", default="France"); ap.add_argument("--dst", default="China")
ap.add_argument("--deferred", action="store_true",
                help="Move the readout PAST the patch boundary, matching the skill "
                     "harness. Without this the answer is the very next token after "
                     "the last patched position, so patching layer 27 there just "
                     "overwrites the next-token distribution and 4/4 is guaranteed.")
args = ap.parse_args()
NL = CONFIGS[args.model]["n_layers"]; PLAN = depth_plan(NL)
EXTENTS = {"band": PLAN["band"], "all": list(range(NL))}
model, hf, tok, device, dtype = load_model(args.model)
CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cats = json.load(open(os.path.join(CODE, "jacobian-lens",
                 "data/experiments/flexible-generalization.json")))["categories"]
funcs = next(c for c in cats if c["name"] == "countries")["funcs"]
ALIASES = {"yuan": {"yuan", "renminbi", "rmb", "cny"}, "euro": {"euro", "euros", "eur"},
           "beijing": {"beijing", "peking"}, "paris": {"paris"},
           "french": {"french"}, "chinese": {"chinese", "mandarin"},
           "europe": {"europe"}, "asia": {"asia"}}
acc = lambda f: ALIASES.get(f.strip().lower(), {f.strip().lower()})
names = lambda t, f: bool(set(re.findall(r"[a-z]+", t.lower())) & acc(f))


def csl(a, b):
    n = 0
    while n < len(a) and n < len(b) and a[-1 - n] == b[-1 - n]:
        n += 1
    return max(n, 1)


@torch.no_grad()
def cap(t, layers):
    ids = model.encode(t, max_length=256)
    with ActivationRecorder(model.layers, at=layers) as rec:
        model.forward(ids)
    return {l: rec.activations[l][0].float().clone() for l in layers}


def mk(src, k):
    def fn(h):
        if h.shape[1] == 1:
            return h
        out = h.clone(); kk = min(k, out.shape[1], src.shape[0])
        out[:, -kk:, :] = src[-kk:].to(h.dtype)
        return out
    return fn


@torch.no_grad()
def gen(t, hooks=()):
    ids = model.encode(t, max_length=256)
    with contextlib.ExitStack() as st:
        for l, fn in hooks:
            st.enter_context(intervene(model, l, fn))
        o = hf.generate(ids, max_new_tokens=8, do_sample=False,
                        pad_token_id=tok.eos_token_id)
    return tok.decode(o[0][ids.shape[1]:], skip_special_tokens=True)


print(f"{CONFIGS[args.model]['name']}  {args.src} -> {args.dst}"
      f"{'  [DEFERRED readout]' if args.deferred else '  [readout AT boundary]'}\n")
rows = []; sanity = []
for ename, layers in EXTENTS.items():
    hits_tok = hits_off = base_ok = 0
    for f in funcs:
        if args.deferred:
            # Same framing the skills use: the differing token is followed by ~20
            # tokens of shared boilerplate, and the answer must be emitted
            # several UNPATCHED generated positions later.
            FMT = ("\n\nRespond with a single line beginning `ANSWER:` followed by "
                   "only the answer, with no explanation.")
            wrap = lambda t: tok.apply_chat_template(
                [{"role": "user", "content": t + FMT}], tokenize=False,
                add_generation_prompt=True)
            ps, pd = wrap(f["template"].format(arg=args.src)), wrap(f["template"].format(arg=args.dst))
        else:
            ps, pd = f["template"].format(arg=args.src), f["template"].format(arg=args.dst)
        a_s, a_d = f["answers"][args.src], f["answers"][args.dst]
        base = gen(ps)
        base_ok += names(base, a_s)
        isx = tok(ps, add_special_tokens=False).input_ids
        idx = tok(pd, add_special_tokens=False).input_ids
        k_tok, k_off = csl(isx, idx), min(len(isx), len(idx))
        cd = cap(pd, layers)
        g_tok = gen(ps, [(l, mk(cd[l], k_tok)) for l in layers])
        # SANITY: under an all-layer patch, generated token 1 must equal the
        # dst-run's greedy token 1. If it does not, the hook is not landing and
        # every 0% in this project is mechanical.
        if ename == "all":
            t1_src = gen(ps, [(l, mk(cd[l], k_tok)) for l in layers])[:12]
            t1_dst = gen(pd)[:12]
            sanity.append((t1_src, t1_dst))
        g_off = gen(ps, [(l, mk(cd[l], k_off)) for l in layers])
        ok_t, ok_o = names(g_tok, a_d), names(g_off, a_d)
        hits_tok += ok_t; hits_off += ok_o
        if ename == "band":
            print(f"  {f['name']:10s} base={base.strip()[:20]!r:22s} "
                  f"tok={g_tok.strip()[:20]!r:22s} {'HIT' if ok_t else '   '}  "
                  f"off={g_off.strip()[:18]!r:20s} {'HIT' if ok_o else ''}"
                  f"   shared={k_tok}/{len(isx)} tokens")
    n = len(funcs)
    rows.append({"extent": ename, "token_aligned": hits_tok / n,
                 "offset_aligned": hits_off / n, "baseline_correct": base_ok / n})
    print(f"  [{ename}] token-aligned {hits_tok}/{n}   offset-aligned {hits_off}/{n}"
          f"   (baseline recovers source {base_ok}/{n})\n")
if sanity:
    match = sum(1 for a, b in sanity if a.strip()[:6] == b.strip()[:6])
    print(f"  SANITY (all-layer): patched token-1 matches dst token-1 in "
          f"{match}/{len(sanity)} cases")
    for a, b in sanity[:2]:
        print(f"    patched={a.strip()!r:16s} dst={b.strip()!r}")
    if match == 0:
        print("    !! hook is NOT landing -- every 0% in this project is mechanical")
tok_best = max(r["token_aligned"] for r in rows)
print("VERDICT:", end=" ")
if tok_best >= 0.75:
    print("PIPELINE WORKS. Token-aligned patching transfers an ITEM edit.")
    print("  -> the skill's 0% isolates item-vs-procedure; the negative stands.")
elif tok_best == 0:
    print("PIPELINE ALSO FAILS on the known-good item edit.")
    print("  -> the skill result is NOT interpretable. Stop everything downstream")
    print("     until this is understood.")
else:
    print(f"PARTIAL ({tok_best:.0%}). Neither clean; treat the skill result as")
    print("  provisional and report this number with it.")
json.dump({"model": CONFIGS[args.model]["name"], "pair": [args.src, args.dst],
           "rows": rows}, open(os.path.join(OUT, f"control_fc_{args.model}.json"), "w"),
          indent=1)
