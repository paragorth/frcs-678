"""Is the model still coherent across Job 2's alpha range?

"I was fitting a line through wreckage" was the failure two runs ago. The piano
contrast probe does NOT cover this: wreckage produced by a spider injection
would be spider-flavoured wreckage, so it would score 0% on piano while still
being degenerate. Specificity and coherence are different questions.

Quantitative degradation markers alongside the text, so this is not an eyeball
judgement: next-token entropy, KL from the uninjected baseline, and the fraction
of generated tokens that are non-ASCII or repeated.
"""
from __future__ import annotations
import argparse, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn.functional as F
from common import CONFIGS, OUT, add_model_arg, depth_plan, lens_path, load_model
from lens_ops import CoordinatePatch, coordinate_rms, intervene
from restricted import RestrictedLens
import tokens as toks

ap = argparse.ArgumentParser(); add_model_arg(ap)
ap.add_argument("--tag", default="")
ap.add_argument("--concept", default="spider")
ap.add_argument("--alphas", default="0,4,6,8,10,13,16,20,30")
ap.add_argument("--n-prompts", type=int, default=8)
ap.add_argument("--gen", type=int, default=40)
args = ap.parse_args()

NL = CONFIGS[args.model]["n_layers"]; plan = depth_plan(NL); INJ = plan["inject_layer"]
model, hf, tok, device, dtype = load_model(args.model)
lens = RestrictedLens.load(lens_path(args.model, args.tag))
tracked, _ = toks.build(tok, OUT)
d = lens.direction(INJ, tracked[args.concept]["id"])
# CRITICAL: the alpha unit must be computed from the SAME prompt pool as the
# run being checked, or the alpha axes are not comparable. stage5_report.py
# derives it from the first 40 held-out wikitext prompts; using the 20 neutral
# sweep prompts instead gave 45.47 vs 9.82 -- a factor of 4.6, which would make
# this check test a completely different injection strength than the run it is
# supposed to validate.
CLUSTER = set("spider web insect legs eight piano music keys sound notes "
              "arachnid spiders webs cobweb silk".split())
fit_used = json.load(open(os.path.join(OUT, "fit_prompts.json")))
from jlens.examples import load_wikitext_prompts
pool = load_wikitext_prompts(len(fit_used) + 6 * 120)
assert pool[:len(fit_used)] == fit_used, "prompt order not reproducible"
held = [q for q in pool[len(fit_used):]
        if not (set(re.findall(r"[a-z]+", q.lower())) & CLUSTER) and len(q) >= 40]
SCALE = coordinate_rms(model, held[:40], INJ, d)
prompts = held[: args.n_prompts]
print(f"alpha unit from the SAME pool stage5 used: {SCALE:.4f}")
ALPHAS = [float(x) for x in args.alphas.split(",")]
print(f"{CONFIGS[args.model]['name']}  inject=L{INJ}  alpha unit={SCALE:.3f}\n")


@torch.no_grad()
def run(p, a):
    ids = model.encode(p, max_length=64)
    patch = CoordinatePatch(d, a * SCALE) if a else None
    if patch:
        with intervene(model, INJ, patch):
            out = hf.generate(ids, max_new_tokens=args.gen, do_sample=False,
                              pad_token_id=tok.eos_token_id)
            lg = hf(ids).logits[0, -1].float()
    else:
        out = hf.generate(ids, max_new_tokens=args.gen, do_sample=False,
                          pad_token_id=tok.eos_token_id)
        lg = hf(ids).logits[0, -1].float()
    txt = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    return txt, F.log_softmax(lg, -1)


rows = []
base_lp = {p: run(p, 0.0)[1] for p in prompts}
for a in ALPHAS:
    ents, kls, nonascii, reps, texts = [], [], [], [], []
    for p in prompts:
        txt, lp = run(p, a)
        ents.append(float(-(lp.exp() * lp).sum()))
        b = base_lp[p]
        kls.append(float((b.exp() * (b - lp)).sum()))
        w = txt.split()
        nonascii.append(sum(1 for ch in txt if ord(ch) > 127) / max(len(txt), 1))
        reps.append(1 - len(set(w)) / max(len(w), 1))
        texts.append(txt)
    row = {"alpha": a, "entropy": sum(ents)/len(ents), "kl": sum(kls)/len(kls),
           "nonascii_frac": sum(nonascii)/len(nonascii),
           "repeat_frac": sum(reps)/len(reps), "sample": texts[0][:110]}
    rows.append(row)
    print(f"a={a:5.1f}  H={row['entropy']:5.2f}  KL={row['kl']:6.3f}  "
          f"nonASCII={100*row['nonascii_frac']:4.1f}%  rep={100*row['repeat_frac']:4.1f}%")
    print(f"         {texts[0][:110]!r}", flush=True)

b = rows[0]
print(f"\nbaseline H={b['entropy']:.2f}")
for r in rows:
    flag = ("DEGRADED" if (r["entropy"] > b["entropy"] + 1.5 or r["kl"] > 3.0
            or r["nonascii_frac"] > 0.05 or r["repeat_frac"] > 0.5) else "ok")
    r["verdict"] = flag
ok = [r["alpha"] for r in rows if r["verdict"] == "ok"]
print(f"coherent alphas: {ok}")
print(f"COHERENT UP TO alpha = {max(ok) if ok else 'none'}")
json.dump(rows, open(os.path.join(OUT, f"coherence_{args.model}.json"), "w"), indent=1)
