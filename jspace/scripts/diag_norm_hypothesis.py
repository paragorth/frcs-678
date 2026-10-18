"""Does norm, not direction, explain the non-monotone install curve?

Install: 20.8% exact -> 70.8% at sigma 0.25-0.5 -> 0% at sigma 2.0, with cosine
1.000 across the whole rise. Cosine discards magnitude, so it cannot see what is
driving that. Isotropic noise INCREASES norm: E||t+eps||^2 = ||t||^2 + sigma^2*d.

Hypothesis: the captured with-file state sits at a different norm from the
model's natural state at those positions. Noise moves the patched norm toward the
natural one, install rises; past it, install collapses. If so, install is a
monotone function of ||patched||/||natural|| and the non-monotonicity in sigma is
an artefact of using a direction-only yardstick.

This matters beyond the anomaly: the ">= 0.999 cosine" criterion for the next
experiment was derived from this curve. If norm is the operative variable, that
criterion is measuring the wrong thing.
"""
from __future__ import annotations
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from common import CONFIGS, OUT, add_model_arg, depth_plan, load_model
from jlens.hooks import ActivationRecorder

ap = argparse.ArgumentParser(); add_model_arg(ap)
ap.add_argument("--skill", default="beacon"); ap.add_argument("--n", type=int, default=12)
ap.add_argument("--sigmas", default="0,0.05,0.1,0.15,0.25,0.5,1.0,2.0")
args = ap.parse_args()
NL = CONFIGS[args.model]["n_layers"]; LAYERS = depth_plan(NL)["band"]
model, hf, tok, device, dtype = load_model(args.model)
SK = os.path.join(OUT, "skills")
spec = open(os.path.join(SK, args.skill, "skill.md")).read()
inst = json.load(open(os.path.join(SK, args.skill, "instances.json")))[: args.n]
FORMAT = ("\n\nRespond with a single line beginning `ANSWER:` followed by only "
          "the answer, with no explanation.")
chat = lambda t: tok.apply_chat_template([{"role": "user", "content": t}],
                                         tokenize=False, add_generation_prompt=True)
SIG = [float(x) for x in args.sigmas.split(",")]


@torch.no_grad()
def cap(text):
    ids = model.encode(text, max_length=512)
    with ActivationRecorder(model.layers, at=LAYERS) as rec:
        model.forward(ids)
    return {l: rec.activations[l][0].float().clone() for l in LAYERS}


def common_suffix_len(a, b):
    n = 0
    while n < len(a) and n < len(b) and a[-1 - n] == b[-1 - n]:
        n += 1
    return max(n, 1)


tgt_n, nat_n = [], []
rows = []
ks = []
for it in inst:
    pw, pn = chat(spec + "\n\n---\n\nTask: " + it["prompt"] + FORMAT), chat(it["prompt"] + FORMAT)
    cw, cn = cap(pw), cap(pn)
    # Align on the common TOKEN suffix, not on min(len). Offset alignment puts
    # the short prompt's attention-sink tokens (huge norm in Qwen) against the
    # long prompt's content -- the same bug that made L_target 2000x too large.
    k = common_suffix_len(tok(pn, add_special_tokens=False).input_ids,
                          tok(pw, add_special_tokens=False).input_ids)
    ks.append(k)
    for l in LAYERS:
        k_ = min(k, cw[l].shape[0], cn[l].shape[0])
        t, nat = cw[l][-k_:], cn[l][-k_:]
        tgt_n.append(float(t.norm(dim=-1).mean()))
        nat_n.append(float(nat.norm(dim=-1).mean()))
import statistics as st
T, N = st.mean(tgt_n), st.mean(nat_n)
d = cap(chat("x"))[LAYERS[0]].shape[-1]
print(f"{CONFIGS[args.model]['name']} band L{LAYERS[0]}-{LAYERS[-1]}, d={d}, "
      f"n={args.n}, aligned positions {st.mean(ks):.1f} (common token suffix)")
print(f"\n  mean per-position norm of the CAPTURED with-file target : {T:8.3f}")
print(f"  mean per-position norm of the model's NATURAL no-file state: {N:8.3f}")
print(f"  target/natural = {T/N:.4f}\n")
INSTALL = {0.0: .208, 0.05: .333, 0.1: .458, 0.15: .500, 0.25: .708,
           0.5: .708, 1.0: .458, 2.0: .000}
print(f"  {'sigma':>6} {'E||patched||':>13} {'/natural':>9} {'install':>8}")
for s in SIG:
    pn_ = (T ** 2 + s * s * d) ** 0.5
    rows.append({"sigma": s, "patched_norm": pn_, "ratio": pn_ / N,
                 "install": INSTALL.get(s)})
    inс = INSTALL.get(s)
    print(f"  {s:6.2f} {pn_:13.3f} {pn_/N:9.4f} "
          f"{'' if inс is None else f'{inс:7.1%}'}")
best = max((r for r in rows if r["install"] is not None), key=lambda r: r["install"])
print(f"\n  install peaks at sigma={best['sigma']} where ||patched||/||natural|| = "
      f"{best['ratio']:.4f}")
if abs(best["ratio"] - 1.0) < 0.25:
    print("  -> PEAK SITS AT THE NATURAL NORM. Install is a monotone function of")
    print("     norm ratio; the non-monotonicity in sigma is an artefact of using")
    print("     a direction-only measure. Cosine is the WRONG yardstick and the")
    print("     '>=0.999 cosine' criterion must be replaced by a norm-aware one.")
else:
    print("  -> peak is NOT at the natural norm; the norm hypothesis does not")
    print("     explain the curve and the driver is still unidentified.")
json.dump({"model": CONFIGS[args.model]["name"], "target_norm": T,
           "natural_norm": N, "ratio": T / N, "d": d, "rows": rows},
          open(os.path.join(OUT, f"diag_norm_{args.model}.json"), "w"), indent=1)
