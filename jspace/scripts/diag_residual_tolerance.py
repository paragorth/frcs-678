"""How close does the match have to be, in behavioural units?

The matching arms drove L_target from 975.23 to ~1.0. That is 99.9% of the
SQUARED error but only 96.7% of the distance -- residual RMS ~1.0 per coordinate
against an initial 31.2. Whether that residual matters cannot be argued, only
measured.

So: take the EXACT captured target, corrupt it with isotropic noise of known RMS
sigma, patch it in, and measure install. sigma=0 is the ceiling; sigma=31.2 is
no-patch. The trained arms sit at sigma~1.0.

  install(sigma=1.0) ~= install(0)  -> the arms should have installed and did
                                       not; the negative is real (though see the
                                       power note below)
  install(sigma=1.0) ~= 0           -> the residual IS the story and the
                                       "matching does not install" headline is
                                       false

Power note, from the cold read: the band ceiling is 0.208 and the arms scored
0/12. P(0 successes | p=0.208, n=12) = 0.061. The observed null is not
significant at p<0.05 even against a PERFECT match. Install must therefore be
reported as a fraction of the achievable ceiling, never against 100%.
"""
from __future__ import annotations
import argparse, json, os, re, sys, contextlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from common import CONFIGS, OUT, add_model_arg, depth_plan, load_model
from lens_ops import intervene
from jlens.hooks import ActivationRecorder

ap = argparse.ArgumentParser(); add_model_arg(ap)
ap.add_argument("--skill", default="beacon"); ap.add_argument("--n", type=int, default=24)
ap.add_argument("--sigmas", default="0,0.25,0.5,1.0,2.0,4.0,8.0,31.2")
ap.add_argument("--extent", default="band", choices=["band", "all"])
ap.add_argument("--renorm", action="store_true",
                help="rescale the noised target back to the original norm, so only DIRECTION varies")
args = ap.parse_args()
NL = CONFIGS[args.model]["n_layers"]; plan = depth_plan(NL)
LAYERS = plan["band"] if args.extent == "band" else list(range(NL))
model, hf, tok, device, dtype = load_model(args.model)
SK = os.path.join(OUT, "skills")
spec = open(os.path.join(SK, args.skill, "skill.md")).read()
inst = json.load(open(os.path.join(SK, args.skill, "instances.json")))[: args.n]
FORMAT = ("\n\nRespond with a single line beginning `ANSWER:` followed by only "
          "the answer, with no explanation.")
chat = lambda t: tok.apply_chat_template([{"role": "user", "content": t}],
                                         tokenize=False, add_generation_prompt=True)
SIG = [float(x) for x in args.sigmas.split(",")]
RENORM = args.renorm
print(f"{CONFIGS[args.model]['name']} extent={args.extent} layers={LAYERS[0]}-{LAYERS[-1]} "
      f"sigmas={SIG}", flush=True)


@torch.no_grad()
def capture(text):
    ids = model.encode(text, max_length=512)
    with ActivationRecorder(model.layers, at=LAYERS) as rec:
        model.forward(ids)
    return {l: rec.activations[l][0].float().clone() for l in LAYERS}


def hook(cap, l, sigma, gen):
    def fn(h):
        if h.shape[1] == 1:
            return h
        out = h.clone(); src = cap[l]
        k = min(out.shape[1], src.shape[0])
        t = src[-k:].clone()
        if sigma > 0:
            t = t + torch.randn(t.shape, generator=gen, device="cpu").to(t.device) * sigma
            if RENORM:
                # Hold per-position norm at the ORIGINAL target's norm so ONLY
                # direction changes. Without this the sweep varies magnitude:
                # install peaks at norm ratio ~1.0 and cosine stays >= 0.995
                # throughout, so the curve calibrates norm tolerance and says
                # nothing about direction tolerance.
                src_t = src[-k:]
                t = t * (src_t.norm(dim=-1, keepdim=True) /
                         (t.norm(dim=-1, keepdim=True) + 1e-9))
        out[:, -k:, :] = t.to(h.dtype)
        return out
    return fn


def norm(s): return re.sub(r"\s+", " ", s.strip()).upper()


@torch.no_grad()
def ask(text, cap=None, sigma=0.0, gen=None):
    ids = model.encode(text, max_length=512)
    with contextlib.ExitStack() as st:
        if cap is not None:
            for l in LAYERS:
                st.enter_context(intervene(model, l, hook(cap, l, sigma, gen)))
        o = hf.generate(ids, max_new_tokens=32, do_sample=False,
                        pad_token_id=tok.eos_token_id)
    g = tok.decode(o[0][ids.shape[1]:], skip_special_tokens=True)
    m = re.findall(r"ANSWER:\s*(.+)", g)
    return re.sub(r"[`*_.]+$", "", (m[-1] if m else g.strip().split("\n")[-1]
                                    if g.strip() else "")).strip()


res = {s: 0 for s in SIG}; cosv = {s: [] for s in SIG}; nop = 0; wf = 0
for it in inst:
    pw, pn = chat(spec + "\n\n---\n\nTask: " + it["prompt"] + FORMAT), chat(it["prompt"] + FORMAT)
    cap = capture(pw)
    nop += norm(ask(pn)) == norm(it["answer"])
    wf += norm(ask(pw)) == norm(it["answer"])
    for s in SIG:
        g = torch.Generator(device="cpu").manual_seed(1234)
        res[s] += norm(ask(pn, cap, s, g)) == norm(it["answer"])
        g2 = torch.Generator(device="cpu").manual_seed(1234)
        for l in LAYERS:
            t = cap[l]
            noisy = t + torch.randn(t.shape, generator=g2, device="cpu").to(t.device) * s
            cosv[s].append(float(torch.nn.functional.cosine_similarity(
                noisy.flatten(), t.flatten(), dim=0)))
n = len(inst)
print(f"\nRESIDUAL TOLERANCE, {args.skill}, n={n}")
print(f"  no_patch {nop/n:6.1%}   with_file {wf/n:6.1%}")
print(f"  {'sigma':>7} {'cosine':>8} {'install':>9}")
for s in SIG:
    c = sum(cosv[s]) / len(cosv[s])
    print(f"  {s:7.2f} {c:8.3f} {res[s]/n:9.1%}")
ARMS_COS = 0.928
best = min(SIG, key=lambda x: abs(sum(cosv[x])/len(cosv[x]) - ARMS_COS))
bc = sum(cosv[best]) / len(cosv[best])
print(f"\n  trained arms reached cosine {ARMS_COS:.3f}")
print(f"  nearest patched sigma: {best} (cosine {bc:.3f}) -> install {res[best]/n:.1%}")
if res[best] / n > 0.15:
    print("  -> patching at the arms' OWN cosine DOES install. The arms cleared")
    print("     the bar on match quality and still installed 0%. Negative SUPPORTED.")
else:
    print("  -> patching at the arms' own cosine does NOT install either. The arms")
    print("     never cleared the bar; the negative is NOT supported.")
ceil = res[SIG[0]] / n
print(f"\nceiling (sigma=0) = {ceil:.1%}")
at1 = res[min(SIG, key=lambda x: abs(x - 1.0))] / n
print(f"install at sigma~1.0 (where the trained arms sat) = {at1:.1%}")
if ceil > 0:
    print(f"  -> that is {100*at1/ceil:.0f}% of the achievable ceiling")
    if at1 < 0.3 * ceil:
        print("  -> THE RESIDUAL IS THE STORY. Matching to residual ~1.0 cannot")
        print("     install even when patched in exactly. The 'matching does not")
        print("     install' headline is NOT supported.")
    else:
        print("  -> residual ~1.0 still installs; the arms' 0% is a real negative")
        print("     (though underpowered: P(0/12 | p=ceiling) must be reported).")
import math
p0 = (1 - ceil) ** n if ceil > 0 else 1.0
print(f"  P(0 successes in {n} | p=ceiling={ceil:.3f}) = {p0:.4f}")
json.dump({"model": CONFIGS[args.model]["name"], "skill": args.skill, "n": n,
           "extent": args.extent, "no_patch": nop / n, "with_file": wf / n,
           "curve": {str(s): res[s] / n for s in SIG},
           "cosine_at_sigma": {str(s): sum(cosv[s])/len(cosv[s]) for s in SIG}, "ceiling": ceil,
           "install_at_sigma1": at1, "p_zero_given_ceiling": p0},
          open(os.path.join(OUT, f"diag_residual_{args.model}.json"), "w"), indent=1)
