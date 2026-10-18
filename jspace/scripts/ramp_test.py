"""Hysteresis ramp test — the qualitative discriminator from ignition-criterion.md.

H0 (thresholding): report onset is a memoryless function of the current alpha.
An up-ramp and a down-ramp must cross at the SAME alpha. Predicted loop: exactly 0.

H1 (ignition): entry is self-amplifying. Once the concept is in the context it
sustains itself, so a down-ramp keeps reporting BELOW the alpha at which an
up-ramp started. Predicted loop: > 0.

Why this and not an up/down sweep of independent forward passes: a stateless
forward pass has no history, so up and down are identical BY CONSTRUCTION and
such a test returns zero regardless of the truth. The only genuine feedback path
here is autoregressive generation, where emitted tokens re-enter the context. So
alpha is ramped across TOKEN POSITIONS within a single generation.

Coherence gate is recorded per ramp; a degenerate ramp is excluded, because a
model stuck emitting one token trivially "sustains report".
"""
from __future__ import annotations
import argparse, json, os, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from common import CONFIGS, OUT, add_model_arg, depth_plan, lens_path, load_model
from lens_ops import coordinate_rms, intervene, random_direction
from restricted import RestrictedLens
import tokens as toks

ap = argparse.ArgumentParser(); add_model_arg(ap)
ap.add_argument("--tag", default="")
ap.add_argument("--concept", default="spider")
ap.add_argument("--n-prompts", type=int, default=200)
ap.add_argument("--alpha-max", type=float, default=8.0)   # coherent ceiling
ap.add_argument("--gen", type=int, default=48)
args = ap.parse_args()

CLUSTER_WORDS = {"spider", "spiders", "web", "webs", "cobweb", "silk", "arachnid",
                 "legs", "eight"}
NL = CONFIGS[args.model]["n_layers"]; plan = depth_plan(NL); INJ = plan["inject_layer"]
model, hf, tok, device, dtype = load_model(args.model)
lens = RestrictedLens.load(lens_path(args.model, args.tag))
tracked, _ = toks.build(tok, OUT)
d_con = lens.direction(INJ, tracked[args.concept]["id"])
d_rnd = random_direction(lens.d_model, 20260819)

CL = set("spider web insect legs eight piano music keys sound notes arachnid "
         "spiders webs cobweb silk".split())
fit_used = json.load(open(os.path.join(OUT, "fit_prompts.json")))
from jlens.examples import load_wikitext_prompts
pool = load_wikitext_prompts(len(fit_used) + 8 * args.n_prompts)
assert pool[:len(fit_used)] == fit_used, "prompt order not reproducible"
held = [q for q in pool[len(fit_used):]
        if not (set(re.findall(r"[a-z]+", q.lower())) & CL) and len(q) >= 40]
prompts = held[: args.n_prompts]
SCALE = coordinate_rms(model, held[:40], INJ, d_con)
print(f"{CONFIGS[args.model]['name']} inject=L{INJ} alpha unit={SCALE:.3f} "
      f"n={len(prompts)} ramp 0..{args.alpha_max}", flush=True)


class RampPatch:
    """Coordinate clamp whose target amount depends on the generation step."""
    def __init__(self, direction, sched, scale):
        self.u = direction / direction.norm(); self.sched = sched
        self.scale = scale; self.t = 0
    def reset(self): self.t = 0
    def __call__(self, h):
        a = self.sched[min(self.t, len(self.sched) - 1)]
        self.t += 1
        u = self.u.to(h.device, h.dtype)
        out = h.clone()
        coeff = (out.float() @ u.float()).unsqueeze(-1)
        out = (out.float() + (a * self.scale - coeff) * u.float()).to(h.dtype)
        return out


@torch.no_grad()
def primed(prompt):
    """CRITICAL CONTROL: no injection at all (alpha=0 throughout), but the first
    generated token is FORCED to the concept token.

    A down-ramp starts at high alpha, so the model emits the concept in its first
    few tokens -- and those tokens then sit in the context. An ordinary language
    model continues a topic once it is raised. So the apparent "loop" may be
    nothing but text autocorrelation, with no workspace bistability anywhere.
    The random-direction control cannot test this: random never induces the
    concept, so it produces no topic to sustain and no loop to measure.

    This arm isolates it. If persistence here matches the down-ramp's, the loop
    is explained WITHOUT any injection and is not evidence of self-amplification.
    """
    T = args.gen
    sched = [args.alpha_max * i / (T - 1) for i in range(T)][::-1]
    ids = model.encode(prompt, max_length=64)
    forced = torch.tensor([[tracked[args.concept]["id"]]], device=ids.device)
    ids2 = torch.cat([ids, forced], 1)
    out = hf.generate(ids2, max_new_tokens=T - 1, do_sample=False,
                      pad_token_id=tok.eos_token_id)
    gen = out[0][ids.shape[1]:]
    tt = [tok.decode([t]).strip().lower() for t in gen]
    hits = [i for i, w in enumerate(tt) if re.sub(r"[^a-z]", "", w) in CLUSTER_WORDS]
    words = tok.decode(gen, skip_special_tokens=True).split()
    rep = 1 - len(set(words)) / max(len(words), 1)
    a_at = lambda i: sched[min(i, len(sched) - 1)]
    return {"last_alpha": a_at(hits[-1]) if hits else None,
            "n_hits": len(hits), "repeat_frac": rep}


@torch.no_grad()
def triangle(prompt, direction):
    """Symmetric triangular ramp: alpha rises 0->max over the first half of the
    generation and falls max->0 over the second, WITHIN ONE generation.

    Separate up- and down-ramps are not comparable. On a down-ramp alpha=max
    lands on the first generated token with the whole sequence left to show
    report; on an up-ramp it lands on the LAST token with nowhere to show it.
    That alone produced 158/200 "never reported going up" against 198/200 going
    down -- an asymmetry of ramp geometry, not of threshold. A triangle gives
    each direction an equal token budget at every alpha, in the same sequence,
    with the same context history up to the apex.

    onset  = alpha at the first cluster hit on the RISING half
    offset = alpha at the last cluster hit on the FALLING half
    loop   = onset - offset  (H0: 0; ignition: > 0)
    """
    T = args.gen
    half = T // 2
    sched = ([args.alpha_max * i / (half - 1) for i in range(half)] +
             [args.alpha_max * (1 - i / (T - half - 1)) for i in range(T - half)])
    patch = RampPatch(direction, sched, SCALE); patch.reset()
    ids = model.encode(prompt, max_length=64)
    with intervene(model, INJ, patch):
        out = hf.generate(ids, max_new_tokens=T, do_sample=False,
                          pad_token_id=tok.eos_token_id)
    gen = out[0][ids.shape[1]:]
    tt = [tok.decode([t]).strip().lower() for t in gen]
    hits = [i for i, w in enumerate(tt) if re.sub(r"[^a-z]", "", w) in CLUSTER_WORDS]
    rise = [i for i in hits if i < half]
    fall = [i for i in hits if i >= half]
    words = tok.decode(gen, skip_special_tokens=True).split()
    rep = 1 - len(set(words)) / max(len(words), 1)
    return {"onset": sched[rise[0]] if rise else None,
            "offset": sched[fall[-1]] if fall else None,
            "n_rise": len(rise), "n_fall": len(fall), "repeat_frac": rep}


@torch.no_grad()
def ramp(prompt, direction, up: bool):
    T = args.gen
    sched = [args.alpha_max * i / (T - 1) for i in range(T)]
    if not up:
        sched = sched[::-1]
    patch = RampPatch(direction, sched, SCALE); patch.reset()
    ids = model.encode(prompt, max_length=64)
    with intervene(model, INJ, patch):
        out = hf.generate(ids, max_new_tokens=T, do_sample=False,
                          pad_token_id=tok.eos_token_id)
    gen = out[0][ids.shape[1]:]
    toks_txt = [tok.decode([t]).strip().lower() for t in gen]
    hits = [i for i, w in enumerate(toks_txt)
            if re.sub(r"[^a-z]", "", w) in CLUSTER_WORDS]
    # position -> alpha under this ramp's schedule
    a_at = lambda i: sched[min(i, len(sched) - 1)]
    words = tok.decode(gen, skip_special_tokens=True).split()
    rep = 1 - len(set(words)) / max(len(words), 1)
    return {"hits": hits, "first_alpha": a_at(hits[0]) if hits else None,
            "last_alpha": a_at(hits[-1]) if hits else None,
            "n_hits": len(hits), "repeat_frac": rep}


rows, t0 = [], time.time()
for i, p in enumerate(prompts):
    r = {"prompt": i}
    for nm, dd in (("concept", d_con), ("random", d_rnd)):
        u = ramp(p, dd, True); dn = ramp(p, dd, False)
        r[f"{nm}_up_on"] = u["first_alpha"]      # alpha where report STARTS going up
        r[f"{nm}_down_off"] = dn["last_alpha"]   # alpha where report STOPS going down
        r[f"{nm}_up_rep"] = u["repeat_frac"]; r[f"{nm}_down_rep"] = dn["repeat_frac"]
        r[f"{nm}_up_n"] = u["n_hits"]; r[f"{nm}_down_n"] = dn["n_hits"]
    for nm, dd in (("concept", d_con), ("random", d_rnd)):
        tr = triangle(p, dd)
        r[f"tri_{nm}_onset"] = tr["onset"]; r[f"tri_{nm}_offset"] = tr["offset"]
        r[f"tri_{nm}_rep"] = tr["repeat_frac"]
    pr = primed(p)
    r["primed_off"] = pr["last_alpha"]; r["primed_rep"] = pr["repeat_frac"]
    r["primed_n"] = pr["n_hits"]
    rows.append(r)
    if (i + 1) % 25 == 0:
        print(f"  {i+1}/{len(prompts)}", flush=True)

import statistics as st
def summarise(nm):
    ok = [r for r in rows if r[f"{nm}_up_on"] is not None
          and r[f"{nm}_down_off"] is not None
          and r[f"{nm}_up_rep"] < 0.5 and r[f"{nm}_down_rep"] < 0.5]
    if len(ok) < 10:
        print(f"{nm}: only {len(ok)} usable ramps — insufficient"); return None
    loops = [r[f"{nm}_up_on"] - r[f"{nm}_down_off"] for r in ok]
    m = st.mean(loops); sd = st.pstdev(loops); sem = sd / len(ok) ** 0.5
    print(f"\n{nm}: n={len(ok)} usable (coherence-gated)")
    print(f"  up-ramp onset   mean {st.mean([r[f'{nm}_up_on'] for r in ok]):.3f}")
    print(f"  down-ramp offset mean {st.mean([r[f'{nm}_down_off'] for r in ok]):.3f}")
    print(f"  LOOP = up_on - down_off: mean {m:+.3f} +- {sem:.3f} (SEM), "
          f"{m/sem if sem else 0:+.2f} sigma")
    return {"n": len(ok), "loop_mean": m, "loop_sem": sem,
            "sigma": m / sem if sem else 0.0}

def tri_summary(nm):
    ok = [r for r in rows if r.get(f"tri_{nm}_onset") is not None
          and r.get(f"tri_{nm}_offset") is not None and r[f"tri_{nm}_rep"] < 0.5]
    if len(ok) < 10:
        print(f"\nTRIANGLE {nm}: only {len(ok)} usable — insufficient"); return None
    loops = [r[f"tri_{nm}_onset"] - r[f"tri_{nm}_offset"] for r in ok]
    m = st.mean(loops); sem = st.pstdev(loops) / len(ok) ** 0.5
    print(f"\nTRIANGLE {nm} (symmetric, same generation): n={len(ok)}")
    print(f"  onset (rising)  mean {st.mean([r[f'tri_{nm}_onset'] for r in ok]):.3f}")
    print(f"  offset (falling) mean {st.mean([r[f'tri_{nm}_offset'] for r in ok]):.3f}")
    print(f"  LOOP mean {m:+.3f} +- {sem:.3f} SEM ({m/sem if sem else 0:+.2f} sigma)")
    return {"n": len(ok), "loop_mean": m, "loop_sem": sem,
            "sigma": m / sem if sem else 0.0}

res = {nm: summarise(nm) for nm in ("concept", "random")}
res["triangle_concept"] = tri_summary("concept")
res["triangle_random"] = tri_summary("random")
pk = [r for r in rows if r.get("primed_off") is not None and r["primed_rep"] < 0.5]
if pk:
    po = st.mean([r["primed_off"] for r in pk])
    print(f"\nPRIMED CONTROL (no injection, concept forced as first token): "
          f"n={len(pk)}, persistence offset mean {po:.3f}")
    res["primed"] = {"n": len(pk), "offset_mean": po}
    if res.get("concept"):
        dm = st.mean([r["concept_down_off"] for r in rows
                      if r["concept_down_off"] is not None and r["concept_down_rep"] < 0.5])
        print(f"  down-ramp offset {dm:.3f} vs primed offset {po:.3f} "
              f"-> difference {dm - po:+.3f}")
        print("  If these match, the loop is TEXT AUTOCORRELATION, not ignition.")
        res["primed"]["down_minus_primed"] = dm - po
c = res["concept"]
if c:
    thr = 0.30
    if abs(c["loop_mean"]) < thr and abs(c["sigma"]) < 2:
        v = f"NO HYSTERESIS (|loop| {abs(c['loop_mean']):.3f} < {thr} threshold) -> self-amplification of the required size EXCLUDED"
    elif c["sigma"] >= 2 and c["loop_mean"] > 0:
        v = f"HYSTERESIS DETECTED (loop {c['loop_mean']:+.3f}, {c['sigma']:.1f} sigma) -> positive evidence for ignition; check random control"
    else:
        v = "INCONCLUSIVE"
    print(f"\nVERDICT: {v}")
    res["verdict"] = v
json.dump({"rows": rows, "summary": res, "alpha_unit": SCALE,
           "alpha_max": args.alpha_max, "wall": time.time() - t0},
          open(os.path.join(OUT, f"ramp_{args.model}.json"), "w"), indent=1)
print(f"-> ramp_{args.model}.json ({(time.time()-t0)/60:.1f} min)")
