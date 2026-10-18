"""Stage 4: is the behavioural response bistable, or just compressive?

The projection measure is near-tautological (it returns what was injected), and
argmax over the vocabulary is itself a threshold, so a visible change in the
generated text proves nothing about the underlying process. The quantity that
would actually distinguish a competitive bottleneck from smooth saturation is
the *margin*:

    margin = logit(concept) - max logit over all other tokens

Rank is not usable here: it is already discretised, so it manufactures the very
structure being tested for.

Trial variance has to come from *prompts*, not sampling. Greedy decoding on a
fixed prompt is deterministic, so N seeds at fixed alpha give N identical
numbers and nothing to test. Each prompt is one trial.

THE PRIMARY ANALYSIS IS PER-PROMPT, NOT POOLED. If entry is threshold-like,
each prompt has its own threshold, jittered by content and by whatever else is
already in the workspace. A population of stepped units with jittered
thresholds pools into a smooth unimodal distribution -- so a pooled dip test
returning "unimodal" would be uninformative rather than negative. That is the
EEG-averaging artefact running in reverse, and building it into the primary
test of *this* project would be a self-inflicted wound.

Because decoding is deterministic, each prompt can be tracked across alpha as
its own curve. So: log the full per-prompt margin trajectory and ask whether
individual curves step. Sharp per-prompt rises at jittered alpha = ignition
with a distributed threshold, invisible to any pooled test. Every prompt
tracing the same smooth compressive curve = saturation, and a far stronger null
than a dip test could give.

The pooled dip test is kept as a secondary check only. It also cannot separate
bistability from prompt *heterogeneity* (some prompts simply being more
spider-adjacent): two clusters, no threshold anywhere. The per-prompt view
separates them -- if mode membership is fixed by prompt identity across all
alpha it is heterogeneity; if prompts cross between modes as alpha rises it is
a threshold.

alpha=0 is recorded for every prompt so trajectories can be normalised to the
change rather than the level; prompts differ in baseline concept affinity.
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from common import CONFIGS, OUT, add_model_arg, depth_plan, lens_path, load_model
from lens_ops import (CoordinatePatch, MatchedDeltaPatch, coordinate_rms,
                      intervene, random_direction)
from restricted import RestrictedLens
import tokens as toks

ap = argparse.ArgumentParser()
add_model_arg(ap)
ap.add_argument("--tag", default="_3x")
ap.add_argument("--concept", default="spider")
ap.add_argument("--n-prompts", type=int, default=250)
ap.add_argument("--alpha-min", type=float, default=0.0)
ap.add_argument("--alpha-max", type=float, default=24.0)
ap.add_argument("--n-alpha", type=int, default=25)
ap.add_argument("--max-len", type=int, default=64)
args = ap.parse_args()

NL = CONFIGS[args.model]["n_layers"]
plan = depth_plan(NL); INJECT = plan["inject_layer"]
model, hf, tok, device, dtype = load_model(args.model)
lens = RestrictedLens.load(lens_path(args.model, args.tag))
tracked, seed_only = toks.build(tok, OUT)
concept_id = tracked[args.concept]["id"]

# --- prompt pool: held out from the fit, and neutral -------------------------
CLUSTER = set("spider web insect legs eight piano music keys sound notes "
              "arachnid spiders webs cobweb spidery".split())
import re
fit_used = json.load(open(os.path.join(OUT, "fit_prompts.json")))
n_fit = len(fit_used)
from jlens.examples import load_wikitext_prompts
pool_all = load_wikitext_prompts(n_fit + 4 * args.n_prompts)
assert pool_all[:n_fit] == fit_used, (
    "wikitext prompt order is not reproducible; cannot guarantee held-out set")
held = pool_all[n_fit:]
def neutral(p):
    return not (set(re.findall(r"[a-z]+", p.lower())) & CLUSTER) and len(p) >= 40
prompts = [p for p in held if neutral(p)][: args.n_prompts]
print(f"prompt pool: {len(prompts)} held-out neutral prompts "
      f"(fit used the first {n_fit}; none reused)", flush=True)
assert len(prompts) >= 50, f"only {len(prompts)} usable held-out prompts"

d_con = lens.direction(INJECT, concept_id)
d_rnd = random_direction(lens.d_model, 20260819)
SCALE = coordinate_rms(model, prompts[:40], INJECT, d_con)
ALPHAS = [args.alpha_min + (args.alpha_max - args.alpha_min) * i / (args.n_alpha - 1)
          for i in range(args.n_alpha)]
print(f"model={CONFIGS[args.model]['name']} inject=L{INJECT} concept={args.concept!r}")
print(f"alpha unit={SCALE:.4f}  alphas={ALPHAS[0]:.1f}..{ALPHAS[-1]:.1f} "
      f"({args.n_alpha} values)  trials/alpha={len(prompts)}", flush=True)


@torch.no_grad()
def margin(prompt, patch=None):
    """(margin, concept_logit, best_other_logit, top1_id) at the last position."""
    ids = model.encode(prompt, max_length=args.max_len)
    if patch is None:
        logits = hf(ids).logits[0, -1].float()
    else:
        with intervene(model, INJECT, patch):
            logits = hf(ids).logits[0, -1].float()
    c = float(logits[concept_id])
    other = logits.clone(); other[concept_id] = float("-inf")
    j = int(other.argmax())
    return c - float(other[j]), c, float(other[j]), int(logits.argmax())


path = os.path.join(OUT, f"stage4_margins_{args.model}{args.tag}.jsonl.gz")
import gzip
t0 = time.time(); n = 0
with gzip.open(path, "wt") as fh:
    for pi, p in enumerate(prompts):
        m, c, o, t1 = margin(p, None)
        fh.write(json.dumps({"prompt": pi, "alpha": None, "condition": "baseline",
                             "margin": round(m, 5), "concept_logit": round(c, 5),
                             "best_other": round(o, 5), "top1_id": t1}) + "\n"); n += 1
        for a in ALPHAS:
            cp = CoordinatePatch(d_con, a * SCALE)
            m, c, o, t1 = margin(p, cp)
            fh.write(json.dumps({"prompt": pi, "alpha": a, "condition": "concept",
                                 "margin": round(m, 5), "concept_logit": round(c, 5),
                                 "best_other": round(o, 5), "top1_id": t1}) + "\n"); n += 1
            mp = MatchedDeltaPatch(d_rnd, cp.delta_norms_pos, sign=a if a else 1.0)
            m, c, o, t1 = margin(p, mp)
            fh.write(json.dumps({"prompt": pi, "alpha": a, "condition": "random_matched",
                                 "margin": round(m, 5), "concept_logit": round(c, 5),
                                 "best_other": round(o, 5), "top1_id": t1}) + "\n"); n += 1
        if (pi + 1) % 25 == 0:
            print(f"  prompt {pi+1}/{len(prompts)}  rows={n}", flush=True)
wall = time.time() - t0
json.dump({"model": CONFIGS[args.model]["name"], "concept": args.concept,
           "concept_id": concept_id, "inject_layer": INJECT, "alphas": ALPHAS,
           "alpha_unit_scale": SCALE, "n_prompts": len(prompts), "n_rows": n,
           "held_out_from_fit": True, "n_fit_prompts": n_fit,
           "wall_seconds": wall, "max_len": args.max_len},
          open(os.path.join(OUT, f"stage4_meta_{args.model}{args.tag}.json"), "w"), indent=1)
print(f"{n} rows -> {path}  ({wall/60:.1f} min)")
