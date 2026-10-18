"""Stage 3: continuous-alpha injection sweep with two controls.

Injects a concept's J-lens direction into the residual stream at ~25% depth
using the lens-coordinate patch form, and reads every tracked token's raw
projection in the middle-third workspace band.

Output is long-format, one row per (prompt, alpha, layer, tracked_token,
condition), gzipped JSONL. Nothing is averaged across trials, thresholded, or
pre-aggregated: the degraded-readout analysis depends on every raw
single-trial value surviving to disk.

Each row carries two raw readouts of the same trial:
  value          projection at the final token position (canonical)
  value_meanpos  projection averaged over non-BOS positions
The second is a within-prompt spatial reduction, not an average across trials,
and is included because incumbent content words sit early in these prompts --
reading only the last position could hide eviction rather than measure it.

Conditions:
  concept     patch along the injected concept's lens direction
  random      patch along a fixed random unit direction (magnitude artefact
              control: a step here means the step is not ignition)
  orthogonal  patch along a direction orthogonal to the tracked readout span
              (isolation control: the readout should barely move)
  baseline    no intervention at all, recorded once per prompt. alpha=0 under
              the patch form still clamps the concept coordinate to zero, so
              it is NOT the same thing as no intervention.
"""
import argparse, gzip, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from jlens.hooks import ActivationRecorder
from common import (OUT, CONFIGS, add_model_arg, depth_plan, lens_path,
                    load_model, peak_vram_gb)
from lens_ops import (CoordinatePatch, MatchedDeltaPatch, coordinate_rms,
                      intervene, layer_mean_residual_norm,
                      orthogonal_direction, random_direction)
from restricted import RestrictedLens
import tokens as toks

ap = argparse.ArgumentParser()
add_model_arg(ap)
ap.add_argument("--tag", default="")
ap.add_argument("--concept", default="spider")
ap.add_argument("--n-alpha", type=int, default=25)
ap.add_argument("--seed", type=int, default=20260819)
args = ap.parse_args()

plan = depth_plan(CONFIGS[args.model]["n_layers"])
INJECT, BAND = plan["inject_layer"], plan["band"]
model, hf, tok, device, dtype = load_model(args.model)
lens = RestrictedLens.load(lens_path(args.model, args.tag))
tracked, _ = toks.build(tok, OUT)
prompts = json.load(open(os.path.join(OUT, "sweep_prompts.json")))

# Readout table: for each band layer, the [n_tracked, d_model] unit directions.
words = list(tracked)
tids = [tracked[w]["id"] for w in words]
groups = [tracked[w]["group"] for w in words]
rowidx = torch.tensor([lens.index[i] for i in tids])
READOUT = {l: (lens.matrix(l)[rowidx]).to(device) for l in BAND}   # [n_tracked, d]

concept_id = tracked[args.concept]["id"]
DIRS = {
    "concept": lens.direction(INJECT, concept_id),
    "random": random_direction(lens.d_model, args.seed),
}
orth, leak = orthogonal_direction(lens, INJECT, args.seed + 1)
DIRS["orthogonal"] = orth

# alpha unit: the amount of the concept direction the residual normally
# carries. One common unit is used for every condition, so a given alpha means
# the same target magnitude everywhere ("same alpha values, same norms").
SCALE = coordinate_rms(model, prompts, INJECT, DIRS["concept"])
MRN = layer_mean_residual_norm(model, prompts, INJECT)
print(f"model={CONFIGS[args.model]['name']}  inject=L{INJECT}  "
      f"band=L{BAND[0]}..L{BAND[-1]}  concept={args.concept!r} id={concept_id}")
print(f"alpha unit = baseline RMS of the concept coordinate at L{INJECT} = {SCALE:.3f}")
print(f"  (mean residual norm there is {MRN:.1f}, i.e. {MRN/SCALE:.0f}x larger; using "
      f"that as the unit\n   would put the whole residual magnitude on one direction "
      f"and break every condition)")
print(f"tracked tokens: {len(words)}   orthogonal control leakage "
      f"(max |cos| vs tracked) = {leak:.2e}")
for name, d in DIRS.items():
    cos = float(torch.dot(d / d.norm(), DIRS['concept'] / DIRS['concept'].norm()))
    print(f"  dir {name:11s} |v|={d.norm():.4f}  cos vs concept={cos:+.4f}")


@torch.no_grad()
def run(prompt, patch=None):
    """One forward. Returns (last[n_tracked], meanpos[n_tracked]) per band
    layer, plus the applied delta norm."""
    ids = model.encode(prompt, max_length=64)
    ctx = intervene(model, INJECT, patch) if patch else None
    FINAL = model.n_layers - 1
    with ActivationRecorder(model.layers, at=[*BAND, FINAL]) as rec:
        if ctx:
            with ctx:
                model.forward(ids)
        else:
            model.forward(ids)
        acts = {l: rec.activations[l].detach() for l in [*BAND, FINAL]}
    out = {}
    for l in BAND:
        h = acts[l][0].float()                    # [S, d]
        P = h @ READOUT[l].float().T              # [S, n_tracked]
        out[l] = (P[-1].cpu(), P[1:].mean(0).cpu())   # skip BOS in the mean
    logits = model.unembed(acts[FINAL][0][-1:].float())[0].float().cpu()
    return out, (patch.delta_norm if patch else 0.0), logits


def behaviour(logits, base_logp=None):
    """Per-trial behavioural readout at the output, independent of the lens.

    The lens projection can be linear in alpha while the model's actual output
    still turns over sharply, so this is recorded as a second channel: if a
    step exists anywhere, it should be visible here even when the projection
    curve is straight.
    """
    logp = torch.log_softmax(logits, -1)
    top = int(logits.argmax())
    order = logits.argsort(descending=True)
    rank = int((order == concept_id).nonzero()[0, 0]) + 1
    out = {"top1_id": top, "top1": tok.decode([top]),
           "concept_logit": round(float(logits[concept_id]), 5),
           "concept_logprob": round(float(logp[concept_id]), 5),
           "concept_rank": rank,
           "entropy": round(float(-(logp.exp() * logp).sum()), 5)}
    if base_logp is not None:
        out["kl_from_baseline"] = round(
            float((base_logp.exp() * (base_logp - logp)).sum()), 6)
    return out, logp


CIDX = words.index(args.concept)


@torch.no_grad()
def baseline_stats(prompt_subset):
    """Per-(prompt, layer) baseline std of the tracked projections. This is the
    fixed yardstick the coarse metric is expressed in -- crucially it does NOT
    move with alpha."""
    out = {}
    for i, p in enumerate(prompt_subset):
        res, _, _ = run(p, None)
        for l in BAND:
            out[(i, l)] = float(res[l][0].std()) + 1e-9
    return out


def coarse_probe(prompt_subset, base_std, alpha):
    """Three metrics per coarse alpha, all reported.

    ``m``   BRACKETING METRIC. (concept - mean of tracked) / baseline std.
            Subtracting the tracked mean removes the large common-mode lift
            that injection applies to every tracked direction at once;
            dividing by the *baseline* std keeps it dose-dependent.
    ``z``   the same contrast divided by the *current* std. Diagnostic only.
            Reported to make the artefact visible rather than hide it: because
            the denominator grows with alpha at the same rate as the numerator,
            z is scale-invariant and shows a hard step at alpha=0 even when the
            underlying response is perfectly linear. Bracketing on z, or on
            arg, would manufacture the very step this experiment is testing for.
    ``arg`` fraction of cells where the concept is the argmax over tracked
            tokens. Diagnostic only; saturates the moment the concept leads.
    """
    ms, zs, hits, tot = [], [], 0, 0
    for i, p in enumerate(prompt_subset):
        res, _, _ = run(p, CoordinatePatch(DIRS["concept"], alpha * SCALE))
        for l in BAND:
            v = res[l][0]
            contrast = (v[CIDX] - v.mean()).item()
            ms.append(contrast / base_std[(i, l)])
            zs.append(contrast / (v.std().item() + 1e-9))
            hits += int(v.argmax().item() == CIDX)
            tot += 1
    return sum(ms) / len(ms), sum(zs) / len(zs), hits / tot


# ---------------- coarse scan: bracket where anything happens ----------------
COARSE = [-16, -8, -4, -2, -1, -0.5, -0.25, 0, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2,
          4, 8, 16, 32, 64]
sub = prompts[:5]
t0 = time.time()
BASE_STD = baseline_stats(sub)
cm, cz, ca = {}, {}, {}
print("\ncoarse scan over 5 prompts x band")
print("  m = (concept - tracked mean)/baseline std   [bracketing metric]")
print("  z = same contrast / current std             [scale-invariant, diagnostic]")
print("  arg = concept-is-argmax fraction            [binarised, diagnostic]")
for a in COARSE:
    cm[a], cz[a], ca[a] = coarse_probe(sub, BASE_STD, a)
    print(f"  alpha={a:>6}   m={cm[a]:>12.2f}   z={cz[a]:+7.3f}   arg={ca[a]:.3f}",
          flush=True)

# Bracketing rule, fixed in advance.
#
# First test whether the response is linear in alpha: for a linear system the
# per-alpha slope m(a)/a is constant, and there is no threshold to bracket --
# "where anything happens" is everywhere, and any bracket would be arbitrary.
# Only if the slope actually varies do we look for a sigmoid crossing.
pos = [a for a in COARSE if a > 0]
# Slope constancy must be tested on the intercept-corrected response. The
# metric has a non-zero value at alpha=0 (the clamp itself moves the
# coordinate), so raw m(a)/a is inflated at small |a| and a perfectly linear
# response reports as nonlinear. Subtract m(0) first.
m0 = cm.get(0, 0.0)
slopes = [(cm[a] - m0) / a for a in COARSE if a > 0]
ratio = max(slopes) / min(slopes) if min(slopes) > 0 else float("inf")
LINEAR = ratio < 1.5
print(f"\nslope (m(a)-m(0))/a over positive alpha: min={min(slopes):.2f} "
      f"max={max(slopes):.2f} ratio={ratio:.3f}  -> "
      f"{'LINEAR (no threshold)' if LINEAR else 'NONLINEAR'}")
print(f"  [uncorrected ratio would be "
      f"{max(cm[a]/a for a in COARSE if a>0)/min(cm[a]/a for a in COARSE if a>0):.3f}]")

if LINEAR:
    # The raw response has no threshold, so there is no crossing to bracket on
    # it. But the *binarised* readout does step, and the point of the dense
    # sweep is to have every raw single-trial value across exactly that region
    # -- otherwise the degraded-readout analysis has nothing to bite on. So the
    # range is chosen to span the apparent step in `arg`, with headroom.
    sat = [a for a in COARSE if a > 0 and ca[a] >= 0.99]
    hi = min(1.5 * sat[0], float(COARSE[-1])) if sat else float(COARSE[-1])
    lo, bracketed = -0.25 * hi, False
else:
    norm = {a: (cm[a] - min(cm.values())) / (max(cm.values()) - min(cm.values()) + 1e-12)
            for a in COARSE}
    below = [a for a in COARSE if norm[a] < 0.05]
    above = [a for a in COARSE if norm[a] > 0.95]
    bracketed = bool(below and above and min(above) > max(below))
    if bracketed:
        lo_i = max(0, COARSE.index(max(a for a in below if a < min(above))) - 1)
        hi_i = min(len(COARSE) - 1, COARSE.index(min(above)) + 1)
        lo, hi = float(COARSE[lo_i]), float(COARSE[hi_i])
    else:
        lo, hi = float(COARSE[0]), float(COARSE[-1])

ALPHAS = [lo + (hi - lo) * i / (args.n_alpha - 1) for i in range(args.n_alpha)]
print(f"dense range [{lo}, {hi}]  (sigmoid crossing used: {bracketed})")
print(f"{args.n_alpha} alphas from {ALPHAS[0]:.4f} to {ALPHAS[-1]:.4f} "
      f"(step {(hi-lo)/(args.n_alpha-1):.4f})")

# ------------------------------- dense sweep --------------------------------
path = os.path.join(OUT, f"stage3_sweep_{args.model}{args.tag}.jsonl.gz")
tpath = os.path.join(OUT, f"stage3_trials_{args.model}{args.tag}.jsonl.gz")
n_rows = 0
t1 = time.time()
with gzip.open(path, "wt") as fh, gzip.open(tpath, "wt") as tfh:
    def emit(pi, alpha, cond, res, dn):
        global n_rows
        for l in BAND:
            last, meanpos = res[l]
            for j, w in enumerate(words):
                fh.write(json.dumps({
                    "prompt": pi, "alpha": alpha, "layer": l, "token": w,
                    "token_id": tids[j], "group": groups[j], "condition": cond,
                    "value": round(float(last[j]), 6),
                    "value_meanpos": round(float(meanpos[j]), 6),
                    "delta_norm": round(float(dn), 4),
                }) + "\n")
                n_rows += 1

    n_trials = 0
    for pi, p in enumerate(prompts):
        res, _, lg = run(p, None)
        emit(pi, None, "baseline", res, 0.0)
        beh, base_logp = behaviour(lg)
        tfh.write(json.dumps({"prompt": pi, "alpha": None, "condition": "baseline",
                              "delta_norm": 0.0, **beh}) + "\n")
        n_trials += 1
        for a in ALPHAS:
            ref_norms = None
            for cond, d in DIRS.items():
                patch = CoordinatePatch(d, a * SCALE)
                res, dn, lg = run(p, patch)
                if cond == "concept":
                    ref_norms = patch.delta_norms_pos
                emit(pi, a, cond, res, dn)
                beh, _ = behaviour(lg, base_logp)
                tfh.write(json.dumps({"prompt": pi, "alpha": a, "condition": cond,
                                      "delta_norm": round(float(dn), 4),
                                      **beh}) + "\n")
                n_trials += 1
            # random direction, perturbation norm matched exactly per position
            # to what the concept condition just applied
            mp = MatchedDeltaPatch(DIRS["random"], ref_norms, sign=a)
            res, dn, lg = run(p, mp)
            emit(pi, a, "random_matched", res, dn)
            beh, _ = behaviour(lg, base_logp)
            tfh.write(json.dumps({"prompt": pi, "alpha": a,
                                  "condition": "random_matched",
                                  "delta_norm": round(float(dn), 4), **beh}) + "\n")
            n_trials += 1
        print(f"  prompt {pi+1}/{len(prompts)}  rows={n_rows}", flush=True)

wall = time.time() - t1
meta = {"model": CONFIGS[args.model]["name"], "tag": args.tag,
        "concept": args.concept, "concept_id": concept_id,
        "inject_layer": INJECT, "band": BAND, "alpha_unit_scale": SCALE,
        "coarse_grid": COARSE, "coarse_metric_m": {str(k): v for k, v in cm.items()},
        "coarse_metric_z": {str(k): v for k, v in cz.items()},
        "coarse_metric_argmax_frac": {str(k): v for k, v in ca.items()},
        "slope_ratio": ratio,
        "slope_ratio_intercept_corrected": True, "linear_response": LINEAR,
        "bracketed": bracketed, "bracket": [lo, hi], "alphas": ALPHAS,
        "n_prompts": len(prompts), "n_tracked": len(words),
        "conditions": list(DIRS) + ["random_matched", "baseline"],
        "alpha_unit_meaning": "multiples of the baseline RMS of the concept coordinate at the injection layer",
        "mean_residual_norm_at_inject": MRN,
        "orthogonal_leakage_max_abs_cos": leak, "seed": args.seed,
        "n_rows": n_rows, "n_trials": n_trials,
        "trials_file": os.path.basename(tpath),
        "trial_schema": ["prompt", "alpha", "condition", "delta_norm", "top1_id",
                         "top1", "concept_logit", "concept_logprob",
                         "concept_rank", "entropy", "kl_from_baseline"], "coarse_seconds": t1 - t0, "sweep_seconds": wall,
        "peak_vram_gb": peak_vram_gb(),
        "row_schema": ["prompt", "alpha", "layer", "token", "token_id", "group",
                       "condition", "value", "value_meanpos", "delta_norm"],
        "note": ("value = raw projection at the final token position; "
                 "value_meanpos = same trial, averaged over non-BOS positions. "
                 "No aggregation across trials, no thresholding.")}
json.dump(meta, open(os.path.join(OUT, f"stage3_meta_{args.model}{args.tag}.json"), "w"),
          indent=1)
print(f"\n{n_rows} rows -> {path}  ({wall/60:.1f} min, "
      f"{os.path.getsize(path)/1e6:.1f} MB gz)")
