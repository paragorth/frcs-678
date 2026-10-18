"""Stage 1: fit and cache the restricted Jacobian lens. Resumable.

Refuses to refit over an existing lens file (pass --tag to fit a variant, e.g.
the 3x refit Stage 2 falls back to).
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import transformers
import jlens
from common import (OUT, CONFIGS, add_model_arg, depth_plan, depth_plan_at, lens_path,
                    load_model, peak_rss_gb, peak_vram_gb)
from restricted import RestrictedLens, fit_restricted
import tokens as toks

ap = argparse.ArgumentParser()
add_model_arg(ap)
ap.add_argument("--n-prompts", type=int, default=500)
ap.add_argument("--seed-batch", type=int, default=32)
ap.add_argument("--max-seq-len", type=int, default=256)
ap.add_argument("--tag", default="", help="suffix for a variant fit, e.g. _3x")
ap.add_argument("--inject", type=int, default=None,
                help="explicit injection layer; band from depth_plan_at. "
                     "Omitted -> the inherited depth_plan formula.")
ap.add_argument("--crosscheck", type=int, default=0,
                help="also fit the full J on N prompts and compare directions")
args = ap.parse_args()

LENS = lens_path(args.model, args.tag)
CKPT = os.path.join(OUT, f"lens_ckpt{args.tag}_{args.model}.pt")
if os.path.exists(LENS):
    print(f"lens already exists at {LENS} -- not refitting."); sys.exit(0)

NL_ = CONFIGS[args.model]["n_layers"]
plan = depth_plan(NL_) if args.inject is None else depth_plan_at(args.inject, NL_)
prompts = json.load(open(os.path.join(OUT, "fit_prompts.json")))[: args.n_prompts]
model, hf, tok, device, dtype = load_model(args.model)
tracked, seed_only = toks.build(tok, OUT)
seeds = toks.seed_table(tracked, seed_only)

print(f"model={CONFIGS[args.model]['name']} device={device} dtype={dtype}")
print(f"inject=L{plan['inject_layer']} band=L{plan['band'][0]}..L{plan['band'][-1]} "
      f"source_layers={plan['source_layers']}")
print(f"fitting {len(prompts)} prompts, {len(seeds)} seed rows, "
      f"seed_batch={args.seed_batch}, max_seq_len={args.max_seq_len}", flush=True)

t0 = time.time()
lens = fit_restricted(model, prompts, seeds,
                      source_layers=plan["source_layers"],
                      seed_batch=args.seed_batch, max_seq_len=args.max_seq_len,
                      checkpoint_path=CKPT, checkpoint_every=10, resume=True,
                      log=lambda s: print(s, flush=True))
wall = time.time() - t0

meta = {"model": CONFIGS[args.model]["name"], "n_prompts": lens.n_prompts,
        "n_seed": len(seeds), "source_layers": plan["source_layers"],
        "inject_layer": plan["inject_layer"], "band": plan["band"],
        "wall_seconds": wall, "peak_rss_gb": peak_rss_gb(),
        "peak_vram_gb": peak_vram_gb(), "device": str(device),
        "dtype": str(dtype), "seed_batch": args.seed_batch,
        "max_seq_len": args.max_seq_len, "tag": args.tag}
lens.meta = meta
lens.save(LENS)
json.dump(meta, open(os.path.join(OUT, f"fit_meta{args.tag}_{args.model}.json"), "w"),
          indent=1)
print(json.dumps(meta, indent=1))
print(f"saved -> {LENS}  ({wall/60:.1f} min)")

# ---- cross-check: full J on the same prompts, compare readout directions ----
# Restricted seeding should return exactly J^T c for the library's estimator,
# so agreement here is a correctness test of the restricted path, not a
# similarity metric. Anything below ~0.999 cosine means the two disagree.
if args.crosscheck:
    sub = prompts[: args.crosscheck]
    print(f"\ncross-check: full-J fit on {len(sub)} prompts", flush=True)
    t1 = time.time()
    full = jlens.fit(model, sub, source_layers=plan["source_layers"],
                     dim_batch=args.seed_batch, max_seq_len=args.max_seq_len,
                     checkpoint_path=None, checkpoint_every=None, resume=False)
    ref = fit_restricted(model, sub, seeds, source_layers=plan["source_layers"],
                         seed_batch=args.seed_batch, max_seq_len=args.max_seq_len,
                         checkpoint_path=None, checkpoint_every=None,
                         resume=False, log=lambda s: None)
    from restricted import cotangent_rows
    cot = cotangent_rows(model, [s["id"] for s in seeds]).cpu()
    report = {}
    for l in plan["source_layers"]:
        J = full.jacobians[l].float()                 # [d, d]
        want = (cot @ J).float()                      # rows of J^T c
        want = want / want.norm(dim=-1, keepdim=True)
        got = ref.matrix(l)
        cos = (want * got).sum(-1)
        report[l] = {"cos_min": cos.min().item(), "cos_mean": cos.mean().item()}
        print(f"  L{l}: cos min={cos.min():.6f} mean={cos.mean():.6f}", flush=True)
    report["_wall_seconds"] = time.time() - t1
    report["_n_prompts"] = len(sub)
    # Per-tag filename: with a shared name the depth sweep overwrote this on
    # every fit and 8 of 9 cos_min values were lost.
    json.dump(report, open(os.path.join(OUT,
              f"crosscheck_{args.model}{args.tag}.json"), "w"),
              indent=1)
    worst = min(v["cos_min"] for k, v in report.items() if isinstance(v, dict))
    print(f"cross-check worst cosine: {worst:.6f} "
          f"({'PASS' if worst > 0.999 else 'FAIL'})")
