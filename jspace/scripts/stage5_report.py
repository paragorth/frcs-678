"""Job 2: ignition sweep with a DOWNSTREAM-ANSWER report measure.

Top-1 margin was the wrong readout. Only 35/250 prompts ever reached margin>0,
so the sweep sat almost entirely below the regime where report happens, and the
coherent window never reached it -- the measure could not have detected report
even if it occurred.

The paper's notion of report is downstream answers changing, as in
France->China. So: inject the concept at coherent alpha, then ask a question
whose answer depends on whether the concept is deployed, and score the answer
by fact identity. That is measurable well inside the coherent window and does
not require the concept token to win argmax anywhere.

Everything else is unchanged from Stage 4: margin not rank for the continuous
channel, variance from prompts not seeds, per-prompt trajectories not pooled,
held-out prompts asserted against the fit corpus, random_matched at every
alpha, alpha=0 recorded for normalisation.
"""
from __future__ import annotations
import argparse, gzip, json, os, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from common import CONFIGS, OUT, add_model_arg, depth_plan, lens_path, load_model
from lens_ops import (CoordinatePatch, MatchedDeltaPatch, coordinate_rms,
                      intervene, random_direction)
from restricted import RestrictedLens
import tokens as toks

ap = argparse.ArgumentParser()
add_model_arg(ap)
ap.add_argument("--tag", default="")  # selects the lens checkpoint
ap.add_argument("--out-tag", default=None)  # names the output; defaults to --tag
ap.add_argument("--concept", default="spider")
ap.add_argument("--n-prompts", type=int, default=120)
ap.add_argument("--alpha-max", type=float, default=24.0)
ap.add_argument("--n-alpha", type=int, default=13)
ap.add_argument("--gen-tokens", type=int, default=10)
args = ap.parse_args()

# Probes whose answers depend on the concept being deployed. Aliases are claims
# about the world, not scoring conveniences.
PROBES = [
    ("How many legs does it have?", {"eight", "8"}),
    ("What does it spin?", {"web", "webs", "silk"}),
    ("What kind of creature is it?", {"spider", "arachnid"}),
]
CONTRAST = [  # cluster B: must NOT light up if the injection is specific
    ("What instrument is it?", {"piano", "keyboard"}),
]

NL = CONFIGS[args.model]["n_layers"]
plan = depth_plan(NL); INJECT = plan["inject_layer"]
model, hf, tok, device, dtype = load_model(args.model)
lens = RestrictedLens.load(lens_path(args.model, args.tag))
tracked, seed_only = toks.build(tok, OUT)
concept_id = tracked[args.concept]["id"]

CLUSTER = set("spider web insect legs eight piano music keys sound notes "
              "arachnid spiders webs cobweb silk".split())
fit_used = json.load(open(os.path.join(OUT, "fit_prompts.json")))
n_fit = len(fit_used)
from jlens.examples import load_wikitext_prompts
pool = load_wikitext_prompts(n_fit + 6 * args.n_prompts)
assert pool[:n_fit] == fit_used, "prompt order not reproducible; cannot hold out"
held = pool[n_fit:]
prompts = [p for p in held
           if not (set(re.findall(r"[a-z]+", p.lower())) & CLUSTER) and len(p) >= 40
           ][: args.n_prompts]
assert len(prompts) >= 40, f"only {len(prompts)} held-out prompts"
print(f"{len(prompts)} held-out neutral prompts (fit used first {n_fit})", flush=True)

d_con = lens.direction(INJECT, concept_id)
d_rnd = random_direction(lens.d_model, 20260819)
SCALE = coordinate_rms(model, prompts[:40], INJECT, d_con)
ALPHAS = [args.alpha_max * i / (args.n_alpha - 1) for i in range(args.n_alpha)]
print(f"inject L{INJECT} alpha unit={SCALE:.4f} alphas 0..{args.alpha_max} "
      f"({args.n_alpha})", flush=True)


def names(text, accept):
    return bool(set(re.findall(r"[a-z0-9]+", text.lower())) & accept)


def names_strict(text, accept):
    """Strict report: alphabetic word-forms only, asserted early in the answer.

    The lenient matcher accepts any accept-set token anywhere in the
    continuation, including bare numerals. `PROBES[0]` accepts "8", and the
    held-out pool is wikitext, so scorelines and years scored as concept
    reports -- 26 of 27 probe0 hits on 7B, and the ENTIRE apparent signal of the
    original 14B sweep. Numerals are dropped and the match must fall in the
    first six words, i.e. the model must be answering the question rather than
    happening to contain the token."""
    acc = {a for a in accept if a.isalpha()}
    return bool(set(re.findall(r"[a-z]+", text.lower())[:6]) & acc)


def degeneracy(text):
    """Repeat-fraction of the generated continuation. The coherence gate is now
    recorded PER TRIAL rather than as a separate after-the-fact check: the Job 2
    headline had to be withdrawn because report was scored at alphas where the
    model emitted 97% pure repetition, and a spider-flavoured collapse scores 0%
    on the cluster-B contrast probe while being entirely degenerate."""
    w = text.split()
    return 1 - len(set(w)) / max(len(w), 1)


@torch.no_grad()
def probe(prompt, question, patch):
    """Inject into the passage, then ask the question and score the answer."""
    full = f"{prompt}\n\nQ: {question}\nA:"
    ids = model.encode(full, max_length=128)
    ctx = intervene(model, INJECT, patch) if patch else None
    if ctx:
        with ctx:
            out = hf.generate(ids, max_new_tokens=args.gen_tokens,
                              do_sample=False, pad_token_id=tok.eos_token_id)
    else:
        out = hf.generate(ids, max_new_tokens=args.gen_tokens, do_sample=False,
                          pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


@torch.no_grad()
def margin_of(prompt, patch):
    ids = model.encode(prompt, max_length=64)
    if patch is None:
        lg = hf(ids).logits[0, -1].float()
    else:
        with intervene(model, INJECT, patch):
            lg = hf(ids).logits[0, -1].float()
    c = float(lg[concept_id]); o = lg.clone(); o[concept_id] = -1e30
    return c - float(o.max())


OT = args.out_tag if args.out_tag is not None else args.tag
path = os.path.join(OUT, f"stage5_report_{args.model}{OT}.jsonl.gz")
t0, n = time.time(), 0
with gzip.open(path, "wt") as fh:
    for pi, p in enumerate(prompts):
        for a in ALPHAS:
            cp = CoordinatePatch(d_con, a * SCALE)
            m = margin_of(p, cp if a > 0 else CoordinatePatch(d_con, 0.0))
            rec = {"prompt": pi, "alpha": a, "condition": "concept",
                   "margin": round(m, 5)}
            degs = []
            for qi, (q, acc) in enumerate(PROBES):
                g = probe(p, q, cp)
                rec[f"probe{qi}"] = g.strip()[:60]
                rec[f"probe{qi}_hit"] = names(g, acc)
                rec[f"probe{qi}_strict"] = names_strict(g, acc)
                degs.append(degeneracy(g))
            rec["repeat_frac"] = round(sum(degs) / len(degs), 4)
            for qi, (q, acc) in enumerate(CONTRAST):
                g = probe(p, q, cp)
                rec[f"contrast{qi}_hit"] = names(g, acc)
            rec["report_hits"] = sum(rec[f"probe{i}_hit"] for i in range(len(PROBES)))
            rec["report_strict"] = sum(rec[f"probe{i}_strict"] for i in range(len(PROBES)))
            fh.write(json.dumps(rec) + "\n"); n += 1
            # matched-norm random control at the same alpha
            mp = MatchedDeltaPatch(d_rnd, cp.delta_norms_pos, sign=a if a else 1.0)
            mr = {"prompt": pi, "alpha": a, "condition": "random_matched",
                  "margin": round(margin_of(p, mp), 5)}
            mdegs = []
            for qi, (q, acc) in enumerate(PROBES):
                g = probe(p, q, mp)
                mr[f"probe{qi}"] = g.strip()[:60]
                mr[f"probe{qi}_hit"] = names(g, acc)
                mr[f"probe{qi}_strict"] = names_strict(g, acc)
                mdegs.append(degeneracy(g))
            mr["repeat_frac"] = round(sum(mdegs) / len(mdegs), 4)
            for qi, (q, acc) in enumerate(CONTRAST):
                g = probe(p, q, mp)
                mr[f"contrast{qi}_hit"] = names(g, acc)
            mr["report_hits"] = sum(mr[f"probe{i}_hit"] for i in range(len(PROBES)))
            mr["report_strict"] = sum(mr[f"probe{i}_strict"] for i in range(len(PROBES)))
            fh.write(json.dumps(mr) + "\n"); n += 1
        if (pi + 1) % 10 == 0:
            print(f"  prompt {pi+1}/{len(prompts)} rows={n}", flush=True)
wall = time.time() - t0
json.dump({"model": CONFIGS[args.model]["name"], "concept": args.concept,
           "inject_layer": INJECT, "alphas": ALPHAS, "alpha_unit_scale": SCALE,
           "n_prompts": len(prompts), "n_rows": n, "probes": [p[0] for p in PROBES],
           "contrast": [c[0] for c in CONTRAST], "wall_seconds": wall},
          open(os.path.join(OUT, f"stage5_meta_{args.model}{OT}.json"), "w"), indent=1)
print(f"{n} rows -> {path} ({wall/60:.1f} min)")
