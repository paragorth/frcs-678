"""Stage 2: France->China multi-fact swap. THE VALIDATION GATE.

Scoring is on FACT IDENTITY, not token string. The previous scorer compared the
model's top-1 token against a target token id, which failed twice for reasons
that had nothing to do with the model:

  * baseline emitted `' euro'` against an expected `'Euro'` -- a *correct*
    answer scored wrong, capping baseline at 3/4 and making PASS unreachable
  * swap emitted `' ren'` -- the leading token of "Renminbi", the right fact
    under a different tokenisation, scored wrong

So: greedily continue a few tokens and check whether the continuation names the
fact, case- and leading-space-insensitively, allowing known aliases. A rank
measure is kept alongside, taken as the *best* rank over the accept-set rather
than over a single id.

The scorer is validated against the unintervened baseline before any
intervention is scored. If it cannot score an unintervened model correctly it
cannot score an intervened one, so a baseline below the gate aborts the run
rather than producing a number.
"""
from __future__ import annotations
import argparse, contextlib, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from common import (CONFIGS, OUT, add_model_arg, depth_plan, depth_plan_at,
                    lens_path, load_model)
from lens_ops import intervene, make_swapper
from restricted import RestrictedLens

ap = argparse.ArgumentParser()
add_model_arg(ap)
ap.add_argument("--tag", default="")
ap.add_argument("--inject", type=int, default=None)
ap.add_argument("--src", default="France")
ap.add_argument("--dst", default="China")
ap.add_argument("--gen-tokens", type=int, default=6)
args = ap.parse_args()

# Aliases: alternative surface names for the same fact. Membership here is a
# claim about the world, not a scoring convenience -- Renminbi and Yuan are the
# same currency; Peking is the same city.
ALIASES = {
    "yuan": {"yuan", "renminbi", "rmb", "cny"},
    "euro": {"euro", "euros", "eur"},
    "beijing": {"beijing", "peking"},
    "paris": {"paris"},
    "french": {"french"},
    "chinese": {"chinese", "mandarin"},
    "europe": {"europe"},
    "asia": {"asia"},
}


def accept_set(fact: str) -> set[str]:
    return ALIASES.get(fact.strip().lower(), {fact.strip().lower()})


def names_fact(text: str, fact: str) -> bool:
    """Does a generated continuation name ``fact``? Case- and space-insensitive,
    alias-aware, matched on word boundaries within the first few tokens."""
    import re
    words = set(re.findall(r"[a-z]+", text.lower()))
    return bool(words & accept_set(fact))


def accept_token_ids(tok, fact: str) -> list[int]:
    """Token ids whose decoded form is (a prefix-free match for) the fact.
    Used only for the rank measure, never for the pass/fail decision."""
    ids = []
    for surface in accept_set(fact):
        for form in (" " + surface, surface, " " + surface.capitalize(),
                     surface.capitalize()):
            enc = tok.encode(form, add_special_tokens=False)
            if enc:
                ids.append(enc[0])
    return sorted(set(ids))


NL = CONFIGS[args.model]["n_layers"]
plan = depth_plan(NL) if args.inject is None else depth_plan_at(args.inject, NL)
BAND = plan["band"]
model, hf, tok, device, dtype = load_model(args.model)
lens = RestrictedLens.load(lens_path(args.model, args.tag))

# The data files live next to the code, NOT next to the output directory.
# JLENS_OUT points at /workspace/out on the pods while the code sits in
# /root/jlens, so deriving this from OUT silently broke Stage 2 -- and an
# empty verdict then read as a FAILURE and triggered a needless refit.
CODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.join(CODE_ROOT, "jacobian-lens")
if not os.path.isdir(REPO):
    sys.exit(f"cannot find jacobian-lens data at {REPO}")
cats = json.load(open(os.path.join(
    REPO, "data/experiments/flexible-generalization.json")))["categories"]
countries = next(c for c in cats if c["name"] == "countries")

src_id = tok.encode(" " + args.src, add_special_tokens=False)
dst_id = tok.encode(" " + args.dst, add_special_tokens=False)
assert len(src_id) == 1 and len(dst_id) == 1, (src_id, dst_id)
src_id, dst_id = src_id[0], dst_id[0]
dirs = {l: (lens.direction(l, src_id), lens.direction(l, dst_id)) for l in BAND}


@torch.no_grad()
def ask(prompt, stack_fns=None):
    """Greedy continuation + logits at the answer position."""
    ids = model.encode(prompt, max_length=128)
    with contextlib.ExitStack() as st:
        for layer, fn in (stack_fns or []):
            st.enter_context(intervene(model, layer, fn))
        out = hf.generate(ids, max_new_tokens=args.gen_tokens, do_sample=False,
                          pad_token_id=tok.eos_token_id)
        logits = hf(ids).logits[0, -1].float()
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True), logits


def best_rank(logits, fact):
    ids = accept_token_ids(tok, fact)
    if not ids:
        return None
    order = logits.argsort(descending=True)
    pos = {int(t): i for i, t in enumerate(order[:200000].tolist())}
    return min(pos.get(i, 10 ** 9) for i in ids) + 1


rows, t0 = [], time.time()
for func in countries["funcs"]:
    prompt = func["template"].format(arg=args.src)
    a_src, a_dst = func["answers"][args.src], func["answers"][args.dst]
    gen_b, lg_b = ask(prompt)
    swap = [(l, make_swapper(dirs[l][0], dirs[l][1])) for l in BAND]
    gen_s, lg_s = ask(prompt, swap)
    r = {"func": func["name"], "prompt": prompt,
         "expect_src": a_src, "expect_dst": a_dst,
         "baseline_gen": gen_b, "swapped_gen": gen_s,
         "baseline_correct": names_fact(gen_b, a_src),
         "swapped_correct": names_fact(gen_s, a_dst),
         "swapped_still_src": names_fact(gen_s, a_src),
         "baseline_rank_dst": best_rank(lg_b, a_dst),
         "swapped_rank_dst": best_rank(lg_s, a_dst),
         "swapped_rank_src": best_rank(lg_s, a_src)}
    r["moved"] = bool((not r["swapped_still_src"]) and r["baseline_rank_dst"]
                      and r["swapped_rank_dst"]
                      and r["baseline_rank_dst"] >= 10 * r["swapped_rank_dst"])
    rows.append(r)
    print(f"{r['func']:10s} base={gen_b[:28]!r} ok={r['baseline_correct']} | "
          f"swap={gen_s[:28]!r} ok={r['swapped_correct']} moved={r['moved']} "
          f"rank_dst {r['baseline_rank_dst']}->{r['swapped_rank_dst']}", flush=True)

n = len(rows)
n_base = sum(r["baseline_correct"] for r in rows)
n_swap = sum(r["swapped_correct"] for r in rows)
n_moved = sum(r["moved"] for r in rows)
# Stricter gate, from fields already recorded: the swap must name the
# destination fact AND no longer name the source. The base scorer is already
# alphabetic-alias-only and the continuation is capped at 6 tokens, so it
# already satisfies the log-24 strictness criteria; this is a further tightening
# reported alongside, not a replacement.
n_strict = sum(r["swapped_correct"] and not r["swapped_still_src"] for r in rows)

# --- scorer validation gate ------------------------------------------------
# If the scorer cannot recover the unintervened model's own facts, no
# intervened number it produces means anything.
if n_base < n:
    verdict = "SCORER_OR_MODEL_FAILS_BASELINE"
elif n_swap >= 3 and n_moved == n:
    verdict = "PASS"
elif n_moved >= 3:
    verdict = "PARTIAL"
else:
    verdict = "FAIL"

print(f"\nbaseline {n_base}/{n} | swapped {n_swap}/{n} | strict {n_strict}/{n} | moved {n_moved}/{n}")
print(f"VERDICT: {verdict}")
if verdict == "SCORER_OR_MODEL_FAILS_BASELINE":
    print("  baseline is not fully recovered -- the scorer is not validated on "
          "this model, so the swap numbers above are NOT interpretable.")
out = {"model": CONFIGS[args.model]["name"], "tag": args.tag, "rows": rows,
       "n": n, "baseline_correct": n_base, "swap_correct": n_swap,
       "swap_strict": n_strict, "inject_layer": plan["inject_layer"],
       "band_width": len(BAND),
       "moved": n_moved, "verdict": verdict, "band": BAND,
       "wall_seconds": time.time() - t0}
path = os.path.join(OUT, f"stage2_verdict_{args.model}{args.tag}.json")
json.dump(out, open(path, "w"), indent=1)
print(f"-> {path}")
