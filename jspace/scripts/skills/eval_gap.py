"""Measure the fail-without / pass-with gap for every skill instance.

Both conditions get the *same* output-format instruction. Only the procedural
content differs. Without that control the empty-context condition would fail
partly for formatting reasons, and the gap would be inflated by an artefact
rather than measuring whether the procedure is available.

A skill the base model already passes without the file is discarded: there is
nothing to install, so it contributes only noise to Phase 3.
"""
from __future__ import annotations
import argparse, json, os, re, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch
from common import CONFIGS, OUT, add_model_arg, load_model

ap = argparse.ArgumentParser()
add_model_arg(ap)
ap.add_argument("--max-new", type=int, default=48)
ap.add_argument("--keep-threshold", type=float, default=0.25,
                help="discard a skill whose no-skill pass rate exceeds this")
args = ap.parse_args()

SKILLS = os.path.join(OUT, "skills")
FORMAT = ("\n\nRespond with a single line beginning `ANSWER:` followed by only "
          "the answer, with no explanation.")

model, hf, tok, device, dtype = load_model(args.model)


def extract(text: str) -> str:
    m = re.findall(r"ANSWER:\s*(.+)", text)
    out = (m[-1] if m else text.strip().splitlines()[-1] if text.strip() else "")
    return re.sub(r"[`*_.]+$", "", out.strip()).strip()


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip()).upper()


@torch.no_grad()
def answer(prompt: str) -> str:
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").input_ids.to(model.input_device)
    out = hf.generate(ids, max_new_tokens=args.max_new, do_sample=False,
                      pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


manifest = json.load(open(os.path.join(SKILLS, "manifest.json")))
results, rows = {}, []
for name in sorted(manifest):
    d = os.path.join(SKILLS, name)
    spec = open(os.path.join(d, "skill.md")).read()
    inst = json.load(open(os.path.join(d, "instances.json")))
    n_wo = n_w = 0
    per = []
    t0 = time.time()
    for it in inst:
        g_wo = answer(it["prompt"] + FORMAT)
        g_w = answer(spec + "\n\n---\n\nTask: " + it["prompt"] + FORMAT)
        ok_wo = norm(extract(g_wo)) == norm(it["answer"])
        ok_w = norm(extract(g_w)) == norm(it["answer"])
        n_wo += ok_wo; n_w += ok_w
        per.append({"id": it["id"], "answer": it["answer"],
                    "got_without": extract(g_wo), "got_with": extract(g_w),
                    "ok_without": ok_wo, "ok_with": ok_w})
    r_wo, r_w = n_wo / len(inst), n_w / len(inst)
    keep = r_wo <= args.keep_threshold and r_w > r_wo
    results[name] = {"n": len(inst), "pass_without": r_wo, "pass_with": r_w,
                     "gap": r_w - r_wo, "keep": keep,
                     "wall_seconds": time.time() - t0}
    json.dump(per, open(os.path.join(d, f"gap_{args.model}.json"), "w"), indent=1)
    rows.append((name, len(inst), r_wo, r_w, r_w - r_wo, keep))
    print(f"  {name:10s} n={len(inst):3d}  without={r_wo:5.1%}  with={r_w:5.1%}  "
          f"gap={r_w-r_wo:+5.1%}  {'KEEP' if keep else 'DISCARD'}", flush=True)

print(f"\n{'skill':12s} {'n':>4} {'fail-without':>13} {'pass-with':>10} {'gap':>8}  verdict")
print("-" * 62)
for name, n, a, b, g, keep in sorted(rows, key=lambda r: -r[4]):
    print(f"{name:12s} {n:4d} {1-a:12.1%} {b:10.1%} {g:+8.1%}  {'KEEP' if keep else 'DISCARD'}")
kept = [r for r in rows if r[5]]
print(f"\n{len(kept)}/{len(rows)} skills kept"
      + (f"; widest gap: {max(kept, key=lambda r: r[4])[0]}" if kept else ""))
json.dump({"model": CONFIGS[args.model]["name"], "skills": results,
           "kept": [r[0] for r in kept],
           "widest": max(kept, key=lambda r: r[4])[0] if kept else None},
          open(os.path.join(SKILLS, f"gap_summary_{args.model}.json"), "w"), indent=1)
print(f"-> {os.path.join(SKILLS, f'gap_summary_{args.model}.json')}")
