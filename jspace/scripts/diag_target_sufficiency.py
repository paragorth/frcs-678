"""Is the J-space target SUFFICIENT to install the skill?

Supply the captured J-space state directly at inference, with the skill file
removed, and see whether the behaviour follows. One forward pass per prompt, no
training.

If patching the J-space component alone does not install the behaviour, then no
weight update aimed at that component will either — and the Phase 3 result stops
being about confinement and becomes about the target. That is a real finding and
far cheaper than any sweep.

Conditions:
  no_patch     floor: skill file removed, nothing patched
  patch_J      the test: J-space component clamped to the captured state
  patch_full   positive control: the WHOLE activation replaced. If this installs
               and patch_J does not, the state carries the skill and it is
               specifically the J-space restriction that is insufficient. If this
               also fails, the capture/patch site itself is wrong.
  with_file    ceiling: the skill file in context
"""
from __future__ import annotations
import argparse, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from common import CONFIGS, OUT, add_model_arg, depth_plan, lens_path, load_model
from lens_ops import intervene
from restricted import RestrictedLens
from jlens.hooks import ActivationRecorder

ap = argparse.ArgumentParser(); add_model_arg(ap)
ap.add_argument("--tag", default=""); ap.add_argument("--skill", default="beacon")
ap.add_argument("--n", type=int, default=24)
args = ap.parse_args()

NL = CONFIGS[args.model]["n_layers"]; plan = depth_plan(NL); INJ = plan["inject_layer"]
model, hf, tok, device, dtype = load_model(args.model)
lens = RestrictedLens.load(lens_path(args.model, args.tag))
SK = os.path.join(OUT, "skills")
spec = open(os.path.join(SK, args.skill, "skill.md")).read()
inst = json.load(open(os.path.join(SK, args.skill, "instances.json")))[: args.n]
FORMAT = ("\n\nRespond with a single line beginning `ANSWER:` followed by only "
          "the answer, with no explanation.")
R = lens.matrix(INJ).float().to(model.input_device)
U_, S_, _ = torch.linalg.svd(R.T, full_matrices=False)
RANK = int((S_ > S_[0] * 1e-3).sum()); Q = U_[:, :RANK].detach()
print(f"{CONFIGS[args.model]['name']} L{INJ} J-rank {RANK}/{Q.shape[0]}", flush=True)

chat = lambda t: tok.apply_chat_template([{"role": "user", "content": t}],
                                         tokenize=False, add_generation_prompt=True)
PJ = lambda h: (h @ Q) @ Q.T


@torch.no_grad()
def capture(text):
    ids = model.encode(text, max_length=512)
    with ActivationRecorder(model.layers, at=[INJ]) as rec:
        model.forward(ids)
    return rec.activations[INJ][0, -1].float().clone()


def patcher(target, full):
    """Patch the FINAL position only.

    An earlier version assigned to [:, :, :], broadcasting the captured
    last-token vector across every position. That overwrites the prompt's own
    representation, so `patch_full` produced garbage and could not serve as a
    positive control. During generation with KV caching the hook sees one token
    (seq==1), which IS the current final position.
    """
    def fn(h):
        # Patch the PROMPT forward only (seq > 1), at its final position.
        # Re-imposing the captured prompt-final state at every generation step
        # (seq == 1) pins layer INJ to one constant vector for every token and
        # forces degenerate output -- observed as 'ANSANSANS...'. The question is
        # whether the state AT THE DECISION POINT carries the skill, so the patch
        # belongs on the prompt and generation then runs free.
        if h.shape[1] == 1:
            return h
        t = target.to(h.device, h.dtype)
        out = h.clone()
        if full:
            out[:, -1, :] = t
        else:
            cur = out[:, -1, :].float()
            out[:, -1, :] = (cur - PJ(cur) + PJ(t.float().unsqueeze(0))).to(h.dtype)
        return out
    return fn


def norm(s): return re.sub(r"\s+", " ", s.strip()).upper()


@torch.no_grad()
def ask(text, fn=None):
    ids = model.encode(text, max_length=512)
    if fn is None:
        o = hf.generate(ids, max_new_tokens=32, do_sample=False,
                        pad_token_id=tok.eos_token_id)
    else:
        with intervene(model, INJ, fn):
            o = hf.generate(ids, max_new_tokens=32, do_sample=False,
                            pad_token_id=tok.eos_token_id)
    g = tok.decode(o[0][ids.shape[1]:], skip_special_tokens=True)
    m = re.findall(r"ANSWER:\s*(.+)", g)
    return re.sub(r"[`*_.]+$", "", (m[-1] if m else g.strip().split("\n")[-1]
                                    if g.strip() else "")).strip(), g


res = {k: 0 for k in ("no_patch", "patch_J", "patch_full", "with_file")}
samples = []
for it in inst:
    pw, pn = chat(spec + "\n\n---\n\nTask: " + it["prompt"] + FORMAT), chat(it["prompt"] + FORMAT)
    tgt = capture(pw)
    a0, _ = ask(pn)
    aJ, gJ = ask(pn, patcher(tgt, False))
    aF, _ = ask(pn, patcher(tgt, True))
    aW, _ = ask(pw)
    for k, a in (("no_patch", a0), ("patch_J", aJ), ("patch_full", aF), ("with_file", aW)):
        res[k] += norm(a) == norm(it["answer"])
    if len(samples) < 3:
        samples.append({"want": it["answer"], "no_patch": a0[:40],
                        "patch_J": aJ[:40], "patch_full": aF[:40], "with_file": aW[:40]})
n = len(inst)
print(f"\nTARGET SUFFICIENCY, {args.skill}, n={n}")
for k in ("no_patch", "patch_J", "patch_full", "with_file"):
    print(f"  {k:12s} {res[k]}/{n} = {res[k]/n:6.1%}")
for s in samples:
    print(f"   want={s['want']!r} | J={s['patch_J']!r} | full={s['patch_full']!r}")
suff = res["patch_J"] / n - res["no_patch"] / n
print(f"\nJ-space patch lifts install by {100*suff:+.1f} points")
if suff < 0.10:
    print("  -> J-SPACE TARGET IS NOT SUFFICIENT. No weight update aimed only at")
    print("     this component can install the skill. The Phase 3 result is about")
    print("     the TARGET, not about confinement.")
    if res["patch_full"] / n - res["no_patch"] / n > 0.10:
        print("  -> but the FULL activation does install it: the state carries the")
        print("     skill; the J-space restriction is what fails.")
    else:
        print("  -> the full activation does not install it either: the capture or")
        print("     patch SITE is wrong, not just the subspace.")
json.dump({"model": CONFIGS[args.model]["name"], "skill": args.skill, "n": n,
           "rates": {k: v / n for k, v in res.items()}, "lift_J": suff,
           "samples": samples},
          open(os.path.join(OUT, f"diag_sufficiency_{args.model}.json"), "w"), indent=1)
