"""Three cheap measurements, before any spend on training.

1. DIRECTION TOLERANCE FROM THE CORRECTED-NORM PEAK. The earlier 0.995 figure
   was measured falling from 20.8% (the raw captured target, whose norm is 6%
   low), not from the 70.8% ceiling. Scale the target to the model's natural norm
   FIRST, then perturb direction at that norm. Everything downstream is
   calibrated on this number.

2. RANK. PCA the (with-file - no-file) deltas across the corpus and patch rank-k
   truncations. Small k means the skill has a compact target and there is
   something concrete to aim at. Full rank means it is not compressible into a
   state — the trajectory answer arriving by another route.

3. THE 29%. Even a perfect patch leaves ~29% of prompts failing. Same prompts
   every time, or random? Consistent failures are a property of those prompts
   (and possibly of corpus construction); random failures mean something the
   skill needs is not in the patched slice.

Alignment comes from the shared helpers. Do not re-derive it here.
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


def _csl(a, b):
    n = 0
    while n < len(a) and n < len(b) and a[-1 - n] == b[-1 - n]:
        n += 1
    return max(n, 1)


def align_len(ta, tb):
    return _csl(tok(ta, add_special_tokens=False).input_ids,
                tok(tb, add_special_tokens=False).input_ids)


@torch.no_grad()
def cap(text):
    ids = model.encode(text, max_length=512)
    with ActivationRecorder(model.layers, at=LAYERS) as rec:
        model.forward(ids)
    return {l: rec.activations[l][0].float().clone() for l in LAYERS}


def norm_ans(s): return re.sub(r"\s+", " ", s.strip()).upper()


@torch.no_grad()
def ask(text, hooks=()):
    ids = model.encode(text, max_length=512)
    with contextlib.ExitStack() as st:
        for l, fn in hooks:
            st.enter_context(intervene(model, l, fn))
        o = hf.generate(ids, max_new_tokens=32, do_sample=False,
                        pad_token_id=tok.eos_token_id)
    g = tok.decode(o[0][ids.shape[1]:], skip_special_tokens=True)
    m = re.findall(r"ANSWER:\s*(.+)", g)
    return re.sub(r"[`*_.]+$", "", (m[-1] if m else g.strip().split("\n")[-1]
                                    if g.strip() else "")).strip()


def mk(tvec, l, k):
    def fn(h):
        if h.shape[1] == 1:
            return h
        out = h.clone(); kk = min(k, out.shape[1], tvec.shape[0])
        out[:, -kk:, :] = tvec[-kk:].to(h.dtype)
        return out
    return fn


# ---------- gather aligned pairs once ---------------------------------------
PAIRS = []
for it in inst:
    pw, pn = chat(spec + "\n\n---\n\nTask: " + it["prompt"] + FORMAT), chat(it["prompt"] + FORMAT)
    k = align_len(pn, pw)
    cw, cn = cap(pw), cap(pn)
    PAIRS.append({"it": it, "pw": pw, "pn": pn, "k": k, "cw": cw, "cn": cn})
print(f"{CONFIGS[args.model]['name']} band L{LAYERS[0]}-{LAYERS[-1]} n={len(PAIRS)} "
      f"aligned {sum(p['k'] for p in PAIRS)/len(PAIRS):.1f}", flush=True)


def corrected_target(p, l):
    """Captured with-file state, rescaled per position to the model's NATURAL
    no-file norm. That correction is worth ~50 install points on its own."""
    k = min(p["k"], p["cw"][l].shape[0], p["cn"][l].shape[0])
    t, nat = p["cw"][l][-k:], p["cn"][l][-k:]
    return t * (nat.norm(dim=-1, keepdim=True) / (t.norm(dim=-1, keepdim=True) + 1e-9)), k


# ---------- 1. direction tolerance FROM THE CORRECTED-NORM PEAK -------------
print("\n=== 1. direction tolerance, from the corrected-norm peak ===", flush=True)
SIG = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
per_inst = {s: [] for s in SIG}
for s in SIG:
    ok = 0
    for p in PAIRS:
        g = torch.Generator(device="cpu").manual_seed(99)
        hooks, coss = [], []
        for l in LAYERS:
            t, k = corrected_target(p, l)
            if s > 0:
                nz = t + torch.randn(t.shape, generator=g, device="cpu").to(t.device) * s
                nz = nz * (t.norm(dim=-1, keepdim=True) / (nz.norm(dim=-1, keepdim=True) + 1e-9))
            else:
                nz = t
            coss.append(float(torch.nn.functional.cosine_similarity(
                nz.flatten(), t.flatten(), dim=0)))
            hooks.append((l, mk(nz, l, k)))
        hit = norm_ans(ask(p["pn"], hooks)) == norm_ans(p["it"]["answer"])
        per_inst[s].append(hit); ok += hit
    c = sum(coss) / len(coss)
    print(f"  sigma={s:5.1f}  cos={c:.4f}  install={ok/len(PAIRS):6.1%}", flush=True)

# ---------- 3. the 29%: are failures the same prompts? ----------------------
print("\n=== 3. who fails at the peak, and is it consistent? ===", flush=True)
base = per_inst[0.0]
always_fail = [i for i in range(len(PAIRS)) if not any(per_inst[s][i] for s in SIG)]
ever_pass = len(PAIRS) - len(always_fail)
print(f"  install at sigma=0 (corrected norm): {sum(base)/len(base):.1%}")
print(f"  instances that NEVER pass in any condition: {len(always_fail)}/{len(PAIRS)}")
print(f"  instances that pass somewhere: {ever_pass}/{len(PAIRS)}")
for i in always_fail[:6]:
    print(f"    always-fail: {PAIRS[i]['it']['prompt'][:52]!r} -> "
          f"want {PAIRS[i]['it']['answer']!r}")
import statistics as st
rates = [sum(per_inst[s][i] for s in SIG) / len(SIG) for i in range(len(PAIRS))]
print(f"  per-instance pass rate across conditions: mean {st.mean(rates):.2f} "
      f"sd {st.pstdev(rates):.2f}  (sd near 0 = random; high = prompt-specific)")

# ---------- 2. rank of the delta -------------------------------------------
print("\n=== 2. rank test on the with-file minus no-file delta ===", flush=True)
D = []
for p in PAIRS:
    for l in LAYERS:
        t, k = corrected_target(p, l)
        D.append((t - p["cn"][l][-k:]).cpu())
Dm = torch.cat(D, 0)
U, S, V = torch.linalg.svd(Dm - Dm.mean(0, keepdim=True), full_matrices=False)
ev = (S ** 2) / (S ** 2).sum()
cum = torch.cumsum(ev, 0)
for frac in (0.5, 0.9, 0.95, 0.99):
    print(f"  rank for {frac:.0%} of delta variance: "
          f"{int((cum < frac).sum()) + 1}")
for K in (1, 2, 4, 8, 16, 32, 64, 128):
    Vk = V[:K].to(model.input_device)
    ok = 0
    for p in PAIRS:
        hooks = []
        for l in LAYERS:
            t, k = corrected_target(p, l)
            nat = p["cn"][l][-k:]
            d = t - nat
            hooks.append((l, mk(nat + (d @ Vk.T) @ Vk, l, k)))
        ok += norm_ans(ask(p["pn"], hooks)) == norm_ans(p["it"]["answer"])
    print(f"  rank-{K:4d} truncation -> install {ok/len(PAIRS):6.1%}", flush=True)
json.dump({"model": CONFIGS[args.model]["name"], "skill": args.skill,
           "direction": {str(s): sum(per_inst[s]) / len(PAIRS) for s in SIG},
           "per_instance": {str(s): per_inst[s] for s in SIG},
           "always_fail": always_fail,
           "evr_rank": {f"{f:.2f}": int((cum < f).sum()) + 1
                        for f in (0.5, 0.9, 0.95, 0.99)}},
          open(os.path.join(OUT, f"diag_three_{args.model}.json"), "w"), indent=1)
