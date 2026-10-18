"""Did the patching 'successes' come from copying the SPEC's own activations?

Offset alignment copies the last min(len) positions of the with-file run. The
with-file prompt is much longer, so those positions run back past the shared task
text into the SPEC ITSELF. Token alignment copies only the genuinely shared
suffix. If offset installs and token does not, the earlier ceilings were smuggling
the instructions in as activations -- and the write path cannot use them, because
when the file is removed there are no spec positions to write to.
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
ap.add_argument("--extent", default="band", choices=["band", "all"])
args = ap.parse_args()
NL = CONFIGS[args.model]["n_layers"]
LAYERS = depth_plan(NL)["band"] if args.extent == "band" else list(range(NL))
model, hf, tok, device, dtype = load_model(args.model)
SK = os.path.join(OUT, "skills")
spec = open(os.path.join(SK, args.skill, "skill.md")).read()
inst = json.load(open(os.path.join(SK, args.skill, "instances.json")))[: args.n]
FORMAT = ("\n\nRespond with a single line beginning `ANSWER:` followed by only "
          "the answer, with no explanation.")
chat = lambda t: tok.apply_chat_template([{"role": "user", "content": t}],
                                         tokenize=False, add_generation_prompt=True)


def csl(a, b):
    n = 0
    while n < len(a) and n < len(b) and a[-1 - n] == b[-1 - n]:
        n += 1
    return max(n, 1)


@torch.no_grad()
def cap(t):
    ids = model.encode(t, max_length=512)
    with ActivationRecorder(model.layers, at=LAYERS) as rec:
        model.forward(ids)
    return {l: rec.activations[l][0].float().clone() for l in LAYERS}


def na(s): return re.sub(r"\s+", " ", s.strip()).upper()


@torch.no_grad()
def ask(t, hooks=()):
    ids = model.encode(t, max_length=512)
    with contextlib.ExitStack() as st:
        for l, fn in hooks:
            st.enter_context(intervene(model, l, fn))
        o = hf.generate(ids, max_new_tokens=32, do_sample=False,
                        pad_token_id=tok.eos_token_id)
    g = tok.decode(o[0][ids.shape[1]:], skip_special_tokens=True)
    m = re.findall(r"ANSWER:\s*(.+)", g)
    return re.sub(r"[`*_.]+$", "", (m[-1] if m else g.strip().split("\n")[-1]
                                    if g.strip() else "")).strip()


def mk(src, k):
    def fn(h):
        if h.shape[1] == 1:
            return h
        out = h.clone(); kk = min(k, out.shape[1], src.shape[0])
        out[:, -kk:, :] = src[-kk:].to(h.dtype)
        return out
    return fn


res = {"offset": 0, "token": 0}
lens_w, lens_n, lens_k = [], [], []
samples = []
for it in inst:
    pw, pn = chat(spec + "\n\n---\n\nTask: " + it["prompt"] + FORMAT), chat(it["prompt"] + FORMAT)
    iw = tok(pw, add_special_tokens=False).input_ids
    inn = tok(pn, add_special_tokens=False).input_ids
    k_tok = csl(inn, iw); k_off = min(len(iw), len(inn))
    lens_w.append(len(iw)); lens_n.append(len(inn)); lens_k.append(k_tok)
    cw = cap(pw)
    for name, k in (("offset", k_off), ("token", k_tok)):
        hooks = [(l, mk(cw[l], k)) for l in LAYERS]
        res[name] += na(ask(pn, hooks)) == na(it["answer"])
    if len(samples) < 3:
        extra = iw[-k_off:len(iw) - k_tok] if k_off > k_tok else []
        samples.append(tok.decode(extra)[:90])
n = len(inst)
import statistics as st
print(f"{CONFIGS[args.model]['name']} band L{LAYERS[0]}-{LAYERS[-1]} n={n}")
print(f"  with-file prompt length : {st.mean(lens_w):.1f} tokens")
print(f"  no-file prompt length   : {st.mean(lens_n):.1f} tokens")
print(f"  shared TOKEN suffix     : {st.mean(lens_k):.1f} tokens")
print(f"  offset alignment copies : {st.mean(lens_n):.1f} tokens "
      f"(= {st.mean(lens_n) - st.mean(lens_k):.1f} MORE than are shared)\n")
print(f"  patch with OFFSET alignment -> install {res['offset']/n:6.1%}")
print(f"  patch with TOKEN  alignment -> install {res['token']/n:6.1%}\n")
print("  the extra positions offset alignment copies contain:")
for s in samples:
    print(f"    {s!r}")
if res["offset"] > res["token"]:
    print("\n  -> The patching 'successes' came from copying activations at SPEC")
    print("     positions. Those positions do not exist when the file is removed,")
    print("     so they are unusable by any write path. Every patching ceiling")
    print("     measured with offset alignment is VOID.")
json.dump({"model": CONFIGS[args.model]["name"], "n": n,
           "len_with": st.mean(lens_w), "len_without": st.mean(lens_n),
           "shared_suffix": st.mean(lens_k),
           "install_offset": res["offset"] / n, "install_token": res["token"] / n,
           "extra_text_samples": samples},
          open(os.path.join(OUT, f"diag_align_compare_{args.model}.json"), "w"), indent=1)
