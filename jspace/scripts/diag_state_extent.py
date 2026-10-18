"""How much of the activation state is needed to transmit the skill?

Single-site capture (layer INJ, final prompt position) transmits topic but not
procedure -- 0% install even when the FULL activation is patched. So the target
is under-specified. This brackets how much state is required, by escalating what
gets copied from the with-file run into the no-file run:

  L_inj_last     one layer, final position          (the Phase 3 target's site)
  L_inj_all      one layer, ALL prompt positions
  band_last      the whole readout band, final position
  band_all       the whole readout band, all positions
  all_last       every layer, final position
  all_all        every layer, all positions

Alignment problem, and why it constrains the design: the with-file prompt is
longer than the no-file prompt, so positions do not correspond. Only the FINAL
position is unambiguously alignable. The all-position variants therefore copy
the last-K positions, K = the shorter length, which is an approximation and is
labelled as one.

Read the result as a lower bound on the target's required richness. If nothing
short of `all_all` installs the skill, activation capture is not a practical
write path and the distillation-style successor needs a different handle.
"""
from __future__ import annotations
import argparse, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from common import CONFIGS, OUT, add_model_arg, depth_plan, load_model
from jlens.hooks import ActivationRecorder

ap = argparse.ArgumentParser(); add_model_arg(ap)
ap.add_argument("--skill", default="beacon"); ap.add_argument("--n", type=int, default=24)
args = ap.parse_args()
NL = CONFIGS[args.model]["n_layers"]; plan = depth_plan(NL)
INJ, BAND = plan["inject_layer"], plan["band"]
model, hf, tok, device, dtype = load_model(args.model)
SK = os.path.join(OUT, "skills")
spec = open(os.path.join(SK, args.skill, "skill.md")).read()
inst = json.load(open(os.path.join(SK, args.skill, "instances.json")))[: args.n]
FORMAT = ("\n\nRespond with a single line beginning `ANSWER:` followed by only "
          "the answer, with no explanation.")
chat = lambda t: tok.apply_chat_template([{"role": "user", "content": t}],
                                         tokenize=False, add_generation_prompt=True)
ALL = list(range(NL))
CONDS = {"L_inj_last": ([INJ], False, False), "L_inj_all": ([INJ], True, False),
         "band_last": (BAND, False, False), "band_all": (BAND, True, False),
         "all_last": (ALL, False, False), "all_all": (ALL, True, False),
         # J-space-restricted: the ceiling for any J-space-targeted update
         "Jonly_inj_all": ([INJ], True, True),
         "Jonly_band_all": (BAND, True, True),
         "Jonly_all_all": (ALL, True, True)}
import torch as _t
from restricted import RestrictedLens as _RL
from common import lens_path as _lp
_lens = _RL.load(_lp(args.model, ""))
_PJ = {}
for _l in set(ALL):
    try:
        _R = _lens.matrix(_l).float().to(model.input_device)
    except KeyError:
        continue
    _U, _S, _ = _t.linalg.svd(_R.T, full_matrices=False)
    _k = int((_S > _S[0] * 1e-3).sum())
    _PJ[_l] = _U[:, :_k].detach()
print(f"J-space projectors available at layers: {sorted(_PJ)}", flush=True)


@torch.no_grad()
def capture(text, layers):
    ids = model.encode(text, max_length=512)
    with ActivationRecorder(model.layers, at=layers) as rec:
        model.forward(ids)
    return {l: rec.activations[l][0].float().clone() for l in layers}


def make_hook(cap, l, allpos, jonly=False):
    Qm = _PJ.get(l)
    def fn(h):
        if h.shape[1] == 1:
            return h                       # generation runs free
        out = h.clone(); src = cap[l]
        k = min(out.shape[1], src.shape[0]) if allpos else 1
        sl = slice(-k, None)
        tgt = src[-k:].to(h.dtype) if allpos else src[-1:].to(h.dtype)
        if jonly and Qm is not None:
            cur = out[:, sl, :].float()
            pj = lambda x: (x @ Qm) @ Qm.T
            out[:, sl, :] = (cur - pj(cur) + pj(tgt.float().unsqueeze(0))).to(h.dtype)
        else:
            out[:, sl, :] = tgt
        return out
    return fn


def norm(s): return re.sub(r"\s+", " ", s.strip()).upper()


@torch.no_grad()
def ask(text, cap=None, layers=(), allpos=False, jonly=False):
    import contextlib
    from lens_ops import intervene
    ids = model.encode(text, max_length=512)
    with contextlib.ExitStack() as st:
        for l in layers:
            st.enter_context(intervene(model, l, make_hook(cap, l, allpos, jonly)))
        o = hf.generate(ids, max_new_tokens=32, do_sample=False,
                        pad_token_id=tok.eos_token_id)
    g = tok.decode(o[0][ids.shape[1]:], skip_special_tokens=True)
    m = re.findall(r"ANSWER:\s*(.+)", g)
    return re.sub(r"[`*_.]+$", "", (m[-1] if m else g.strip().split("\n")[-1]
                                    if g.strip() else "")).strip()


res = {k: 0 for k in list(CONDS) + ["no_patch", "with_file"]}
samples = {k: [] for k in CONDS}
for it in inst:
    pw, pn = chat(spec + "\n\n---\n\nTask: " + it["prompt"] + FORMAT), chat(it["prompt"] + FORMAT)
    res["no_patch"] += norm(ask(pn)) == norm(it["answer"])
    res["with_file"] += norm(ask(pw)) == norm(it["answer"])
    for name, (layers, allpos, jonly) in CONDS.items():
        if jonly:
            layers = [l for l in layers if l in _PJ]
        cap = capture(pw, layers)
        a = ask(pn, cap, layers, allpos, jonly)
        res[name] += norm(a) == norm(it["answer"])
        if len(samples[name]) < 2:
            samples[name].append(a[:44])
n = len(inst)
print(f"\nSTATE EXTENT, {args.skill}, n={n}  (layers={NL}, band={BAND[0]}-{BAND[-1]})")
print(f"  {'no_patch':14s} {res['no_patch']/n:6.1%}")
for k in CONDS:
    print(f"  {k:14s} {res[k]/n:6.1%}   e.g. {samples[k]}")
print(f"  {'with_file':14s} {res['with_file']/n:6.1%}")
best = max(CONDS, key=lambda k: res[k])
print(f"\nrichest transmitting condition: {best} at {res[best]/n:.1%}")
if res[best] / n < 0.10:
    print("  -> NO activation-copy condition installs the skill, up to copying")
    print("     every layer at every position. Activation capture at inference")
    print("     does not transmit this procedure; the Phase 3 target is not")
    print("     merely too narrow, the whole single-shot capture framing fails.")
json.dump({"model": CONFIGS[args.model]["name"], "skill": args.skill, "n": n,
           "rates": {k: v / n for k, v in res.items()}, "samples": samples},
          open(os.path.join(OUT, f"diag_state_extent_{args.model}.json"), "w"), indent=1)
