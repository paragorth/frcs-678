"""Where does the skill-in-context state difference actually live?

L_orth is anchored per-instance to init, and init IS the no-skill state. So the
objective reads: move the J-space component toward the skill state, hold the
orthogonal complement at the no-skill state.

If the with/without difference lives mostly OUTSIDE J-space, that objective is
self-defeating by construction -- it would produce "target reached, nothing
installed" no matter what the model does. That is the R^2=0.9997 error in new
clothes: optimise onto a measured coordinate, hit the coordinate, discover the
coordinate was never sufficient.

Decision rule, fixed before the number is seen:
  frac_in_J large   -> J-space carries the skill difference; the negative stands
  frac_in_J small   -> the negative is VOID; L_orth must be re-anchored to
                       unrelated HELD-OUT concepts, since confinement means
                       "do not disturb other meanings", not "do not change the
                       rest of this one".
"""
from __future__ import annotations
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from common import CONFIGS, OUT, add_model_arg, depth_plan, lens_path, load_model
from restricted import RestrictedLens
from jlens.hooks import ActivationRecorder

ap = argparse.ArgumentParser(); add_model_arg(ap)
ap.add_argument("--tag", default="")
ap.add_argument("--skill", default="beacon")
args = ap.parse_args()

NL = CONFIGS[args.model]["n_layers"]; plan = depth_plan(NL); INJ = plan["inject_layer"]
model, hf, tok, device, dtype = load_model(args.model)
lens = RestrictedLens.load(lens_path(args.model, args.tag))
SK = os.path.join(OUT, "skills")
spec = open(os.path.join(SK, args.skill, "skill.md")).read()
inst = json.load(open(os.path.join(SK, args.skill, "instances.json")))
FORMAT = ("\n\nRespond with a single line beginning `ANSWER:` followed by only "
          "the answer, with no explanation.")

R = lens.matrix(INJ).float().to(model.input_device)
U_, S_, _ = torch.linalg.svd(R.T, full_matrices=False)
RANK = int((S_ > S_[0] * 1e-3).sum())
Q = U_[:, :RANK].detach()
D = Q.shape[0]
print(f"{CONFIGS[args.model]['name']}  L{INJ}  J-space rank {RANK} of {D} dims")


def chat(t):
    return tok.apply_chat_template([{"role": "user", "content": t}],
                                   tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def act(text):
    ids = model.encode(text, max_length=512)
    with ActivationRecorder(model.layers, at=[INJ]) as rec:
        model.forward(ids)
    return rec.activations[INJ][0, -1].float()


PJ = lambda h: (h @ Q) @ Q.T
PO = lambda h: h - PJ(h)

rows = []
for it in inst[:12]:
    hw = act(chat(spec + "\n\n---\n\nTask: " + it["prompt"] + FORMAT))
    hn = act(chat(it["prompt"] + FORMAT))
    d = hw - hn
    dj, do = float(PJ(d).norm()), float(PO(d).norm())
    rows.append({"dJ": dj, "dOrth": do, "frac_in_J": dj ** 2 / (dj ** 2 + do ** 2),
                 "relJ": dj / (float(PJ(hn).norm()) + 1e-9),
                 "relOrth": do / (float(PO(hn).norm()) + 1e-9)})

import statistics as st
mj = st.mean(r["dJ"] for r in rows); mo = st.mean(r["dOrth"] for r in rows)
fr = st.mean(r["frac_in_J"] for r in rows)
print(f"\nwith-skill vs no-skill state difference, n={len(rows)} instances:")
print(f"  ||delta|| in J-space              {mj:8.4f}   "
      f"({100*st.mean(r['relJ'] for r in rows):.1f}% of the J-space state)")
print(f"  ||delta|| in orthogonal complement {mo:8.4f}   "
      f"({100*st.mean(r['relOrth'] for r in rows):.1f}% of the complement state)")
print(f"  fraction of squared difference inside J-space: {100*fr:.2f}%")
print(f"  dims: J {RANK} ({100*RANK/D:.1f}% of {D}); complement {D-RANK}")
print(f"  variance-normalised: J-space holds {100*fr:.2f}% of the difference "
      f"in {100*RANK/D:.1f}% of the dimensions "
      f"-> {fr/(RANK/D):.2f}x enrichment")
VOID = fr < 0.25
print(f"\nVERDICT: {'NEGATIVE IS VOID — L_orth is holding back most of the '
      'skill difference; re-anchor it to unrelated held-out concepts' if VOID else
      'J-space carries a large share of the difference; the negative survives '
      'this check'}")
json.dump({"model": CONFIGS[args.model]["name"], "skill": args.skill,
           "rank": RANK, "d_model": D, "mean_dJ": mj, "mean_dOrth": mo,
           "frac_in_J": fr, "enrichment": fr / (RANK / D), "void": bool(VOID),
           "rows": rows},
          open(os.path.join(OUT, f"diag_orth_{args.model}.json"), "w"), indent=1)
