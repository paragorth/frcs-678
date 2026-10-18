"""Job 4: Phase 3 pilot -- is a workspace-targeted install cleaner than
matched-drift fine-tuning?

NOTE ON PROVENANCE: `phase3-confinement-brief.md` was not present on this
machine. This implements the pilot from the inline specification only (one
skill, three conditions, one magnitude plus a matched-drift FT baseline,
raw J-space target, full-rank, three collateral levels). Any detail of the
original brief not restated inline is NOT implemented, because it was never
seen. Read the numbers with that caveat attached.

Conditions
  targeted_penalty_on   install + confinement penalty on off-target J-space drift
  targeted_penalty_off  install, no penalty
  random_null           same machinery aimed at a random direction (not the skill's)
  ft_matched_drift      ordinary fine-tuning, LR bisected to match ||dtheta||

Full-rank throughout. No LoRA: low-rank adaptation is itself a forgetting
mitigation and would confound the collateral measurement, which is the whole
question.

Collateral, all three levels:
  behavioural      pass rate on the OTHER kept skills
  representational J-space readout drift on unrelated held-out concepts
  distributional   KL of next-token distribution on held-out neutral text
"""
from __future__ import annotations
import argparse, copy, json, os, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn.functional as F
from common import CONFIGS, OUT, add_model_arg, depth_plan, lens_path, load_model
from restricted import RestrictedLens
from jlens.hooks import ActivationRecorder
import tokens as toks

ap = argparse.ArgumentParser()
add_model_arg(ap)
ap.add_argument("--tag", default="")
ap.add_argument("--skill", default="beacon")
ap.add_argument("--steps", type=int, default=60)
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--penalty", type=float, default=1.0)
ap.add_argument("--train-frac", type=float, default=0.5)
ap.add_argument("--drift-tol", type=float, default=0.10)
ap.add_argument("--magnitudes", default="0.05,0.1,0.2,0.4,0.8",
                help="||dtheta|| targets; the finding is a slope, not a point")
args = ap.parse_args()

NL = CONFIGS[args.model]["n_layers"]
plan = depth_plan(NL); INJECT, BAND = plan["inject_layer"], plan["band"]
model, hf, tok, device, dtype = load_model(args.model)
lens = RestrictedLens.load(lens_path(args.model, args.tag))
tracked, seed_only = toks.build(tok, OUT)

SK = os.path.join(OUT, "skills")
summary = json.load(open(os.path.join(SK, f"gap_summary_{args.model}.json")))
KEPT = summary["kept"]
skill = args.skill if args.skill in KEPT else summary["widest"]
spec = open(os.path.join(SK, skill, "skill.md")).read()
inst = json.load(open(os.path.join(SK, skill, "instances.json")))
ntr = int(len(inst) * args.train_frac)
train, heldout = inst[:ntr], inst[ntr:]
print(f"skill={skill}  train={len(train)}  held-out={len(heldout)}  "
      f"other kept skills={[s for s in KEPT if s != skill]}", flush=True)

FORMAT = ("\n\nRespond with a single line beginning `ANSWER:` followed by only "
          "the answer, with no explanation.")


def norm(s): return re.sub(r"\s+", " ", s.strip()).upper()


def extract(t):
    m = re.findall(r"ANSWER:\s*(.+)", t)
    return re.sub(r"[`*_.]+$", "", (m[-1] if m else t.strip().split("\n")[-1] if t.strip() else "")).strip()


@torch.no_grad()
def pass_rate(items, use_spec=False, sp=None):
    ok = 0
    for it in items:
        p = ((sp or spec) + "\n\n---\n\nTask: " if use_spec else "") + it["prompt"] + FORMAT
        msgs = [{"role": "user", "content": p}]
        txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(txt, return_tensors="pt").input_ids.to(model.input_device)
        out = hf.generate(ids, max_new_tokens=32, do_sample=False,
                          pad_token_id=tok.eos_token_id)
        ok += norm(extract(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True))) == norm(it["answer"])
    return ok / max(len(items), 1)


def train_batch(it):
    """Loss on the answer tokens only, with NO skill file in context -- the
    install has to live in the weights, not the prompt."""
    p = it["prompt"] + FORMAT
    msgs = [{"role": "user", "content": p}]
    txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    tgt = "ANSWER: " + it["answer"] + tok.eos_token
    pi = tok(txt, return_tensors="pt").input_ids.to(model.input_device)
    ti = tok(tgt, return_tensors="pt", add_special_tokens=False).input_ids.to(model.input_device)
    ids = torch.cat([pi, ti], 1)
    labels = ids.clone(); labels[0, :pi.shape[1]] = -100
    return ids, labels


# ---- the skill's raw J-space target ---------------------------------------
@torch.no_grad()
def workspace_signature(items, n=8):
    """Mean lens-space residual difference (with skill file) - (without), at the
    injection layer. This is the raw J-space vector the install targets; it is
    deliberately NOT a token readout."""
    acc = None
    for it in items[:n]:
        vs = []
        for use in (True, False):
            p = (spec + "\n\n---\n\nTask: " if use else "") + it["prompt"] + FORMAT
            ids = model.encode(p, max_length=512)
            with ActivationRecorder(model.layers, at=[INJECT]) as rec:
                model.forward(ids)
            vs.append(rec.activations[INJECT][0, -1].float())
        d = vs[0] - vs[1]
        acc = d if acc is None else acc + d
    v = acc / min(n, len(items))
    return v / v.norm()


TARGET = workspace_signature(train)
g = torch.Generator(device="cpu").manual_seed(20260820)
RAND = torch.randn(lens.d_model, generator=g).to(TARGET.device, TARGET.dtype)
RAND = RAND - (RAND @ TARGET) * TARGET
RAND = RAND / RAND.norm()
print(f"J-space target ready (|v|={TARGET.norm():.3f}); "
      f"random null orthogonalised (cos={float(TARGET@RAND):+.2e})", flush=True)

# unrelated held-out concepts for the representational measure
UNREL = [w for w, v in tracked.items() if v["group"] == "control"][:10]
UNREL_IDS = [tracked[w]["id"] for w in UNREL]


@torch.no_grad()
def jspace_readouts(probe_texts):
    """Lens readouts on unrelated concepts -- the representational collateral."""
    out = []
    for t in probe_texts:
        ids = model.encode(t, max_length=64)
        with ActivationRecorder(model.layers, at=BAND) as rec:
            model.forward(ids)
        v = torch.cat([(rec.activations[l][0, -1].float() @
                        lens.matrix(l).float().to(rec.activations[l].device).T)
                       [[lens.index[i] for i in UNREL_IDS]] for l in BAND])
        out.append(v)
    return torch.stack(out)


@torch.no_grad()
def next_token_dist(texts):
    return [F.log_softmax(hf(model.encode(t, max_length=64)).logits[0, -1].float(), -1)
            for t in texts]


PROBES = json.load(open(os.path.join(OUT, "sweep_prompts.json")))[:12]
BASE_READ = jspace_readouts(PROBES)
BASE_DIST = next_token_dist(PROBES)
BASE_OTHER = {s: pass_rate(json.load(open(os.path.join(SK, s, "instances.json")))[:8],
                           use_spec=True,
                           sp=open(os.path.join(SK, s, "skill.md")).read())
              for s in KEPT if s != skill}
BASE_SKILL_NOFILE = pass_rate(heldout, use_spec=False)
print(f"baseline: {skill} without file = {BASE_SKILL_NOFILE:.1%}; "
      f"other skills (with file) = "
      f"{ {k: f'{v:.0%}' for k, v in BASE_OTHER.items()} }", flush=True)

THETA0 = {n: p.detach().clone() for n, p in hf.named_parameters()}


def drift():
    return torch.sqrt(sum(((p.detach() - THETA0[n]) ** 2).sum()
                          for n, p in hf.named_parameters())).item()


def restore():
    with torch.no_grad():
        for n, p in hf.named_parameters():
            p.copy_(THETA0[n])


def run_condition(name, lr, targeted, penalty, direction):
    restore()
    for p in hf.parameters():
        p.requires_grad_(True)
    opt = torch.optim.SGD(hf.parameters(), lr=lr, momentum=0.0)
    t0 = time.time()
    for step in range(args.steps):
        it = train[step % len(train)]
        ids, labels = train_batch(it)
        if targeted:
            with ActivationRecorder(model.layers, at=[INJECT]) as rec:
                out = hf(ids, labels=labels)
                h = rec.activations[INJECT][0, -1].float()
            align = -(h @ direction) / (h.norm() + 1e-6)
            loss = out.loss + align
            if penalty > 0:
                # confinement: hold unrelated J-space readouts where they were
                cur = jspace_readouts_grad(PROBES[:2])
                loss = loss + penalty * F.mse_loss(cur, BASE_READ[:2])
        else:
            loss = hf(ids, labels=labels).loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(hf.parameters(), 1.0)
        opt.step()
    d = drift()
    res = {"condition": name, "lr": lr, "steps": args.steps, "drift": d,
           "wall": time.time() - t0}
    res["skill_nofile"] = pass_rate(heldout, use_spec=False)
    res["other_skills"] = {s: pass_rate(
        json.load(open(os.path.join(SK, s, "instances.json")))[:8], use_spec=True,
        sp=open(os.path.join(SK, s, "skill.md")).read()) for s in BASE_OTHER}
    cur = jspace_readouts(PROBES)
    res["repr_drift"] = float((cur - BASE_READ).norm() / (BASE_READ.norm() + 1e-9))
    cur_d = next_token_dist(PROBES)
    res["dist_kl"] = float(sum(float((b.exp() * (b - c)).sum())
                               for b, c in zip(BASE_DIST, cur_d)) / len(PROBES))
    print(f"  {name:22s} drift={d:9.4f} skill={res['skill_nofile']:5.1%} "
          f"repr={res['repr_drift']:.4f} KL={res['dist_kl']:.4f}", flush=True)
    return res


def jspace_readouts_grad(texts):
    out = []
    for t in texts:
        ids = model.encode(t, max_length=64)
        with ActivationRecorder(model.layers, at=BAND) as rec:
            model.forward(ids)
        v = torch.cat([(rec.activations[l][0, -1].float() @
                        lens.matrix(l).float().to(rec.activations[l].device).T)
                       [[lens.index[i] for i in UNREL_IDS]] for l in BAND])
        out.append(v)
    return torch.stack(out)


# ---------------------------------------------------------------- experiment
# THREE FIXES over the withdrawn pilot (see log 20a):
#
# 1. EQUAL SEARCH BUDGET. Previously only the FT baseline was LR-bisected; the
#    targeted arm got one hand-computed LR. A searched arm beating an unsearched
#    one says nothing about the methods. Now EVERY arm is bisected to the same
#    drift target with the same iteration budget.
# 2. MATCHED INSTALL, not just matched drift. Comparing collateral at matched
#    drift but 75% vs 100% install is not a comparison -- the missing 25% has
#    unknown cost. Sweeping magnitude yields install-vs-collateral curves, so
#    arms can be read off at equal install as well as equal drift.
# 3. MAGNITUDE SWEEP. The quantity of interest is a slope, not a point. One
#    operating point cannot separate "confinement does not work" from
#    "confinement does not work here".

ARMS = [("targeted_penalty_on", True, args.penalty, "TARGET"),
        ("targeted_penalty_off", True, 0.0, "TARGET"),
        ("random_null", True, 0.0, "RAND"),
        ("ft_matched_drift", False, 0.0, None)]
DIRS = {"TARGET": TARGET, "RAND": RAND}
MAGNITUDES = [float(x) for x in args.magnitudes.split(",")]


def bisect_to_drift(target_d, targeted, penalty, direction, iters=10):
    """Same search budget for every arm. Returns (lr, achieved_drift)."""
    lo, hi = 1e-5, 5.0
    best = (None, None)
    for _ in range(iters):
        mid = (lo * hi) ** 0.5
        d = train_once(mid, targeted, penalty, direction, measure_only=True)
        if best[0] is None or abs(d - target_d) < abs(best[1] - target_d):
            best = (mid, d)
        if abs(d - target_d) / target_d <= args.drift_tol:
            return mid, d
        if d < target_d:
            lo = mid
        else:
            hi = mid
    return best


def train_once(lr, targeted, penalty, direction, measure_only=False):
    restore()
    for prm in hf.parameters():
        prm.requires_grad_(True)
    opt = torch.optim.SGD(hf.parameters(), lr=lr, momentum=0.0)
    for step in range(args.steps):
        it = train[step % len(train)]
        ids, labels = train_batch(it)
        if targeted:
            with ActivationRecorder(model.layers, at=[INJECT]) as rec:
                o = hf(ids, labels=labels)
                h = rec.activations[INJECT][0, -1].float()
            loss = o.loss - (h @ direction) / (h.norm() + 1e-6)
            if penalty > 0:
                loss = loss + penalty * F.mse_loss(
                    jspace_readouts_grad(PROBES[:2]), BASE_READ[:2])
        else:
            loss = hf(ids, labels=labels).loss
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(hf.parameters(), 1.0); opt.step()
    return drift()


results = []
for mag in MAGNITUDES:
    print(f"\n=== magnitude ||dtheta|| target = {mag} ===", flush=True)
    for name, targeted, pen, dkey in ARMS:
        direction = DIRS.get(dkey)
        lr, achieved = bisect_to_drift(mag, targeted, pen, direction)
        if lr is None:
            print(f"  {name:22s} BISECTION FAILED"); continue
        rel = abs(achieved - mag) / mag
        train_once(lr, targeted, pen, direction)      # leave weights at this point
        r = {"condition": name, "magnitude": mag, "lr": lr, "drift": drift(),
             "drift_rel_err": rel, "drift_matched": rel <= args.drift_tol}
        r["skill_nofile"] = pass_rate(heldout, use_spec=False)
        r["other_skills"] = {sname: pass_rate(
            json.load(open(os.path.join(SK, sname, "instances.json")))[:8],
            use_spec=True, sp=open(os.path.join(SK, sname, "skill.md")).read())
            for sname in BASE_OTHER}
        cur = jspace_readouts(PROBES)
        r["repr_drift"] = float((cur - BASE_READ).norm() / (BASE_READ.norm() + 1e-9))
        cd = next_token_dist(PROBES)
        r["dist_kl"] = float(sum(float((b.exp() * (b - c)).sum())
                                 for b, c in zip(BASE_DIST, cd)) / len(PROBES))
        results.append(r)
        print(f"  {name:22s} lr={lr:.2e} drift={r['drift']:.4f} "
              f"({'matched' if r['drift_matched'] else f'off by {100*rel:.0f}%'}) "
              f"install={r['skill_nofile']:5.1%} repr={r['repr_drift']:.4f} "
              f"KL={r['dist_kl']:.4f}", flush=True)
restore()

print("\n=== install-vs-collateral (read arms at EQUAL INSTALL, not just equal drift) ===")
for name, _, _, _ in ARMS:
    rs = sorted([r for r in results if r["condition"] == name],
                key=lambda r: r["skill_nofile"])
    if rs:
        print(f"  {name:22s} " + "  ".join(
            f"[inst {r['skill_nofile']:.0%} -> repr {r['repr_drift']:.3f}, "
            f"KL {r['dist_kl']:.3f}, drift {r['drift']:.3f}]" for r in rs))

out = {"model": CONFIGS[args.model]["name"], "skill": skill,
       "baseline": {"skill_nofile": BASE_SKILL_NOFILE, "other": BASE_OTHER},
       "results": results, "arms": [a[0] for a in ARMS],
       "magnitudes": MAGNITUDES, "note": "phase3-confinement-brief.md absent; "
       "implemented from inline spec only"}
json.dump(out, open(os.path.join(OUT, f"phase3_pilot_{args.model}.json"), "w"), indent=1)
print(f"\n-> {os.path.join(OUT, f'phase3_pilot_{args.model}.json')}")
