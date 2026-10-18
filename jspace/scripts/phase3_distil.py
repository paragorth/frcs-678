"""PHASE 3 — arms 1 and 2: hard-label FT vs distillation. v2 after cold read.

Match at the OUTPUT, not the input. Teacher runs `[spec + task]`, student runs
`[task]`. The answer positions exist in both, so positional correspondence never
arises (the prefix-geometry wall that ruled out activation patching).

    L_distil = KL(teacher || student) at answer positions, temperature T

WHAT CHANGED AFTER THE DESIGN COLD READ (all of these were fatal):

1. `restore()` did an in-place copy_ on leaf tensors with requires_grad=True ->
   RuntimeError on first call. Nothing had ever run past it. Now under no_grad.
2. **fp32 parameters.** SGD on bf16 weights cannot express these updates: bf16
   has an 8-bit mantissa, ||theta|| ~ 1e2, and the drift targets are 1e-1
   absolute, i.e. ~1e-4 relative -- orders of magnitude below the per-coordinate
   rounding floor. Achieved "drift" would have measured how many coordinates
   crossed a rounding boundary. theta0 is held on CPU so fp32 params + grads fit.
3. **No min-over-runs selection.** The old primary took, per arm, the run with
   the lowest collateral among those clearing an install gate -- which rewards
   the arm that learned least, and handed distillation 12 draws against
   ft_hard's 4. Replaced by an lr LADDER per arm: every arm gets the same number
   of runs, and arms are compared at matched install by INTERPOLATING each arm's
   own curve. No selection over runs anywhere.
4. **Clip is a validity condition, not a counter.** Any cell with clip_hits > 0
   is excluded: once clipping binds, drift = steps*lr*clip independent of the
   loss, and "matched drift" matches nothing about the optimisation.
5. **EOS symmetry.** ft_hard appended eos to its target while distil did not, so
   one arm was trained to stop and the other was not -- and the install gate is
   an exact match on a greedy continuation. The teacher's forced continuation now
   carries eos too.
6. **Teacher accuracy is verified and the run aborts if it is low.** If the
   teacher is wrong, distil trains toward wrong answers while ft_hard trains
   toward right ones, and the arms differ in target correctness, not loss shape.
7. Model defaults to 7B (validated 4/4). Duplicate re-run of the training at the
   chosen lr removed.

THE RULE: the ground-truth label never enters L_distil. Implemented by forcing
the TEACHER'S OWN greedy continuation. Recorded honestly: the teacher is ~100%
accurate, so its greedy tokens usually equal the label, and teacher-forcing that
sequence conditions the student on the correct answer. The rule is honoured in
that no label tensor is a target; it is NOT honoured in the stronger sense that
no label information is present. `distil@T=1` vs `ft_hard` is the built-in null
check -- they should be close, and if they are not, measurement noise exceeds the
effect.
"""
from __future__ import annotations
import argparse, json, os, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn.functional as F
import transformers
from common import CONFIGS, OUT, add_model_arg, depth_plan, lens_path
from restricted import RestrictedLens
from jlens.hooks import ActivationRecorder
import jlens, tokens as toks

ap = argparse.ArgumentParser(); add_model_arg(ap)
ap.set_defaults(model="7b")
ap.add_argument("--tag", default="")
ap.add_argument("--skills", default="beacon,ledger,stamp")
ap.add_argument("--temps", default="1.0,4.0")
ap.add_argument("--lr-ladder", default="3e-6,1e-5,3e-5,1e-4,3e-4,1e-3")
ap.add_argument("--steps", type=int, default=60)
ap.add_argument("--clip", type=float, default=1e9, help="effectively off; any "
                "binding invalidates matched drift")
ap.add_argument("--answer-tokens", type=int, default=16)
ap.add_argument("--teacher-min", type=float, default=0.90)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--seeds", default="", help="replication mode: comma-separated "
                "seeds run at --only-lr only, to estimate the training-run "
                "noise floor the ladder cannot see")
ap.add_argument("--only-lr", type=float, default=None)
args = ap.parse_args()

NAME = CONFIGS[args.model]["name"]
NL = CONFIGS[args.model]["n_layers"]; BAND = depth_plan(NL)["band"]
print(f"loading {NAME} in float32 (bf16 cannot express these updates)", flush=True)
hf = transformers.AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32)
hf.to("cuda").eval()
tok = transformers.AutoTokenizer.from_pretrained(NAME)
model = jlens.from_hf(hf, tok)
lens = RestrictedLens.load(lens_path(args.model, args.tag))
tracked, _ = toks.build(tok, OUT)
SK = os.path.join(OUT, "skills")
FORMAT = ("\n\nRespond with a single line beginning `ANSWER:` followed by only "
          "the answer, with no explanation.")
chat = lambda t: tok.apply_chat_template([{"role": "user", "content": t}],
                                         tokenize=False, add_generation_prompt=True)
norm_a = lambda s: re.sub(r"\s+", " ", s.strip()).upper()


def extract(g):
    m = re.findall(r"ANSWER:\s*(.+)", g)
    return re.sub(r"[`*_.]+$", "",
                  (m[-1] if m else g.strip().split("\n")[-1] if g.strip() else "")).strip()


CORPUS = []
for sk in args.skills.split(","):
    spec = open(os.path.join(SK, sk, "skill.md")).read()
    inst = json.load(open(os.path.join(SK, sk, "instances.json")))
    n = len(inst) // 2
    CORPUS.append({"skill": sk, "spec": spec, "train": inst[:n], "held": inst[n:]})
print(f"skills={[c['skill'] for c in CORPUS]}", flush=True)


@torch.no_grad()
def gen(prompt, k=32):
    ids = tok(chat(prompt), return_tensors="pt").input_ids.to("cuda")
    o = hf.generate(ids, max_new_tokens=k, do_sample=False,
                    pad_token_id=tok.eos_token_id)
    return o[0][ids.shape[1]:], ids.shape[1]


@torch.no_grad()
def install_rate(c, items, use_spec=False):
    ok = 0
    for it in items:
        p = (c["spec"] + "\n\n---\n\nTask: " if use_spec else "") + it["prompt"] + FORMAT
        g, _ = gen(p)
        ok += norm_a(extract(tok.decode(g, skip_special_tokens=True))) == norm_a(it["answer"])
    return ok / max(len(items), 1)


# ---- teacher accuracy gate -------------------------------------------------
print("\nverifying teacher accuracy (aborts if low -- otherwise distil would "
      "train toward wrong answers while ft_hard trains toward right ones)",
      flush=True)
for c in CORPUS:
    c["teacher_acc"] = install_rate(c, c["train"], use_spec=True)
    print(f"  {c['skill']:8s} teacher {c['teacher_acc']:.0%}", flush=True)
worst = min(c["teacher_acc"] for c in CORPUS)
if worst < args.teacher_min:
    sys.exit(f"ABORT: teacher accuracy {worst:.0%} < {args.teacher_min:.0%}; the "
             f"arms would differ in target correctness, not loss shape.")


@torch.no_grad()
def teacher_targets(c, it):
    """Teacher's OWN greedy continuation + its full distribution at each answer
    position. eos appended so both arms are trained to stop."""
    tp = chat(c["spec"] + "\n\n---\n\nTask: " + it["prompt"] + FORMAT)
    ids = tok(tp, return_tensors="pt").input_ids.to("cuda")
    o = hf.generate(ids, max_new_tokens=args.answer_tokens, do_sample=False,
                    pad_token_id=tok.eos_token_id)
    ans = o[0][ids.shape[1]:]
    if ans[-1].item() != tok.eos_token_id:
        ans = torch.cat([ans, torch.tensor([tok.eos_token_id], device=ans.device)])
    full = torch.cat([ids, ans.unsqueeze(0)], 1)
    lg = hf(full).logits[0, ids.shape[1] - 1: -1, :].float()
    return ans.detach().clone(), lg.cpu()


print("\ncaching teacher distributions (frozen, computed once)", flush=True)
for c in CORPUS:
    c["tgt"] = [teacher_targets(c, it) for it in c["train"]]

THETA0 = {n: p.detach().cpu().clone() for n, p in hf.named_parameters()}


@torch.no_grad()
def restore():
    for n, p in hf.named_parameters():
        p.copy_(THETA0[n].to(p.device))


@torch.no_grad()
def drift():
    s = 0.0
    for n, p in hf.named_parameters():
        s += float(((p.detach().cpu() - THETA0[n]) ** 2).sum())
    return s ** 0.5


def loss_distil(c, i, T):
    it = c["train"][i]; ans, tlg = c["tgt"][i]
    ids = tok(chat(it["prompt"] + FORMAT), return_tensors="pt").input_ids.to("cuda")
    full = torch.cat([ids, ans.unsqueeze(0)], 1)
    slg = hf(full).logits[0, ids.shape[1] - 1: -1, :].float()
    t = tlg.to(slg.device)
    return F.kl_div(F.log_softmax(slg / T, -1), F.log_softmax(t / T, -1),
                    log_target=True, reduction="batchmean") * (T * T)


def loss_hard(c, i):
    """The ONLY place a ground-truth label appears."""
    it = c["train"][i]
    pi = tok(chat(it["prompt"] + FORMAT), return_tensors="pt").input_ids.to("cuda")
    ti = tok("ANSWER: " + it["answer"] + tok.eos_token, return_tensors="pt",
             add_special_tokens=False).input_ids.to("cuda")
    ids = torch.cat([pi, ti], 1); lab = ids.clone(); lab[0, :pi.shape[1]] = -100
    return hf(ids, labels=lab).loss


def run(arm, lr, T):
    restore()
    for p in hf.parameters():
        p.requires_grad_(True)
    opt = torch.optim.SGD(hf.parameters(), lr=lr, momentum=0.0)
    rng = __import__("random").Random(args.seed)
    order = [(ci, i) for ci, c in enumerate(CORPUS) for i in range(len(c["train"]))]
    order = (order * (args.steps // len(order) + 1))[: args.steps]
    rng.shuffle(order)
    clips = 0
    for ci, i in order:
        c = CORPUS[ci]
        opt.zero_grad(set_to_none=True)
        (loss_hard(c, i) if arm == "ft_hard" else loss_distil(c, i, T)).backward()
        gn = float(torch.nn.utils.clip_grad_norm_(hf.parameters(), args.clip))
        clips += gn > args.clip
        if gn != gn:
            return float("nan"), clips
        opt.step()
    return drift(), clips


UNREL = [w for w, v in tracked.items() if v["group"] == "control"]
IDX = [lens.index[tracked[w]["id"]] for w in UNREL]
READ = {l: lens.matrix(l).float().cuda()[IDX] for l in BAND}
PROBES = json.load(open(os.path.join(OUT, "sweep_prompts.json")))[:12]


@torch.no_grad()
def probe_state(t):
    ids = tok(t, return_tensors="pt").input_ids.to("cuda")
    with ActivationRecorder(model.layers, at=BAND) as rec:
        model.forward(ids)
    v = torch.cat([rec.activations[l][0, -1].float() @ READ[l].T for l in BAND])
    return v, F.log_softmax(hf(ids).logits[0, -1].float(), -1)


BASE = [probe_state(t) for t in PROBES]
BASE_INSTALL = {c["skill"]: install_rate(c, c["held"]) for c in CORPUS}
print(f"\nbaseline install (no spec): "
      f"{ {k: f'{v:.0%}' for k, v in BASE_INSTALL.items()} }", flush=True)


@torch.no_grad()
def collateral():
    r, k = [], []
    for t, b in zip(PROBES, BASE):
        v, lg = probe_state(t)
        r.append(float((v - b[0]).norm() / (b[0].norm() + 1e-9)))
        k.append(float((b[1].exp() * (b[1] - lg)).sum()))
    return sum(r) / len(r), sum(k) / len(k)


# ---- the ladder: EVERY arm gets the SAME lr grid, no selection over runs ----
ARMS = [("ft_hard", None)] + [("distil", float(T)) for T in args.temps.split(",")]
LADDER = [float(x) for x in args.lr_ladder.split(",")]

if args.seeds:
    # REPLICATION MODE. Every number in the ladder's primary table is a linear
    # function of install at ONE rung, and that rung was measured once. This
    # samples the variance component the ladder sets to zero: training-run to
    # training-run spread. Until that floor is known, no arm comparison is
    # readable -- which is what the pre-registered null check was for.
    lr = args.only_lr or 3e-5
    print(f"\nREPLICATION at lr={lr:.1e}, seeds={args.seeds}", flush=True)
    reps = []
    for sd in [int(x) for x in args.seeds.split(",")]:
        args.seed = sd
        for arm, T in ARMS:
            label = arm + (f"@T{T:g}" if T else "")
            d, clips = run(arm, lr, T)
            inst = {c["skill"]: install_rate(c, c["held"]) for c in CORPUS}
            mi = sum(inst.values()) / len(inst)
            rep_, kl = collateral()
            reps.append({"seed": sd, "arm": label, "lr": lr, "drift": d,
                         "install": inst, "install_mean": mi,
                         "repr_collateral": rep_, "dist_kl": kl})
            print(f"  seed={sd} {label:11s} drift={d:.4f} install={mi:6.1%} "
                  f"coll={rep_:.4f} KL={kl:.4f}", flush=True)
    restore()
    import statistics as st
    print("\n=== NOISE FLOOR ===")
    for arm, T in ARMS:
        label = arm + (f"@T{T:g}" if T else "")
        v = [r["install_mean"] for r in reps if r["arm"] == label]
        if len(v) > 1:
            print(f"  {label:11s} install mean {st.mean(v):.3f} sd {st.pstdev(v):.3f} "
                  f"range {min(v):.3f}-{max(v):.3f}  (n={len(v)} seeds)")
    a = [r["install_mean"] for r in reps if r["arm"] == "ft_hard"]
    b = [r["install_mean"] for r in reps if r["arm"] == "distil@T1"]
    if len(a) > 1 and len(b) > 1:
        gap = st.mean(a) - st.mean(b)
        pooled = ((st.pstdev(a) ** 2 + st.pstdev(b) ** 2) / 2) ** 0.5
        print(f"\n  ft_hard - distil@T1 = {gap:+.3f}  pooled sd {pooled:.3f}  "
              f"=> {gap/pooled if pooled else float('inf'):.2f} sd")
        print("  The pre-reg says these should be CLOSE. If the gap is inside the")
        print("  noise floor, the ladder's primary table is not interpretable.")
    json.dump({"model": NAME, "lr": lr, "reps": reps},
              open(os.path.join(OUT, f"phase3_replication_{args.model}.json"), "w"),
              indent=1)
    sys.exit(0)
results, t0 = [], time.time()
for arm, T in ARMS:
    label = arm + (f"@T{T:g}" if T else "")
    print(f"\n=== {label} ===", flush=True)
    for lr in LADDER:
        d, clips = run(arm, lr, T)
        if d != d:
            print(f"  lr={lr:.1e}  DIVERGED"); continue
        inst = {c["skill"]: install_rate(c, c["held"]) for c in CORPUS}
        mi = sum(inst.values()) / len(inst)
        rep, kl = collateral()
        results.append({"arm": label, "base_arm": arm, "T": T, "lr": lr,
                        "drift": d, "clip_hits": clips, "valid": clips == 0,
                        "install": inst, "install_mean": mi,
                        "repr_collateral": rep, "dist_kl": kl})
        print(f"  lr={lr:.1e} drift={d:.4f} install={mi:5.1%} "
              f"coll={rep:.4f} KL={kl:.4f}"
              f"{'  INVALID(clipped)' if clips else ''}", flush=True)
restore()


def interp(rows, key, target):
    """Value of `key` where install_mean crosses `target`, on this arm's own
    curve. No selection over runs."""
    rs = sorted([r for r in rows if r["valid"]], key=lambda r: r["install_mean"])
    if not rs or rs[0]["install_mean"] > target or rs[-1]["install_mean"] < target:
        return None
    for a, b in zip(rs, rs[1:]):
        if a["install_mean"] <= target <= b["install_mean"]:
            span = b["install_mean"] - a["install_mean"]
            w = 0.0 if span < 1e-9 else (target - a["install_mean"]) / span
            return a[key] + w * (b[key] - a[key])
    return None


print("\n=== PRIMARY: drift and collateral AT MATCHED INSTALL (interpolated) ===")
prim = {}
for lvl in (0.25, 0.50, 0.75):
    row = {}
    for arm, T in ARMS:
        label = arm + (f"@T{T:g}" if T else "")
        rows = [r for r in results if r["arm"] == label]
        d = interp(rows, "drift", lvl); c = interp(rows, "repr_collateral", lvl)
        k = interp(rows, "dist_kl", lvl)
        if d is not None:
            row[label] = {"drift": d, "collateral": c, "kl": k}
    prim[lvl] = row
    if len(row) >= 2:
        o = sorted(row, key=lambda m: row[m]["drift"])
        print(f"  install {lvl:.0%}: drift  " + " < ".join(
            f"{m}({row[m]['drift']:.4f})" for m in o))
        o2 = sorted(row, key=lambda m: row[m]["collateral"])
        print(f"             collateral  " + " < ".join(
            f"{m}({row[m]['collateral']:.4f})" for m in o2))
    else:
        print(f"  install {lvl:.0%}: only {len(row)} arm(s) reach it — no comparison")
print("\nNULL CHECK: distil@T1 vs ft_hard should be CLOSE (teacher is ~one-hot at "
      "T=1). If they separate, measurement noise exceeds the effect.")
json.dump({"model": NAME, "skills": [c["skill"] for c in CORPUS],
           "teacher_acc": {c["skill"]: c["teacher_acc"] for c in CORPUS},
           "baseline_install": BASE_INSTALL, "ladder": LADDER,
           "results": results, "primary_matched_install": prim,
           "wall": time.time() - t0},
          open(os.path.join(OUT, f"phase3_distil_{args.model}.json"), "w"), indent=1)
print(f"\n-> phase3_distil_{args.model}.json ({(time.time()-t0)/60:.1f} min)")
