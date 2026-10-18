"""Phase 3 — the write path from activation to weight.

NOTE: re-running this with strict scoring corrects the SCORING axis
only. Job 4 remains NOT TESTED (log 20a, 25b): asymmetric optimisation
effort, collateral at matched drift with unmatched install, and n=1 on
beacon -- which the four-skill data shows is the weakest skill of the
set. Do not read a strict-scored result here as a rehabilitated
mechanism comparison.


Capture the workspace state the model reaches when the skill file is in context.
Then reach that same state THROUGH WEIGHTS with the file removed, while
penalising or projecting away change in orthogonal directions.

    L = L_target + lambda_orth * L_orth + lambda_size * ||delta_theta||^2

L_target is pure STATE MATCHING in J-space -- not the LM loss on the answer.
That is deliberate and is the actual thesis under test: if writing the workspace
state into weights is a real write path, the *behaviour* should follow without
ever being trained on directly. Behavioural install is therefore a measured
outcome here, not an objective. Ordinary fine-tuning (the baseline) does use the
LM loss, because that is what ordinary fine-tuning is.

Mechanisms compared, all at MATCHED WEIGHT DRIFT across magnitudes:
  soft_penalty     lambda_orth term in the loss
  grad_projection  GEM/OGD lineage -- project the update off the orthogonal
                   reference gradients instead of penalising them
  verify_gate      take the step, measure orthogonal drift, commit or revert
  ft_baseline      ordinary LM fine-tuning, no J-space targeting

No LoRA anywhere: low-rank adaptation is itself a forgetting mitigation and
would confound the measurement.
"""
from __future__ import annotations
import argparse, json, os, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn.functional as F
from common import CONFIGS, OUT, add_model_arg, depth_plan, lens_path, load_model
from restricted import RestrictedLens
from jlens.hooks import ActivationRecorder
import tokens as toks

ap = argparse.ArgumentParser(); add_model_arg(ap)
ap.add_argument("--tag", default="")
ap.add_argument("--skill", default="beacon")
ap.add_argument("--steps", type=int, default=40)
ap.add_argument("--magnitudes", default="0.05,0.1,0.2,0.4")
ap.add_argument("--lambda-orth", type=float, default=1.0)
ap.add_argument("--lambda-size", type=float, default=0.0)
ap.add_argument("--gate-tol", type=float, default=1.5,
                help="verify_gate: reject a step whose orth drift exceeds this "
                     "multiple of its J-space progress")
ap.add_argument("--drift-tol", type=float, default=0.10)
ap.add_argument("--bisect-iters", type=int, default=6)
ap.add_argument("--clip", type=float, default=50.0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--target-space", choices=["jspace", "full"], default="full",
                help="jspace has a MEASURED ceiling of 0% install (patching the\n                      J-space component at every layer and position installs\n                      nothing); full activation reaches 20-58%. A weight update\n                      cannot beat patching the target in directly.")
ap.add_argument("--target-layers", default="band", choices=["inj", "band"])
args = ap.parse_args()

NL = CONFIGS[args.model]["n_layers"]; plan = depth_plan(NL)
INJ, BAND = plan["inject_layer"], plan["band"]
model, hf, tok, device, dtype = load_model(args.model)
lens = RestrictedLens.load(lens_path(args.model, args.tag))
tracked, _ = toks.build(tok, OUT)
SK = os.path.join(OUT, "skills")
summary = json.load(open(os.path.join(SK, f"gap_summary_{args.model}.json")))
skill = args.skill if args.skill in summary["kept"] else summary["widest"]
spec = open(os.path.join(SK, skill, "skill.md")).read()
inst = json.load(open(os.path.join(SK, skill, "instances.json")))
ntr = len(inst) // 2
train, heldout = inst[:ntr], inst[ntr:]
OTHERS = [s for s in summary["kept"] if s != skill]
FORMAT = ("\n\nRespond with a single line beginning `ANSWER:` followed by only "
          "the answer, with no explanation.")
print(f"model={CONFIGS[args.model]['name']} skill={skill} train={len(train)} "
      f"held-out={len(heldout)} others={len(OTHERS)}", flush=True)

# ---- J-space projector: orthonormal basis of the lens row space at INJ -------
# QR on ~160 corpus-averaged, near-collinear unit rows returns columns whose
# tail is numerical noise. Use the SVD rank at a tolerance instead, so the
# reported dimensionality is real and the ablation's random basis is matched to
# the TRUE rank rather than to an inflated one.
R = lens.matrix(INJ).float().to(model.input_device)          # [n_seed, d_model]
U_, S_, _ = torch.linalg.svd(R.T, full_matrices=False)
RANK = int((S_ > S_[0] * 1e-3).sum())
Q = U_[:, :RANK].detach()
print(f"J-space: {R.shape[0]} seed rows -> effective rank {RANK} "
      f"(nominal {min(R.shape)}), of {Q.shape[0]} dims", flush=True)


def P_J(h):
    return (h @ Q) @ Q.T


def P_orth(h):
    return h - P_J(h)


_g = torch.Generator(device="cpu").manual_seed(20260821)
_M = torch.randn(Q.shape[0], RANK, generator=_g).to(Q.device, Q.dtype)
Q_RAND, _ = torch.linalg.qr(_M)
Q_RAND = Q_RAND[:, :RANK].detach()   # matched to the TRUE rank of J-space


def P_rand_orth(h):
    """Complement of a RANDOM subspace of the same dimension as J-space."""
    return h - (h @ Q_RAND) @ Q_RAND.T


def _chat(text):
    """Every forward -- capture, training and evaluation -- goes through the
    chat template. Previously capture/training used the raw tokenizer while
    pass_rate used the template, so the state being written into the weights was
    one the model never occupies at evaluation time. A null under that mismatch
    is uninterpretable."""
    return tok.apply_chat_template([{"role": "user", "content": text}],
                                   tokenize=False, add_generation_prompt=True)


def prompt_with(it):
    return _chat(spec + "\n\n---\n\nTask: " + it["prompt"] + FORMAT)


def prompt_without(it):
    return _chat(it["prompt"] + FORMAT)


@torch.no_grad()
def last_act(text):
    ids = model.encode(text, max_length=512)
    with ActivationRecorder(model.layers, at=[INJ]) as rec:
        model.forward(ids)
    return rec.activations[INJ][0, -1].float().clone()


TGT_LAYERS = [INJ] if args.target_layers == "inj" else BAND


@torch.no_grad()
def seq_act_multi(text):
    ids = model.encode(text, max_length=512)
    with ActivationRecorder(model.layers, at=TGT_LAYERS) as rec:
        model.forward(ids)
    return {l: rec.activations[l][0].float().clone() for l in TGT_LAYERS}


def seq_act_multi_grad(text):
    ids = model.encode(text, max_length=512)
    with ActivationRecorder(model.layers, at=TGT_LAYERS) as rec:
        model.forward(ids)
    return {l: rec.activations[l][0].float() for l in TGT_LAYERS}


def TSP(x):
    """Target-space projection: identity for the full-activation target."""
    return P_J(x) if args.target_space == "jspace" else x


@torch.no_grad()
def seq_act(text):
    """Full position sequence at INJ. Single-position capture installs 0% even
    with the whole activation copied; only multi-position capture transmits the
    procedure (band_all 20.8%, all_all 54.2%)."""
    ids = model.encode(text, max_length=512)
    with ActivationRecorder(model.layers, at=[INJ]) as rec:
        model.forward(ids)
    return rec.activations[INJ][0].float().clone()


def seq_act_grad(text):
    ids = model.encode(text, max_length=512)
    with ActivationRecorder(model.layers, at=[INJ]) as rec:
        model.forward(ids)
    return rec.activations[INJ][0].float()


def _common_suffix_len(ids_a, ids_b):
    """Length of the longest common TOKEN suffix."""
    n = 0
    while n < len(ids_a) and n < len(ids_b) and ids_a[-1 - n] == ids_b[-1 - n]:
        n += 1
    return n


_ALIGN_CACHE = {}


def align_len(text_a, text_b):
    """Number of trailing positions whose TOKENS actually correspond.

    Aligning by position offset was wrong and dominated the whole objective.
    The with-file prompt is [chat_prefix][spec][---][Task: X][FORMAT] and the
    no-file one is [chat_prefix][Task: X][FORMAT]; only the trailing task text is
    shared. Offset alignment therefore compared the SHORT prompt's leading
    control tokens -- which in Qwen carry massive-norm attention-sink
    activations -- against the long prompt's mid-content tokens. Measured
    consequence: norm ratio 29.9 and ||delta|| at 99.9% of ||h|| at theta0, so
    L_target was mostly sink-norm mismatch. The arms were optimising to shrink
    attention sinks, which is why cosine plateaued at 0.54 with 0% install.
    """
    key = (text_a, text_b)
    if key not in _ALIGN_CACHE:
        ia = tok(text_a, add_special_tokens=False).input_ids
        ib = tok(text_b, add_special_tokens=False).input_ids
        _ALIGN_CACHE[key] = max(_common_suffix_len(ia, ib), 1)
    return _ALIGN_CACHE[key]


def align_tail(a, b, k=None):
    """Align on the final k positions; k from token identity, not offset."""
    k = min(k or min(a.shape[0], b.shape[0]), a.shape[0], b.shape[0])
    return a[-k:], b[-k:]


# ---- captured targets: workspace state WITH the file in context -------------
TARGET_J = [{l: TSP(v).detach() for l, v in seq_act_multi(prompt_with(it)).items()}
            for it in train]
TARGET_J_HELD = [{l: TSP(v).detach() for l, v in seq_act_multi(prompt_with(it)).items()}
                 for it in heldout]
print(f"target: {args.target_space} activation, layers {TGT_LAYERS}, all positions",
      flush=True)
# Anchor set: UNRELATED held-out prompts, never the training instances.
# Anchoring to the training instances pinned 94.4% of the skill's own activation
# change to the no-skill state, so "target reached, nothing installed" was
# guaranteed by construction (see out/diag_orth_7b.json).
ANCHOR_TEXTS = [_chat(t) for t in
                json.load(open(os.path.join(OUT, "sweep_prompts.json")))[:12]]
BASE_ORTH = [P_orth(last_act(t)).detach() for t in ANCHOR_TEXTS]
BASE_RAND = [P_rand_orth(last_act(t)).detach() for t in ANCHOR_TEXTS]
print(f"captured {len(TARGET_J)} workspace targets", flush=True)

THETA0 = {n: p.detach().clone() for n, p in hf.named_parameters()}


def restore():
    with torch.no_grad():
        for n, p in hf.named_parameters():
            p.copy_(THETA0[n])


def drift():
    return torch.sqrt(sum(((p.detach() - THETA0[n]) ** 2).sum()
                          for n, p in hf.named_parameters())).item()


def act_nograd_off(text):
    """Activation at INJ with grad, for the training forward."""
    ids = model.encode(text, max_length=512)
    with ActivationRecorder(model.layers, at=[INJ]) as rec:
        model.forward(ids)
    return rec.activations[INJ][0, -1].float()


def losses_for(i, rand=False):
    """L_target on the training instance; L_orth on an UNRELATED anchor prompt.

    Two different inputs on purpose. L_target says "reach the skill state on this
    task"; L_orth says "leave unrelated meanings where they were". Evaluating
    both on the same input is what made the objective self-defeating.
    """
    cur = seq_act_multi_grad(prompt_without(train[i]))
    k = align_len(prompt_without(train[i]), prompt_with(train[i]))
    lt = sum(F.mse_loss(*align_tail(TSP(cur[l]), TARGET_J[i][l], k)) for l in TGT_LAYERS)
    lt = lt / len(TGT_LAYERS)
    j = i % len(ANCHOR_TEXTS)
    ha = act_nograd_off(ANCHOR_TEXTS[j])
    if rand:
        return lt, F.mse_loss(P_rand_orth(ha), BASE_RAND[j])
    return lt, F.mse_loss(P_orth(ha), BASE_ORTH[j])


@torch.no_grad()
def match_quality(items, targets):
    """MSE falls if the activation merely shrinks toward the target's mean,
    without pointing the same way. Cosine and norm ratio separate a real match
    from a degenerate one. Reported alongside L_target because L_target alone
    cannot tell them apart -- and the arms reach L_target ~1.0 while installing
    0%, where patching the same residual installs 54.2%."""
    cos, nr = [], []
    for it, tg in zip(items, targets):
        cur = seq_act_multi(prompt_without(it))
        k = align_len(prompt_without(it), prompt_with(it))
        for l in TGT_LAYERS:
            a, b = align_tail(TSP(cur[l]), tg[l], k)
            cos.append(float(F.cosine_similarity(a.flatten(), b.flatten(), dim=0)))
            nr.append(float(a.norm() / (b.norm() + 1e-9)))
    return sum(cos) / len(cos), sum(nr) / len(nr)


@torch.no_grad()
def ltarget_on(items, targets):
    """Achieved L_target. Reported on TRAIN and HELD-OUT separately: low on
    train + high on held-out means 12 activation vectors were memorised, which
    is the path of least resistance for ~1900 constrained scalars against 1.5e9
    free parameters."""
    tot = 0.0
    for it, tg in zip(items, targets):
        cur = seq_act_multi(prompt_without(it))
        k = align_len(prompt_without(it), prompt_with(it))
        tot += float(sum(F.mse_loss(*align_tail(TSP(cur[l]), tg[l], k))
                         for l in TGT_LAYERS) / len(TGT_LAYERS))
    return tot / max(len(items), 1)


def lm_loss(i):
    it = train[i]
    p = prompt_without(it)
    pi = tok(p, return_tensors="pt").input_ids.to(model.input_device)
    ti = tok("ANSWER: " + it["answer"] + tok.eos_token, return_tensors="pt",
             add_special_tokens=False).input_ids.to(model.input_device)
    ids = torch.cat([pi, ti], 1); lab = ids.clone(); lab[0, :pi.shape[1]] = -100
    return hf(ids, labels=lab).loss


def grad_snapshot():
    """Per-parameter copy of .grad. Flattening into one vector allocated 14 GiB
    per copy and OOM'd an 80 GB card on top of weights + grads + theta0."""
    return [(p.grad.detach().clone() if p.grad is not None else None)
            for p in hf.parameters()]


def project_grad_off(ref):
    """g <- g - (<g,ref>/||ref||^2) ref, computed without materialising either
    as a flat vector."""
    dot = 0.0; nrm = 0.0
    for p, r in zip(hf.parameters(), ref):
        if r is None or p.grad is None:
            continue
        dot += float(torch.sum(p.grad.detach() * r))
        nrm += float(torch.sum(r * r))
    if nrm <= 1e-12:
        return
    c = dot / nrm
    with torch.no_grad():
        for p, r in zip(hf.parameters(), ref):
            if r is not None and p.grad is not None:
                p.grad.add_(r, alpha=-c)


def run(mech, lr):
    """One training run. Returns achieved drift."""
    restore()
    for p in hf.parameters():
        p.requires_grad_(True)
    opt = torch.optim.SGD(hf.parameters(), lr=lr, momentum=0.0)
    n_rejected = 0; clip_hits = 0
    order = list(range(len(train))) * (args.steps // len(train) + 1)
    rng_local = __import__("random").Random(args.seed)
    rng_local.shuffle(order)
    for step in range(args.steps):
        i = order[step]
        opt.zero_grad(set_to_none=True)
        if mech == "ft_baseline":
            lm_loss(i).backward()
        elif mech in ("soft_penalty", "random_subspace"):
            lt, lo = losses_for(i, rand=(mech == "random_subspace"))
            (lt + args.lambda_orth * lo).backward()
        elif mech == "grad_projection":
            # GEM/OGD: gradient of the orthogonal-preservation term is the
            # reference direction; project the target gradient off it rather
            # than trading the two off in a weighted sum.
            lt, lo = losses_for(i)
            lo.backward(retain_graph=True)
            g_ref = grad_snapshot()
            opt.zero_grad(set_to_none=True)
            lt.backward()
            project_grad_off(g_ref)
            del g_ref
        elif mech == "verify_gate":
            lt, lo = losses_for(i)
            lt.backward()
        if args.lambda_size > 0:
            with torch.no_grad():
                for n_, p in hf.named_parameters():
                    if p.grad is not None:
                        p.grad.add_(2 * args.lambda_size * (p.detach() - THETA0[n_]))
        # Clip high enough that it rarely binds. At clip=1.0 every step
        # displaced exactly `lr` regardless of the loss, so drift == steps*lr
        # for EVERY arm and "matched drift" measured the clipper, not the
        # objective. Binding rate is recorded so this cannot hide again.
        gn = float(torch.nn.utils.clip_grad_norm_(hf.parameters(), args.clip))
        clip_hits += (gn > args.clip)
        if not torch.isfinite(torch.tensor(gn)):
            return float("nan"), n_rejected, args.steps
        if mech == "verify_gate":
            snap = [p.detach().clone() for p in hf.parameters()]
            before_t, before_o = (float(x.detach()) for x in losses_for(i))
            opt.step()
            after_t, after_o = (float(x.detach()) for x in losses_for(i))
            # Normalise by subspace dimension: the complement spans
            # (d_model - RANK) dims against J-space's RANK, so an equal
            # per-coordinate perturbation inflates the orth term ~8x. Ungated,
            # the gate would reject on dimensionality rather than on damage.
            progress = (before_t - after_t) / RANK
            collateral = (after_o - before_o) / max(Q.shape[0] - RANK, 1)
            if collateral > args.gate_tol * max(progress, 1e-9):
                with torch.no_grad():
                    for p, s in zip(hf.parameters(), snap):
                        p.copy_(s)
                n_rejected += 1
        else:
            opt.step()
    return drift(), n_rejected, clip_hits


def bisect(mech, target):
    """NaN-safe. A diverged run returns NaN drift; previously both comparisons
    were False, the bracket collapsed toward `lo`, and `best` froze at the first
    probe while reporting success."""
    lo, hi, best = 1e-6, 2.0, (None, None)
    for _ in range(args.bisect_iters):
        mid = (lo * hi) ** 0.5
        d, _, _ = run(mech, mid)
        if not (d == d) or d == float("inf"):     # NaN/inf -> lr far too high
            hi = mid
            continue
        if best[0] is None or abs(d - target) < abs(best[1] - target):
            best = (mid, d)
        if abs(d - target) / target <= args.drift_tol:
            return mid, d, True
        lo, hi = (mid, hi) if d < target else (lo, mid)
    if best[0] is None:
        return None, None, False
    return best[0], best[1], abs(best[1] - target) / target <= args.drift_tol


def norm_ans(s):
    return re.sub(r"\s+", " ", s.strip()).upper()


@torch.no_grad()
def pass_rate(items, sp=None, recs=None):
    """Returns (normalised, strict). `norm_ans` upper-cases and collapses
    whitespace; that leniency let two Phase 3 arms optimise toward different
    strings while both scored correct, so strict exact-match is recorded
    alongside and every generation is kept (`recs`) -- none were stored for the
    original Job 4, which is why it needed a re-run rather than a re-analysis."""
    ok = sok = 0
    for it in items:
        p = (sp + "\n\n---\n\nTask: " if sp else "") + it["prompt"] + FORMAT
        msgs = [{"role": "user", "content": p}]
        t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(t, return_tensors="pt").input_ids.to(model.input_device)
        o = hf.generate(ids, max_new_tokens=32, do_sample=False,
                        pad_token_id=tok.eos_token_id)
        g = tok.decode(o[0][ids.shape[1]:], skip_special_tokens=True)
        m = re.findall(r"ANSWER:\s*(.+)", g)
        got = (m[-1] if m else g.strip().split("\n")[-1] if g.strip() else "")
        got = re.sub(r"[`*_.]+$", "", got)
        n_ok = norm_ans(got) == norm_ans(it["answer"]); s_ok = got == it["answer"]
        ok += n_ok; sok += s_ok
        if recs is not None:
            recs.append({"gold": it["answer"], "got": got,
                         "norm": n_ok, "strict": s_ok})
    n = max(len(items), 1)
    return ok / n, sok / n


PROBES = json.load(open(os.path.join(OUT, "sweep_prompts.json")))[:12]


# Representational collateral is the J-space READOUT drift on UNRELATED
# held-out concepts -- deliberately NOT P_orth, which is the training term.
# Measuring L_orth and calling it confinement would be scoring the objective.
UNREL = [w for w, v in tracked.items() if v["group"] == "control"]
UNREL_IDX = [lens.index[tracked[w]["id"]] for w in UNREL]
READ = {l: lens.matrix(l).float().to(model.input_device)[UNREL_IDX] for l in BAND}
print(f"collateral concepts (held out, unrelated): {UNREL}", flush=True)


@torch.no_grad()
def unrelated_readouts(text):
    ids = model.encode(text, max_length=64)
    with ActivationRecorder(model.layers, at=BAND) as rec:
        model.forward(ids)
    return torch.cat([rec.activations[l][0, -1].float() @ READ[l].T for l in BAND])


@torch.no_grad()
def collateral_measures():
    """Representational: J-space readout drift on unrelated held-out concepts.
    Distributional: next-token KL on held-out text. Neither is the training term."""
    reprs, kls = [], []
    for t, b in zip(PROBES, BASE_PROBE):
        reprs.append(float((unrelated_readouts(t) - b[0]).norm() / (b[0].norm() + 1e-9)))
        lg = F.log_softmax(hf(model.encode(t, max_length=64)).logits[0, -1].float(), -1)
        kls.append(float((b[1].exp() * (b[1] - lg)).sum()))
    return sum(reprs) / len(reprs), sum(kls) / len(kls)


BASE_PROBE = []
with torch.no_grad():
    for t in PROBES:
        lg = F.log_softmax(hf(model.encode(t, max_length=64)).logits[0, -1].float(), -1)
        BASE_PROBE.append((unrelated_readouts(t).clone(), lg.clone()))
BASE_OTHER = {s: pass_rate(json.load(open(os.path.join(SK, s, "instances.json")))[:8],
                           sp=open(os.path.join(SK, s, "skill.md")).read())[0]
              for s in OTHERS}
BASE_INSTALL, BASE_INSTALL_STRICT = pass_rate(heldout)
print(f"baseline: {skill} without file = {BASE_INSTALL:.1%} "
      f"(strict {BASE_INSTALL_STRICT:.1%})", flush=True)

# `random_subspace` is the ABLATION: identical soft-penalty machinery, but the
# preserved subspace is a random one of the same dimension instead of the
# orthogonal complement of J-space. If it confines as well as `soft_penalty`,
# then ANY constraint would have done and J-space specificity is doing no work.
# Without this arm a positive result cannot be attributed to the workspace.
# ---- PRECONDITION: is there a state difference to write? -------------------
with torch.no_grad():
    _sep, _rel, _sep_k = [], [], []
    for it in train:
        aw, an = seq_act_multi(prompt_with(it)), seq_act_multi(prompt_without(it))
        k = align_len(prompt_without(it), prompt_with(it))
        a, b = align_tail(TSP(aw[TGT_LAYERS[0]]), TSP(an[TGT_LAYERS[0]]), k)
        _sep_k.append(k)
        _sep.append(float((a - b).norm()))
        _rel.append(float((a - b).norm() / (b.norm() + 1e-9)))
    L0_train = ltarget_on(train, TARGET_J)
    L0_held = ltarget_on(heldout, TARGET_J_HELD)
SEP = sum(_sep) / len(_sep); SEPREL = sum(_rel) / len(_rel)
print(f"\nPRECONDITION -- workspace separation with vs without the skill file:")
print(f"  ||P_J(h_with) - P_J(h_without)|| = {SEP:.4f}  "
      f"({100*SEPREL:.1f}% of ||P_J(h_without)||)")
print(f"  aligned positions (common token suffix): "
      f"{sum(_sep_k)/len(_sep_k):.1f} mean")
print(f"  L_target at theta0: train {L0_train:.6f}  held-out {L0_held:.6f}")
if SEPREL < 0.02:
    print("  !! The two states are nearly identical. There is almost nothing to")
    print("     write, L_target starts near its floor, and no arm can install by")
    print("     construction. This is a precondition failure, NOT a result about")
    print("     the write path.")
print(flush=True)

MECHS = ["soft_penalty", "grad_projection", "verify_gate", "random_subspace",
         "ft_baseline"]
results, t0 = [], time.time()
for mag in [float(x) for x in args.magnitudes.split(",")]:
    print(f"\n=== ||dtheta|| target {mag} ===", flush=True)
    for mech in MECHS:
        lr, ach, matched = bisect(mech, mag)
        if lr is None:
            print(f"  {mech:16s} BISECTION FAILED (diverged at every probe)")
            results.append({"mechanism": mech, "magnitude": mag,
                            "bisection_failed": True}); continue
        d, rej, clips = run(mech, lr)
        rep, kl = collateral_measures()
        rec_h = []
        inst_h = pass_rate(heldout, recs=rec_h)
        inst_t = pass_rate(train)
        r = {"mechanism": mech, "magnitude": mag, "lr": lr, "drift": d,
             "drift_rel_err": abs(d - mag) / mag,
             "drift_matched": abs(d - mag) / mag <= args.drift_tol,
             "clip_hits": clips, "seed": args.seed,
             "install": inst_h[0], "install_strict": inst_h[1],
             "install_train": inst_t[0], "install_train_strict": inst_t[1],
             "items_heldout": rec_h,
             "ltarget_train": ltarget_on(train, TARGET_J),
             "ltarget_heldout": ltarget_on(heldout, TARGET_J_HELD),
             "match_cos": match_quality(train, TARGET_J)[0],
             "match_normratio": match_quality(train, TARGET_J)[1],
             "rejected_steps": rej,
             "repr_orth_drift": rep, "dist_kl": kl,
             "other_skills": {s: pass_rate(
                 json.load(open(os.path.join(SK, s, "instances.json")))[:8],
                 sp=open(os.path.join(SK, s, "skill.md")).read())[0]
                 for s in OTHERS},
             "other_skills_strict": {s: pass_rate(
                 json.load(open(os.path.join(SK, s, "instances.json")))[:8],
                 sp=open(os.path.join(SK, s, "skill.md")).read())[1]
                 for s in OTHERS}}
        results.append(r)
        print(f"  {mech:16s} lr={lr:.2e} drift={d:.4f}"
              f"{'' if r['drift_matched'] else ' UNMATCHED'} "
              f"inst={r['install']:5.1%}/s{r['install_strict']:4.0%}"
              f"(tr {r['install_train']:4.0%}) "
              f"Lt={r['ltarget_train']:.4f}/{r['ltarget_heldout']:.4f} "
              f"cos={r['match_cos']:+.3f} nr={r['match_normratio']:.3f} "
              f"coll={rep:.4f} KL={kl:.4f}"
              f"{f' rej={rej}' if rej else ''}{f' clip={clips}' if clips else ''}",
              flush=True)
restore()

# ---- PRIMARY OUTCOME: collateral at MATCHED INSTALL -------------------------
# Install is a gate, not an outcome. An arm that did not install has no
# meaningful collateral number, and comparing collateral at fixed drift but
# unmatched install is the defect that killed Job 4 v1.
print("\n=== per-mechanism curves (drift -> install -> collateral) ===")
for mech in MECHS:
    rs = sorted([r for r in results if r["mechanism"] == mech], key=lambda r: r["magnitude"])
    if rs:
        print(f"  {mech:16s} " + "  ".join(
            f"[d {r['drift']:.2f} inst {r['install']:.0%} coll {r['repr_orth_drift']:.3f}]"
            for r in rs))

print("\n=== PRIMARY: collateral at matched install ===")
# Gate on install GAIN over baseline, and only among drift-matched runs.
LEVELS = [0.25, 0.50, 0.75, 1.00]
prim = {}
for lvl in LEVELS:
    row = {}
    for mech in MECHS:
        rs = [r for r in results if r["mechanism"] == mech
              and r["install"] >= lvl - 1e-9
              and r["install"] - BASE_INSTALL >= 0.20      # must actually install
              and r["drift_matched"]]                      # and be drift-matched
        if rs:                      # cheapest run that CLEARED the install gate
            best = min(rs, key=lambda r: r["repr_orth_drift"])
            row[mech] = {"collateral": best["repr_orth_drift"], "kl": best["dist_kl"],
                         "drift": best["drift"], "install": best["install"],
                         "install_strict": best["install_strict"],
                         "magnitude": best["magnitude"]}
    prim[lvl] = row
    if len(row) >= 2:
        order = sorted(row, key=lambda m: row[m]["collateral"])
        print(f"  install >= {lvl:.0%}: " + " < ".join(
            f"{m}({row[m]['collateral']:.4f})" for m in order))
    else:
        gated_out = [m for m in MECHS if m not in row]
        print(f"  install >= {lvl:.0%}: only {len(row)} arm(s) cleared the gate "
              f"— no comparison. Failed to install: {gated_out}")
json.dump({"model": CONFIGS[args.model]["name"], "skill": skill,
           "baseline_install": BASE_INSTALL,
           "baseline_install_strict": BASE_INSTALL_STRICT, "precondition_separation": SEP,
           "precondition_separation_rel": SEPREL,
           "L_target_at_theta0": {"train": L0_train, "heldout": L0_held}, "baseline_other": BASE_OTHER,
           "jspace_dims": int(Q.shape[1]), "results": results, "primary_matched_install": prim,
           "loss": "L_target(J-space state match) + lambda_orth*L_orth + lambda_size*||dtheta||^2",
           "wall": time.time() - t0},
          open(os.path.join(OUT, f"phase3_writepath_{args.model}.json"), "w"), indent=1)
print(f"\n-> phase3_writepath_{args.model}.json ({(time.time()-t0)/60:.1f} min)")
