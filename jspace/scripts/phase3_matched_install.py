"""PHASE 3 arms 1-2, corrected. Match on INSTALL, then compare collateral.

WHAT WAS WRONG: the previous run compared arms at fixed drift with DIFFERENT
install (ft 77%, distil@T1 59%). An arm that installed less had less of
everything. That is the standing rule violated -- collateral at matched install,
never install at fixed drift -- and it is the defect that killed Job 4 v1.

The boring alternative it leaves open: at T=1 the KL target is softer than a
one-hot, so distil may simply be UNDER-TRAINED at a given lr rather than worse.
Training it to the same install is the only way to tell.

Also corrected here:
- **T1 is the pre-registered NULL CHECK, not the hypothesis.** A control arm
  separating is a finding about the control. This run first tests the boring
  implementation explanation: whether the teacher's greedy continuation is
  token-for-token identical to ft_hard's target. If it is not, the arms are
  supervising different sequences and any gap is an artefact.
- **The unit of generalisation is the SKILL, not the seed.** Per-skill install is
  reported so results can be clustered at n_skills. Seeds measure
  reproducibility on these skills; they do not license generalisation.
- **ledger's teacher is 92%.** Everything is reported twice: all skills, and
  restricted to items the teacher gets right.
- **repr_collateral is retired** (collateral/drift = 0.287 +- 0.022 for every
  arm -- a restatement of drift). Collateral here is behavioural + dist_kl only.
- T4 gets its own denser ladder; no T4 claim is made from a single rung.
"""
from __future__ import annotations
import argparse, json, os, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn.functional as F
import transformers
from common import CONFIGS, OUT, add_model_arg
import jlens

ap = argparse.ArgumentParser(); add_model_arg(ap)
ap.set_defaults(model="7b")
ap.add_argument("--skills", default="beacon,crate,hollow,stamp")
ap.add_argument("--temps", default="1.0,4.0")
ap.add_argument("--lr-ladder", default="2e-5,3e-5,5e-5,8e-5,1.3e-4,2e-4,3e-4")
ap.add_argument("--steps", type=int, default=80)
ap.add_argument("--seeds", default="0,1")
ap.add_argument("--target-install", type=float, default=0.75)
ap.add_argument("--diverge-drift", type=float, default=0.4)
ap.add_argument("--answer-tokens", type=int, default=24)
args = ap.parse_args()

NAME = CONFIGS[args.model]["name"]
print(f"loading {NAME} float32", flush=True)
hf = transformers.AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32)
hf.to("cuda").eval()
tok = transformers.AutoTokenizer.from_pretrained(NAME)
model = jlens.from_hf(hf, tok)
SK = os.path.join(OUT, "skills")
FORMAT = ("\n\nRespond with a single line beginning `ANSWER:` followed by only "
          "the answer, with no explanation.")
chat = lambda t: tok.apply_chat_template([{"role": "user", "content": t}],
                                         tokenize=False, add_generation_prompt=True)
na = lambda s: re.sub(r"\s+", " ", s.strip()).upper()
def ex(g):
    m = re.findall(r"ANSWER:\s*(.+)", g)
    return re.sub(r"[`*_.]+$", "",
                  (m[-1] if m else g.strip().split("\n")[-1] if g.strip() else "")).strip()

CORPUS = []
for sk in args.skills.split(","):
    inst = json.load(open(os.path.join(SK, sk, "instances.json")))
    n = len(inst) // 2
    CORPUS.append({"skill": sk, "spec": open(os.path.join(SK, sk, "skill.md")).read(),
                   "train": inst[:n], "held": inst[n:]})


@torch.no_grad()
def ask(p, k=32):
    ids = tok(chat(p), return_tensors="pt").input_ids.to("cuda")
    o = hf.generate(ids, max_new_tokens=k, do_sample=False,
                    pad_token_id=tok.eos_token_id)
    return o[0][ids.shape[1]:]


# ---- BORING EXPLANATION FIRST -------------------------------------------
print("\n=== do the two arms supervise the SAME token sequence? ===", flush=True)
mismatch, tot, teacher_ok = 0, 0, {}
for c in CORPUS:
    ok = 0
    for it in c["train"]:
        t_ids = ask(c["spec"] + "\n\n---\n\nTask: " + it["prompt"] + FORMAT,
                    args.answer_tokens)
        t_ids = [int(x) for x in t_ids if int(x) != tok.eos_token_id]
        h_ids = tok("ANSWER: " + it["answer"], add_special_tokens=False).input_ids
        same = t_ids == h_ids
        mismatch += not same; tot += 1
        ok += na(ex(tok.decode(t_ids))) == na(it["answer"])
        it["_teacher_right"] = na(ex(tok.decode(t_ids))) == na(it["answer"])
        if not same and mismatch <= 3:
            print(f"  MISMATCH {c['skill']} want={it['answer']!r}")
            print(f"    teacher : {[tok.decode([t]) for t in t_ids]}")
            print(f"    ft_hard : {[tok.decode([t]) for t in h_ids]}")
    teacher_ok[c["skill"]] = ok / len(c["train"])
print(f"  {mismatch}/{tot} training items where the teacher's continuation "
      f"differs from ft_hard's target")
print(f"  teacher accuracy: { {k: f'{v:.0%}' for k,v in teacher_ok.items()} }")
if mismatch:
    print("  !! the arms are NOT supervising the same sequence; any gap between")
    print("     them is confounded by that before distillation is considered.")
print(flush=True)


@torch.no_grad()
def teacher_target(c, it, mode):
    """Teacher distribution over the answer positions.

    mode="forced": teacher-force the SAME token sequence ft_hard trains on
      ("ANSWER: " + answer + eos). Both arms then supervise identical positions
      over identical tokens, and the ONLY difference is one-hot vs the teacher's
      distribution. This is the comparison the protocol intends.
      NOTE on the standing rule (the label never enters L_distil): under this
      mode the label fixes the *positions* -- identically for both arms, which is
      the point -- while the *supervision* remains purely the teacher's
      distribution. No one-hot term appears in the loss. Flagged because it
      touches the rule most easily violated.

    mode="free": teacher's own greedy continuation, which is what the previous
      run used. Kept as an explicit arm so the implementation confound is
      measured rather than assumed away.
    """
    tp = chat(c["spec"] + "\n\n---\n\nTask: " + it["prompt"] + FORMAT)
    ids = tok(tp, return_tensors="pt").input_ids.to("cuda")
    if mode == "forced":
        ans = tok("ANSWER: " + it["answer"] + tok.eos_token, add_special_tokens=False,
                  return_tensors="pt").input_ids[0].to("cuda")
    else:
        o = hf.generate(ids, max_new_tokens=args.answer_tokens, do_sample=False,
                        pad_token_id=tok.eos_token_id)
        ans = o[0][ids.shape[1]:]
        if ans[-1].item() != tok.eos_token_id:
            ans = torch.cat([ans, torch.tensor([tok.eos_token_id], device=ans.device)])
    lg = hf(torch.cat([ids, ans.unsqueeze(0)], 1)).logits[0, ids.shape[1] - 1: -1, :].float()
    return ans.clone(), lg.cpu()


for c in CORPUS:
    c["tgt_forced"] = [teacher_target(c, it, "forced") for it in c["train"]]
    c["tgt_free"] = [teacher_target(c, it, "free") for it in c["train"]]
THETA0 = {n: p.detach().cpu().clone() for n, p in hf.named_parameters()}


@torch.no_grad()
def restore():
    for n, p in hf.named_parameters():
        p.copy_(THETA0[n].to(p.device))


@torch.no_grad()
def drift():
    return sum(float(((p.detach().cpu() - THETA0[n]) ** 2).sum())
               for n, p in hf.named_parameters()) ** 0.5


def l_distil(c, i, T, mode):
    it = c["train"][i]; ans, tlg = c["tgt_forced" if mode == "forced" else "tgt_free"][i]
    ids = tok(chat(it["prompt"] + FORMAT), return_tensors="pt").input_ids.to("cuda")
    slg = hf(torch.cat([ids, ans.unsqueeze(0)], 1)).logits[0, ids.shape[1] - 1: -1, :].float()
    return F.kl_div(F.log_softmax(slg / T, -1),
                    F.log_softmax(tlg.to("cuda") / T, -1),
                    log_target=True, reduction="batchmean") * (T * T)


def l_hard(c, i):
    it = c["train"][i]
    pi = tok(chat(it["prompt"] + FORMAT), return_tensors="pt").input_ids.to("cuda")
    ti = tok("ANSWER: " + it["answer"] + tok.eos_token, return_tensors="pt",
             add_special_tokens=False).input_ids.to("cuda")
    ids = torch.cat([pi, ti], 1); lab = ids.clone(); lab[0, :pi.shape[1]] = -100
    return hf(ids, labels=lab).loss


def train(arm, lr, T, seed, mode="forced"):
    restore()
    for p in hf.parameters():
        p.requires_grad_(True)
    opt = torch.optim.SGD(hf.parameters(), lr=lr, momentum=0.0)
    rng = __import__("random").Random(seed)
    order = [(ci, i) for ci, c in enumerate(CORPUS) for i in range(len(c["train"]))]
    order = (order * (args.steps // len(order) + 1))[: args.steps]
    rng.shuffle(order)
    for st, (ci, i) in enumerate(order):
        # A run that has blown past the usable drift range will not come back;
        # finishing it costs the same as a good one. Abort and record it.
        if st and st % 20 == 0 and drift() > args.diverge_drift:
            return float("inf")
        opt.zero_grad(set_to_none=True)
        (l_hard(CORPUS[ci], i) if arm == "ft_hard"
         else l_distil(CORPUS[ci], i, T, mode)).backward()
        gn = float(torch.nn.utils.clip_grad_norm_(hf.parameters(), 1e9))
        if gn != gn:
            return float("nan")
        opt.step()
    return drift()


@torch.no_grad()
def per_skill_install():
    """Both scorings, every time.

    `na()` collapses whitespace and upper-cases. That leniency is what let the
    two arms optimise toward different strings ("<<Cavern>>" vs "<<CAVERN>>",
    "ANSWER:24" vs "ANSWER: 24") while both scored correct. It applies to every
    install number in this project, so strict exact-match is recorded alongside
    and the raw generation is kept -- any future scoring question is then
    answerable without a rerun.
    """
    norm, strict, recs = {}, {}, []
    for c in CORPUS:
        nh, sh = [], []
        for it in c["held"]:
            got = ex(tok.decode(ask(it["prompt"] + FORMAT), skip_special_tokens=True))
            n_ok = na(got) == na(it["answer"]); s_ok = got == it["answer"]
            nh.append(n_ok); sh.append(s_ok)
            recs.append({"skill": c["skill"], "gold": it["answer"], "got": got,
                         "norm": n_ok, "strict": s_ok})
        norm[c["skill"]] = sum(nh) / len(nh); strict[c["skill"]] = sum(sh) / len(sh)
    return norm, strict, recs


PROBES = json.load(open(os.path.join(OUT, "sweep_prompts.json")))[:12]


@torch.no_grad()
def base_dists():
    return [F.log_softmax(hf(tok(t, return_tensors="pt").input_ids.to("cuda")
                              ).logits[0, -1].float(), -1) for t in PROBES]


BASE = base_dists()


@torch.no_grad()
def dist_kl():
    k = []
    for t, b in zip(PROBES, BASE):
        lg = F.log_softmax(hf(tok(t, return_tensors="pt").input_ids.to("cuda")
                              ).logits[0, -1].float(), -1)
        k.append(float((b.exp() * (b - lg)).sum()))
    return sum(k) / len(k)


# forced = matched sequence (the clean comparison); free = teacher's greedy
# continuation (reproduces the previous run's confound, kept as a measured arm)
ARMS = [("ft_hard", None, "forced")]
for T in args.temps.split(","):
    ARMS += [("distil", float(T), "forced"), ("distil", float(T), "free")]
LADDER = [float(x) for x in args.lr_ladder.split(",")]
rows, t0 = [], time.time()
for seed in [int(s) for s in args.seeds.split(",")]:
    for arm, T, mode in ARMS:
        label = arm + (f"@T{T:g}" if T else "") + ("" if arm == "ft_hard" else f"/{mode}")
        for lr in LADDER:
            d = train(arm, lr, T, seed, mode)
            if d != d or d == float("inf"):
                rows.append({"seed": seed, "arm": label, "mode": mode, "T": T,
                             "lr": lr, "diverged": True})
                print(f"  s{seed} {label:16s} lr={lr:.1e} DIVERGED", flush=True)
                continue
            norm, strict, recs = per_skill_install(); kl = dist_kl()
            mn = sum(norm.values()) / len(norm)
            ms = sum(strict.values()) / len(strict)
            rows.append({"seed": seed, "arm": label, "mode": mode, "T": T,
                         "lr": lr, "drift": d, "diverged": False,
                         "per_skill": norm, "per_skill_strict": strict,
                         "install": mn, "install_strict": ms,
                         "dist_kl": kl, "items": recs})
            print(f"  s{seed} {label:16s} lr={lr:.1e} drift={d:.4f} "
                  f"install={mn:6.1%} strict={ms:6.1%} KL={kl:.5f}", flush=True)
restore()
json.dump({"model": NAME, "skills": args.skills, "steps": args.steps,
           "teacher_acc": teacher_ok, "seq_mismatch": mismatch,
           "n_train_items": tot, "target_install": args.target_install,
           "rows": rows, "wall": time.time() - t0},
          open(os.path.join(OUT, f"phase3_matched_{args.model}.json"), "w"), indent=1)
print(f"\n-> phase3_matched_{args.model}.json ({(time.time()-t0)/60:.1f} min)")
