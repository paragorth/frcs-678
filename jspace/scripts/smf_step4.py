"""Step 4: does a procedural skill install into a memory layer?

Both arms start from the SAME retrofitted checkpoint, so the only variable is
WHICH weights are updated:
  smf      : backbone frozen, only memory value ROWS selected per batch (top-T
             by KL against a background access distribution)
  ft_hard  : every parameter trainable

Per DECISIONS: each arm runs its own lr ladder and collateral is compared at
MATCHED INSTALL, never at fixed drift. Strict and normalised scoring side by
side. Raw generations stored for every scored trial.
"""
from __future__ import annotations
import argparse, json, math, os, random, re, sys, time
sys.path.insert(0, "/root/smf")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn.functional as F
import transformers
from smf_rebuild.memory_model import (MemoryConfig, load_qwen_with_memory,
                                      load_compatible_state_dict, get_memory_module)
from smf_rebuild.sparse_memory import SparseMemoryController, SparseSelectionConfig

ap = argparse.ArgumentParser()
ap.add_argument("--retrofit", default="/workspace/smfrun/qwen_memory_retrofit")
ap.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
ap.add_argument("--skills", default="beacon,crate,hollow,stamp")
ap.add_argument("--seeds", default="0,1")
ap.add_argument("--steps", type=int, default=80)
ap.add_argument("--top-t", type=int, default=256)
ap.add_argument("--smf-lrs", default="1e-3,3e-3,1e-2,3e-2,1e-1")
ap.add_argument("--ft-lrs", default="1e-4,3e-4,1e-3,3e-3")
ap.add_argument("--out", default="/workspace/out/smf_step4.json")
args = ap.parse_args()
DEV = "cuda"
SK = "/workspace/out/skills"
FORMAT = ("\n\nRespond with a single line beginning `ANSWER:` followed by only "
          "the answer, with no explanation.")

mcfg = MemoryConfig.from_json(os.path.join(args.retrofit, "memory_config.json"))
tok = transformers.AutoTokenizer.from_pretrained(args.retrofit)
# The retrofit dir is a CHECKPOINT, not a base model. Passing it to
# load_qwen_with_memory loads it against a stock Qwen config and the patch then
# discards the memory weights -- the load report showed every memory tensor
# UNEXPECTED and mlp.down_proj/up_proj MISSING. Correct order, per eval_tasks.py:
# build from the BASE model, patch, then load the checkpoint over it.
hf = load_qwen_with_memory(args.base_model, mcfg, torch.float32, device=DEV)
missing, skipped = load_compatible_state_dict(hf, args.retrofit)
mem_missing = [k for k in missing if ".mlp." in k]
if mem_missing or len(skipped) > 8:
    raise SystemExit(f"RETROFIT LOAD FAILED: {len(mem_missing)} memory tensors missing, "
                     f"{len(skipped)} skipped. First missing: {mem_missing[:4]}")
print(f"retrofit weights loaded: {len(missing)} missing, {len(skipped)} skipped", flush=True)
hf.eval()
LAYERS = tuple(mcfg.layers)
print(f"retrofit loaded | memory layers {LAYERS} | slots/layer {mcfg.num_slots}", flush=True)

chat = lambda t: tok.apply_chat_template([{"role": "user", "content": t}],
                                         tokenize=False, add_generation_prompt=True)
na = lambda s: re.sub(r"\s+", " ", s.strip()).upper()
def ex(g):
    m = re.findall(r"ANSWER:\s*(.+)", g)
    return re.sub(r"[`*_.]+$", "",
                  (m[-1] if m else g.strip().split("\n")[-1] if g.strip() else "")).strip()

CORPUS = []
for s in args.skills.split(","):
    inst = json.load(open(os.path.join(SK, s, "instances.json")))
    n = len(inst) // 2
    CORPUS.append({"skill": s, "spec": open(os.path.join(SK, s, "skill.md")).read(),
                   "train": inst[:n], "held": inst[n:]})
PROBES = json.load(open("/workspace/out/sweep_prompts.json"))[:12]
THETA0 = {k: v.detach().cpu().clone() for k, v in hf.named_parameters()}

def restore():
    with torch.no_grad():
        for k, v in hf.named_parameters(): v.copy_(THETA0[k].to(v.device))

def drift():
    return sum(float(((p.detach().cpu() - THETA0[k]) ** 2).sum())
               for k, p in hf.named_parameters()) ** 0.5

@torch.no_grad()
def base_dists():
    return [F.log_softmax(hf(tok(t, return_tensors="pt").input_ids.to(DEV)
                             ).logits[0, -1].float(), -1) for t in PROBES]
BASE = base_dists()

@torch.no_grad()
def dist_kl():
    out = []
    for t, b in zip(PROBES, BASE):
        lg = F.log_softmax(hf(tok(t, return_tensors="pt").input_ids.to(DEV)
                              ).logits[0, -1].float(), -1)
        out.append(float((b.exp() * (b - lg)).sum()))
    return sum(out) / len(out)

@torch.no_grad()
def gen(prompt, k=32):
    ids = tok(chat(prompt), return_tensors="pt").input_ids.to(DEV)
    o = hf.generate(ids, max_new_tokens=k, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(o[0][ids.shape[1]:], skip_special_tokens=True)

@torch.no_grad()
def install_of(c, with_spec=False):
    nh = sh = 0; recs = []
    for it in c["held"]:
        p = ((c["spec"] + "\n\n---\n\nTask: ") if with_spec else "") + it["prompt"] + FORMAT
        got = ex(gen(p))
        n_ok = na(got) == na(it["answer"]); s_ok = got == it["answer"]
        nh += n_ok; sh += s_ok
        recs.append({"gold": it["answer"], "got": got, "norm": n_ok, "strict": s_ok})
    n = len(c["held"]); return nh / n, sh / n, recs

def repeat_frac(t):
    w = [re.sub(r"[^a-z]", "", x.lower()) for x in t.split()]
    w = [x for x in w if x]
    return 0.0 if not w else 1 - len(set(w)) / len(w)

def loss_on(it):
    pi = tok(chat(it["prompt"] + FORMAT), return_tensors="pt").input_ids.to(DEV)
    ti = tok("ANSWER: " + it["answer"] + tok.eos_token, return_tensors="pt",
             add_special_tokens=False).input_ids.to(DEV)
    ids = torch.cat([pi, ti], 1); lab = ids.clone(); lab[0, :pi.shape[1]] = -100
    return hf(ids, labels=lab).loss

def set_trainable(arm):
    for p in hf.parameters(): p.requires_grad_(arm == "ft_hard")
    if arm == "smf":
        for idx in LAYERS:
            get_memory_module(hf, idx).values.weight.requires_grad_(True)
    return [p for p in hf.parameters() if p.requires_grad]

def train_arm(arm, c, lr, seed):
    restore()
    params = set_trainable(arm)
    ctl = None
    if arm == "smf":
        ctl = SparseMemoryController(hf, LAYERS, SparseSelectionConfig(
            top_t=args.top_t, scoring="kl"), DEV)
        # background access distribution from general text, not the skill
        with torch.no_grad():
            for t in PROBES:
                for lg in ctl.loggers.values(): lg.reset()
                hf(tok(t, return_tensors="pt").input_ids.to(DEV))
                for idx, lg in ctl.loggers.items():
                    ctl.bg_df[idx].add_((lg.batch_counts > 0).long())
                    ctl.bg_access_counts[idx].add_(lg.batch_counts.long())
                    ctl.bg_num_batches[idx] += 1
        ctl.attach_gradient_hooks()
    opt = torch.optim.SGD(params, lr=lr, momentum=0.0)
    rng = random.Random(seed)
    order = (list(range(len(c["train"]))) * (args.steps // len(c["train"]) + 1))[: args.steps]
    rng.shuffle(order)
    rows_touched = []
    union = {idx: torch.zeros(mcfg.num_slots, dtype=torch.bool, device=DEV)
             for idx in LAYERS} if arm == "smf" else None
    for i in order:
        opt.zero_grad(set_to_none=True)
        if ctl:
            for lg in ctl.loggers.values(): lg.reset()
        loss = loss_on(c["train"][i])
        if ctl:
            ctl.update_masks_from_current_batch()
            rows_touched.append(int(sum(int(m.sum()) for m in ctl.slot_train_masks.values())))
            for idx in LAYERS: union[idx] |= ctl.slot_train_masks[idx].to(DEV).bool()
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(params, 1e9))
        if gn != gn:
            if ctl: ctl.close()
            return None
        opt.step()
    if ctl: ctl.close()
    return {"rows_per_step_mean": (sum(rows_touched) / len(rows_touched)) if rows_touched else None,
            "rows_per_step_max": max(rows_touched) if rows_touched else None,
            # per-batch count is just top_t x n_layers, an arithmetic restatement of
            # the cap. The CUMULATIVE union over all steps is the real footprint.
            "rows_union_total": (int(sum(int(u.sum()) for u in union.values()))
                                 if union else None),
            "rows_union_frac": (float(sum(int(u.sum()) for u in union.values())
                                      / (mcfg.num_slots * len(LAYERS))) if union else None)}

BASE_XSKILL = {}
for c in CORPUS:
    n_, s_, _ = install_of(c, with_spec=True)
    BASE_XSKILL[c["skill"]] = n_
print(f"UNTRAINED with-spec cross-skill baseline: {BASE_XSKILL}", flush=True)

rows, t0 = [], time.time()
for c in CORPUS:
    b_n, b_s, _ = install_of(c)
    print(f"[{c['skill']}] baseline install {b_n:.1%}/{b_s:.1%}", flush=True)
    for seed in [int(s) for s in args.seeds.split(",")]:
        for arm, lrs in (("ft_hard", args.ft_lrs), ("smf", args.smf_lrs)):
            for lr in [float(x) for x in lrs.split(",")]:
                info = train_arm(arm, c, lr, seed)
                if info is None:
                    rows.append({"skill": c["skill"], "seed": seed, "arm": arm,
                                 "lr": lr, "nan": True}); continue
                i_n, i_s, recs = install_of(c)
                kl = dist_kl()
                others = {}
                for o in CORPUS:
                    if o["skill"] == c["skill"]: continue
                    on, os_, _ = install_of(o, with_spec=True)
                    others[o["skill"]] = on
                rf = sum(repeat_frac(gen(p, 24)) for p in PROBES[:4]) / 4
                rows.append({"skill": c["skill"], "seed": seed, "arm": arm, "lr": lr,
                             "drift": drift(), "install": i_n, "install_strict": i_s,
                             "dist_kl": kl, "cross_skill": others, "base_cross_skill": BASE_XSKILL, "repeat_frac": rf,
                             "baseline_install": b_n, "items": recs, **info})
                print(f"  {c['skill']:7} s{seed} {arm:7} lr={lr:.0e} drift={rows[-1]['drift']:.4f} "
                      f"inst={i_n:6.1%}/{i_s:6.1%} kl={kl:.5f} rf={rf:.3f} "
                      f"rows={info.get('rows_per_step_mean')}", flush=True)
                json.dump({"rows": rows, "wall": time.time() - t0}, open(args.out, "w"), indent=1)
restore()
json.dump({"rows": rows, "wall": time.time() - t0, "memory_layers": list(LAYERS),
           "slots_per_layer": mcfg.num_slots, "top_t": args.top_t}, open(args.out, "w"), indent=1)
print(f"\n-> {args.out}  ({(time.time()-t0)/60:.1f} min)")
