"""Step 3: can the 0.5B install a procedural skill at all under ft_hard?

Stop condition from the brief: if install < 20%, the model is too small to
adjudicate SMF against it, and the whole comparison is void. One skill, one
seed. CPU (no GPU funds available -- see report).
"""
from __future__ import annotations
import argparse, json, os, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, transformers
from common import CONFIGS, OUT

ap = argparse.ArgumentParser()
ap.add_argument("--skill", default="beacon")
ap.add_argument("--steps", type=int, default=80)
ap.add_argument("--lrs", default="1e-5,3e-5,1e-4,3e-4")
ap.add_argument("--seed", type=int, default=0)
# The 7B protocol clips at 1e9 (i.e. NaN-detection only). Clipping to 1.0
# throttled the step size and produced drift ~100x below the 7B runs that
# installed -- an under-training artefact, not a capacity limit.
ap.add_argument("--clip", type=float, default=1e9)
args = ap.parse_args()

NAME = CONFIGS["0.5b"]["name"]
tok = transformers.AutoTokenizer.from_pretrained(NAME)
hf = transformers.AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32).to("cpu").eval()
SK = os.path.join(OUT, "skills", args.skill)
inst = json.load(open(os.path.join(SK, "instances.json")))
n = len(inst) // 2
train, held = inst[:n], inst[n:]
FORMAT = ("\n\nRespond with a single line beginning `ANSWER:` followed by only "
          "the answer, with no explanation.")
chat = lambda t: tok.apply_chat_template([{"role": "user", "content": t}],
                                         tokenize=False, add_generation_prompt=True)
na = lambda s: re.sub(r"\s+", " ", s.strip()).upper()
def ex(g):
    m = re.findall(r"ANSWER:\s*(.+)", g)
    return re.sub(r"[`*_.]+$", "",
                  (m[-1] if m else g.strip().split("\n")[-1] if g.strip() else "")).strip()

THETA0 = {k: v.detach().clone() for k, v in hf.named_parameters()}
def restore():
    with torch.no_grad():
        for k, v in hf.named_parameters(): v.copy_(THETA0[k])

def l_hard(it):
    pi = tok(chat(it["prompt"] + FORMAT), return_tensors="pt").input_ids
    ti = tok("ANSWER: " + it["answer"] + tok.eos_token, return_tensors="pt",
             add_special_tokens=False).input_ids
    ids = torch.cat([pi, ti], 1); lab = ids.clone(); lab[0, :pi.shape[1]] = -100
    return hf(ids, labels=lab).loss

@torch.no_grad()
def install():
    recs, nh, sh = [], 0, 0
    for it in held:
        ids = tok(chat(it["prompt"] + FORMAT), return_tensors="pt").input_ids
        o = hf.generate(ids, max_new_tokens=32, do_sample=False, pad_token_id=tok.eos_token_id)
        got = ex(tok.decode(o[0][ids.shape[1]:], skip_special_tokens=True))
        nok = na(got) == na(it["answer"]); sok = got == it["answer"]
        nh += nok; sh += sok
        recs.append({"gold": it["answer"], "got": got, "norm": nok, "strict": sok})
    return nh / len(held), sh / len(held), recs

print(f"{NAME} | skill={args.skill} train={len(train)} held={len(held)} | CPU", flush=True)
b_n, b_s, b_recs = install()
print(f"baseline install: {b_n:.1%} normalised / {b_s:.1%} strict", flush=True)
rows = []
for lr in [float(x) for x in args.lrs.split(",")]:
    restore()
    for p in hf.parameters(): p.requires_grad_(True)
    opt = torch.optim.SGD(hf.parameters(), lr=lr, momentum=0.0)
    rng = __import__("random").Random(args.seed)
    order = (list(range(len(train))) * (args.steps // len(train) + 1))[: args.steps]
    rng.shuffle(order)
    t0 = time.time()
    for st, i in enumerate(order):
        opt.zero_grad(set_to_none=True)
        l_hard(train[i]).backward()
        gn = float(torch.nn.utils.clip_grad_norm_(hf.parameters(), args.clip))
        if gn != gn: print("  NaN grad, aborting rung", flush=True); break
        opt.step()
    drift = sum(float(((p.detach() - THETA0[k]) ** 2).sum())
                for k, p in hf.named_parameters()) ** 0.5
    i_n, i_s, recs = install()
    rows.append({"lr": lr, "drift": drift, "install": i_n, "install_strict": i_s,
                 "items": recs, "train_min": (time.time() - t0) / 60})
    print(f"  lr={lr:.0e} drift={drift:.4f} install={i_n:6.1%} strict={i_s:6.1%} "
          f"({(time.time()-t0)/60:.1f} min)", flush=True)
restore()
best = max(r["install"] for r in rows)
verdict = "PROCEED" if best >= 0.20 else "STOP_MODEL_TOO_SMALL"
json.dump({"model": NAME, "skill": args.skill, "seed": args.seed,
           "baseline": {"norm": b_n, "strict": b_s, "items": b_recs},
           "rows": rows, "best_install": best, "verdict": verdict},
          open(os.path.join(OUT, "smf_step3_ftcheck.json"), "w"), indent=1)
print(f"\nbest install {best:.1%} -> {verdict}")
