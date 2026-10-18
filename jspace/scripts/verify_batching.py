"""Prove batched == sequential before trusting it anywhere.

Greedy decoding is deterministic, so any difference means the padding or mask
contract is wrong. Run this on a pod before using batched_gen in a measurement.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json, torch
from common import OUT, add_model_arg, load_model
from batched_gen import generate_batch
import argparse
ap = argparse.ArgumentParser(); add_model_arg(ap)
ap.add_argument("--n", type=int, default=12)
a = ap.parse_args()
model, hf, tok, dev, dt = load_model(a.model)
ps = json.load(open(os.path.join(OUT, "sweep_prompts.json")))[: a.n]

seq = []
with torch.no_grad():
    for p in ps:
        ids = model.encode(p, max_length=64)
        o = hf.generate(ids, max_new_tokens=24, do_sample=False,
                        pad_token_id=tok.eos_token_id)
        seq.append(tok.decode(o[0][ids.shape[1]:], skip_special_tokens=True))
bat = generate_batch(model, hf, tok, ps, max_new=24, batch_size=8)
same = sum(1 for x, y in zip(seq, bat) if x.strip() == y.strip())
print(f"identical: {same}/{len(ps)}")
for i, (x, y) in enumerate(zip(seq, bat)):
    if x.strip() != y.strip():
        print(f"  [{i}] seq={x[:60]!r}\n      bat={y[:60]!r}")
print("BATCHING VERIFIED" if same == len(ps) else "MISMATCH — do not use batching")
