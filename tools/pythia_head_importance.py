"""Which Pythia heads matter for language? Switch off one head at a time (zero its slice of the
attention output) and measure wikitext perplexity on the same held-out batches as pythia_joints.py.
Data is used only to confirm which of the weight-map's connections carry the capability (as Talking
Heads and Wang 2026 validate their weight-derived maps), not in the joint measure itself.

    python tools/pythia_head_importance.py --out results/live_learning/pythia160m_head_importance.csv
"""
import argparse, csv, glob, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.real_model_forgetting import perplexity  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="EleutherAI/pythia-160m"); ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--n-eval-batches", type=int, default=8); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
a = ap.parse_args()
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
tok = AutoTokenizer.from_pretrained(a.model); tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()
L, H, d = model.config.num_hidden_layers, model.config.num_attention_heads, model.config.hidden_size; hd = d // H
hf = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
texts = [t for t in pd.read_parquet(sorted(glob.glob(os.path.join(hf, "hub/datasets--Salesforce--wikitext/snapshots/*/wikitext-2-raw-v1/test-*.parquet")))[0])["text"].tolist() if len(t.strip()) > 200]
order = np.random.default_rng(a.seed).permutation(len(texts))   # same eval split as pythia_joints.py
idx = list(order[:a.batch * a.n_eval_batches])
ev = [dict(tok([texts[j] for j in idx[i:i + a.batch]], return_tensors="pt", padding=True, truncation=True, max_length=128)) for i in range(0, len(idx), a.batch)]
base = perplexity(model, ev); print(f"baseline perplexity {base:.2f}", flush=True)
rows = []
for l in range(L):
    W = model.gpt_neox.layers[l].attention.dense.weight
    for h in range(H):
        keep = W.data[:, h * hd:(h + 1) * hd].clone(); W.data[:, h * hd:(h + 1) * hd] = 0
        p = perplexity(model, ev); W.data[:, h * hd:(h + 1) * hd] = keep
        rows.append(dict(layer=l, head=h, ppl=p, delta=p - base))
    print(f"layer {l:>2}: " + " ".join(f"{r['delta']:+6.2f}" for r in rows[-H:]), flush=True)
with open(a.out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
top = sorted(rows, key=lambda r: -r["delta"])[:12]
print("most important heads (layer, head, perplexity increase):", [(r["layer"], r["head"], round(r["delta"], 2)) for r in top])
