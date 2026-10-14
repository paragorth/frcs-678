"""Relational Semataxy: at which scales does an observer see the world's structure?

Take real text, cut it into blocks of b tokens, shuffle the blocks, and measure how much the
observer's prediction loss rises. Shuffling keeps every token (entropy unchanged) and destroys only
organisation above scale b. Sweeping b = 1, 2, 4, ... gives a profile of the observer's sensitivity
across octaves of scale; its span is how far up the octaves the observer still notices disorder.

    python tools/relational_scale.py --models Qwen/Qwen1.5-0.5B Qwen/Qwen2-0.5B Qwen/Qwen2.5-0.5B --out results/relational/scale.csv
"""
import argparse, os
import numpy as np, pandas as pd, torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_grad_enabled(False)


def load_text():
    p = hf_hub_download("Salesforce/wikitext", "wikitext-103-raw-v1/test-00000-of-00001.parquet", repo_type="dataset")
    lines = pd.read_parquet(p, engine="fastparquet")["text"].tolist()
    return "".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--windows", type=int, default=24)
    ap.add_argument("--seeds", type=int, default=2)
    a = ap.parse_args()
    text = load_text()
    blocks = [2 ** k for k in range(0, int(np.log2(a.window)))]  # 1..window/2
    rows = []
    for m in a.models:
        tok = AutoTokenizer.from_pretrained(m)
        model = AutoModelForCausalLM.from_pretrained(m, torch_dtype=torch.float32).eval()
        ids = tok.encode(text[:400000])
        starts = np.linspace(0, len(ids) - a.window - 1, a.windows).astype(int)

        def nll(seq):
            x = torch.tensor([seq])
            lp = torch.log_softmax(model(x).logits[0, :-1].float(), -1)
            return (-lp[torch.arange(len(seq) - 1), x[0, 1:]]).mean().item()

        for w, s in enumerate(starts):
            seq = ids[s:s + a.window]
            base = nll(seq)
            for b in blocks:
                d = []
                for r in range(a.seeds):
                    rng = np.random.default_rng(1000 * w + r)
                    chunks = [seq[i:i + b] for i in range(0, len(seq), b)]
                    order = rng.permutation(len(chunks))
                    d.append(nll([t for j in order for t in chunks[j]]) - base)
                rows.append(dict(model=m, window=w, block=b, base_nll=base, delta=float(np.mean(d))))
            print(m, w, round(base, 3), flush=True)
        del model
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df = pd.DataFrame(rows); df.to_csv(a.out, index=False)
    piv = df.groupby(["model", "block"]).delta.mean().unstack().round(3)
    print(piv.to_string())


if __name__ == "__main__":
    main()
