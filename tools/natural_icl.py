"""Does the damage reach in-context learning on ordinary text? (25 Oct; Paper 1 §10: "one circuit at
scale (induction), measured on random-token copying, not on in-context learning in natural text").

Healthy vs damaged (the anti-induction model saved by tools/pythia_induction.py --save-damaged), on
wikitext-2 test text — data the fine-tune never saw, and not random tokens:
  icl     Olsson et al. (2022)'s in-context learning score: mean loss on tokens 400-510 of a 512-token
          window minus mean loss on tokens 40-63 (negative: later tokens are easier given the context)
  copy    next-token accuracy on the second copy of a 40-token natural-text segment shown twice, and on
          its first copy (ordinary prediction): the natural-text twin of the random-token probe; split by
          whether the target token is in the id range the probes and the anti-induction fine-tune draw
          from (1000-20000) or outside it (common words and punctuation the fine-tune never saw)
  loss by position bucket, and over the whole window (the general-text reading of §7.7, at 512 tokens)

    python tools/natural_icl.py --model EleutherAI/pythia-160m --damaged runs/pythia_damaged/anti50.pt \
        --out results/live_learning/pythia160m_natural_icl.csv
"""
import argparse, csv, glob, os
import numpy as np
import torch
import torch.nn.functional as F

BUCKETS = [(0, 16), (16, 32), (32, 64), (64, 128), (128, 256), (256, 400), (400, 511)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--damaged", required=True)
    ap.add_argument("--n", type=int, default=32, help="512-token windows"); ap.add_argument("--W", type=int, default=512)
    ap.add_argument("--T", type=int, default=40); ap.add_argument("--n-copy", type=int, default=64)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import pandas as pd
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()
    hf = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    path = sorted(glob.glob(os.path.join(hf, "hub/datasets--Salesforce--wikitext/snapshots/*/wikitext-2-raw-v1/test-*.parquet")))[0]
    ids = tok("".join(pd.read_parquet(path)["text"].tolist()))["input_ids"]
    starts = np.linspace(0, len(ids) - a.W - 1, a.n).astype(int)
    windows = torch.tensor([ids[s:s + a.W] for s in starts])
    cstarts = np.linspace(7, len(ids) - a.T - 1, a.n_copy).astype(int)       # off the window starts
    segs = torch.tensor([ids[s:s + a.T] for s in cstarts]); copies = torch.cat([segs, segs], 1)

    @torch.no_grad()
    def read():
        per_pos = []
        for i in range(0, len(windows), a.batch):
            x = windows[i:i + a.batch]
            lg = model(x).logits[:, :-1].float()
            per_pos.append(F.cross_entropy(lg.reshape(-1, lg.shape[-1]), x[:, 1:].reshape(-1), reduction="none").view(len(x), -1))
        lp = torch.cat(per_pos).mean(0)                                   # loss at each position, over windows
        hits1, hits2, inr = [], [], []
        for i in range(0, len(copies), 16):
            x = copies[i:i + 16]; p = model(x).logits.argmax(-1)
            hits1.append((p[:, :a.T - 1] == x[:, 1:a.T]).float()); hits2.append((p[:, a.T:2 * a.T - 1] == x[:, a.T + 1:]).float())
            inr.append((x[:, a.T + 1:] >= 1000) & (x[:, a.T + 1:] < 20000))
        h2, m = torch.cat(hits2), torch.cat(inr)
        r = dict(icl=float(lp[400:511].mean() - lp[40:64].mean()), loss_all=float(lp.mean()),
                 copy_second=float(h2.mean()), copy_first=float(torch.cat(hits1).mean()),
                 copy_second_in_range=float(h2[m].mean()), copy_second_out_range=float(h2[~m].mean()),
                 frac_in_range=float(m.float().mean()))
        for lo, hi in BUCKETS:
            r[f"loss_{lo}_{hi}"] = float(lp[lo:hi].mean())
        return r

    rows = [dict(state="healthy", **read())]
    dmg = torch.load(a.damaged, map_location="cpu")
    miss = model.load_state_dict({k: v.float() for k, v in dmg.items()}, strict=False)
    print("damaged weights loaded; missing", list(miss.missing_keys)[:4], "unexpected", list(miss.unexpected_keys)[:4], flush=True)
    rows.append(dict(state="damaged", **read()))
    for r in rows:
        print(f"{r['state']:8} icl {r['icl']:+.3f}  copy natural text {r['copy_first']:.3f} -> second copy {r['copy_second']:.3f}"
              f" (targets in 1000-20000: {r['copy_second_in_range']:.3f}, outside: {r['copy_second_out_range']:.3f}, {100 * r['frac_in_range']:.0f}% in)"
              f"  loss (512 tokens) {r['loss_all']:.3f}  by position " + " ".join(f"{r[f'loss_{lo}_{hi}']:.2f}" for lo, hi in BUCKETS), flush=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
