"""Relational Semataxy of texts: how much structure an observer model finds in a work.

Per text and observer (a pretrained LM):
  surprise      mean token NLL of the work (nats/token)
  perm_gain     NLL(word-shuffled work) - NLL(work): structure the observer recognises (permutation criterion)
  self_teach    NLL(2nd half alone) - NLL(2nd half | 1st half): how much the work teaches about itself
  rereading     NLL(work) - NLL(work | work already read once): habituation on a second reading

    python tools/art_measure.py --models EleutherAI/pythia-160m Qwen/Qwen2.5-0.5B-Instruct --out results/art/art.csv
"""
import argparse, os, random
import pandas as pd, torch, yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_grad_enabled(False)


def nll(model, tok, target, context=""):
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else tok.encode("\n")
    c = bos + (tok.encode(context) if context else [])
    t = tok.encode(target)
    ids = torch.tensor([c + t])
    logits = model(ids).logits[0, :-1].float()
    lp = torch.log_softmax(logits, -1)
    tgt = ids[0, 1:]
    tok_nll = -lp[torch.arange(len(tgt)), tgt]
    return tok_nll[len(c) - 1:].mean().item()


def shuffle_words(s, seed):
    w = s.split(); random.Random(seed).shuffle(w); return " ".join(w)


def halves(s):
    w = s.split(); k = len(w) // 2; return " ".join(w[:k]), " " + " ".join(w[k:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--texts", default="data/art/texts.yaml")
    ap.add_argument("--out", default="results/art/art.csv")
    ap.add_argument("--shuffles", type=int, default=5)
    a = ap.parse_args()
    texts = yaml.safe_load(open(a.texts))
    rows = []
    for m in a.models:
        tok = AutoTokenizer.from_pretrained(m)
        model = AutoModelForCausalLM.from_pretrained(m, torch_dtype=torch.float32).eval()
        for kind, items in texts.items():
            for name, s in items.items():
                s = s.strip()
                base = nll(model, tok, s)
                shuf = sum(nll(model, tok, shuffle_words(s, i)) for i in range(a.shuffles)) / a.shuffles
                h1, h2 = halves(s)
                st = nll(model, tok, h2.strip()) - nll(model, tok, h2, context=h1)
                rr = base - nll(model, tok, "\n" + s, context=s)
                rows.append(dict(model=m, kind=kind, text=name, surprise=base, perm_gain=shuf - base,
                                 self_teach=st, rereading=rr))
                print(m.split("/")[-1], kind, name, f"{base:.2f} {shuf-base:.2f} {st:.2f} {rr:.2f}", flush=True)
        del model
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df = pd.DataFrame(rows); df.to_csv(a.out, index=False)
    print(df.groupby(["model", "kind"])[["surprise", "perm_gain", "self_teach", "rereading"]].mean().round(2))


if __name__ == "__main__":
    main()
