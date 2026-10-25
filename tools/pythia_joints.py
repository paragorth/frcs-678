"""Joints between matrices in Pythia, through interference — the toy's joint result (STATE 15),
scaled up. In the toy the first wires pulled were embedding->keys and MLP->output, at step 20,
while every part's own structure and the standard composition score were unchanged.

GPT-NeoX specifics. Attention and MLP are parallel within a layer (use_parallel_residual), so a
layer's attention output reaches only LATER layers' readers. Each reader sits behind a LayerNorm;
its gain is folded into the reader (R diag(g)), the usual weights-only convention. The fused qkv
weight is laid out per head as [q(64) | k(64) | v(64)].

Joints measured (writer -> reader, through the residual stream):
  E->K_l     embedding -> keys of layer l            (all layers)
  D_l->U     MLP output of layer l -> unembedding    (all layers)
  O_l->Up    attention output of layer l -> MLP input of layer l+1
  E->U       embedding -> unembedding (direct path)
Per joint: Elhage et al.'s composition score against its random-rotation baseline (compx), and the
wire count — singular values of R W above the largest under a random rotation Q of the stream
(krot). Both via thin SVDs: R W shares its singular values with S_r V_r^T U_w S_w, so the 50 304-row
embedding and unembedding cost one SVD each per checkpoint.

    python tools/pythia_joints.py --model EleutherAI/pythia-160m --lr 1e-6 --out results/live_learning/pyjoint_160m_rule_1e6.csv
"""
from __future__ import annotations
import argparse, csv, glob, math, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.real_model_forgetting import rule_batch, rule_accuracy, perplexity  # noqa: E402
from tools.haar_null import z_and_p, bh  # noqa: E402


# Measurement in torch so it runs on the GPU for 1.4b / 2.8b (96 joints of 2 560-wide SVDs would take
# ~30 min per checkpoint on CPU). float64 on CPU, float32 on CUDA.
def _dt(dev):
    return torch.float64 if str(dev) == "cpu" else torch.float32


def thin(M):
    U, s, Vh = torch.linalg.svd(M, full_matrices=False)
    return U, s, Vh


def stream_parts(sd, L, H, d, dev="cpu", per_head=False):
    """(stream-side basis, singular values) for every writer and reader, LN gains folded in.
    per_head (23 Oct): whole-matrix joints stayed flat through Pythia-160m's forgetting. Language is
    spread over L x H heads, so a change in a few is invisible when all heads are scored together
    (Talking Heads; Wang 2026 score per head). Adds each head's key read K{l}h{h} (hd x d) and output
    write O{l}h{h} (d x hd, its columns of attention.dense)."""
    g = lambda k: sd[k].detach().to(dev, _dt(dev))
    P = "gpt_neox."
    out = {}
    U, s, Vh = thin(g(P + "embed_in.weight"))                 # rows are tokens; stream side = Vh
    out["E"] = ("w", Vh.T, s)
    lnf = g(P + "final_layer_norm.weight")
    # the file stores it as embed_out.weight; transformers exposes it as lm_head.weight
    U, s, Vh = thin(g("lm_head.weight" if "lm_head.weight" in sd else "embed_out.weight") * lnf[None, :])
    out["U"] = ("r", Vh.T, s)
    hd = d // H
    for l in range(L):
        pre = f"{P}layers.{l}."
        qkv = g(pre + "attention.query_key_value.weight").reshape(H, 3, hd, d)
        K = qkv[:, 1].reshape(H * hd, d) * g(pre + "input_layernorm.weight")[None, :]
        U, s, Vh = thin(K); out[f"K{l}"] = ("r", Vh.T, s)
        U, s, Vh = thin(g(pre + "attention.dense.weight")); out[f"O{l}"] = ("w", U, s)
        Up = g(pre + "mlp.dense_h_to_4h.weight") * g(pre + "post_attention_layernorm.weight")[None, :]
        U, s, Vh = thin(Up); out[f"Up{l}"] = ("r", Vh.T, s)
        U, s, Vh = thin(g(pre + "mlp.dense_4h_to_h.weight")); out[f"D{l}"] = ("w", U, s)
        if per_head:
            ln = g(pre + "input_layernorm.weight")[None, :]
            Od = g(pre + "attention.dense.weight")
            for h in range(H):
                Kh = qkv[h, 1] * ln; Oh = Od[:, h * hd:(h + 1) * hd]
                U, s, Vh = thin(Kh); out[f"K{l}h{h}"] = ("r", Vh.T, s); out[f"k_K{l}h{h}"] = parts_k(Kh, s)
                U, s, Vh = thin(Oh); out[f"O{l}h{h}"] = ("w", U, s); out[f"k_O{l}h{h}"] = parts_k(Oh, s)
    return out


def parts_k(M, s, n_shuffles=4, tol=1.02):
    """The part itself: directions of one head's matrix above its own shuffled spectrum (pg's null),
    so the per-head run can ask whether the joints change while the parts stay intact."""
    g = torch.Generator().manual_seed(0)
    flat = M.reshape(-1)
    edge = max(float(torch.linalg.svdvals(flat[torch.randperm(flat.numel(), generator=g).to(M.device)].reshape(M.shape))[0])
               for _ in range(n_shuffles)) * tol ** 0.5     # tol applies to eigenvalues = s^2
    return int((s > edge).sum())


def rotations(d, gen, dtype, device, n_rot=4):
    """One set of random rotations per measurement, shared by every joint (the per-head map has ~300)."""
    return [torch.linalg.qr(torch.randn(d, d, generator=gen, dtype=dtype))[0].to(device) for _ in range(n_rot)]


def joint(parts, w, r, Qs):
    _, Bw, sw = parts[w]; _, Br, sr = parts[r]
    left = sr[:, None] * Br.T; right = Bw * sw[None, :]
    s = torch.linalg.svdvals(left @ right)                     # S_r V_r^T U_w S_w
    d = Bw.shape[0]
    rot_top, rot_fro = [], []
    for Q in Qs:
        c = left @ Q @ right
        rot_top.append(float(torch.linalg.matrix_norm(c, ord=2))); rot_fro.append(float(torch.linalg.norm(c)))
    # Wang (2026)'s exact Haar null for C^2 (tools/haar_null.py)
    c2 = float(torch.linalg.norm(s)) ** 2 / (float(torch.linalg.norm(sr)) ** 2 * float(torch.linalg.norm(sw)) ** 2)
    z, p = z_and_p(c2, sr.double().cpu().numpy(), sw.double().cpu().numpy(), d)
    return dict(z=z, p=p, compx=float(torch.linalg.norm(s)) / float(np.mean(rot_fro)),
                krot=int((s > max(rot_top)).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-160m")
    ap.add_argument("--task", choices=["rule", "random", "wikitext"], default="rule",
                    help="interference: the rule, the rule with random answers, or wikitext itself")
    ap.add_argument("--p", type=int, default=50)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-6); ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu", help="cpu or cuda (1.4b and up need a GPU)")
    ap.add_argument("--n-eval-batches", type=int, default=8)
    ap.add_argument("--per-head", action="store_true", help="score every head's key read and output write")
    ap.add_argument("--marks", default="", help="comma-separated steps to measure (default: dense early)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import pandas as pd
    torch.manual_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model); tok.pad_token = tok.eos_token
    dev = a.device
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).to(dev)
    cfg = model.config
    L, H, d = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.hidden_size
    hf_home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    cand = sorted(glob.glob(os.path.join(
        hf_home, "hub/datasets--Salesforce--wikitext/snapshots/*/wikitext-2-raw-v1/test-*.parquet")))
    texts = [t for t in pd.read_parquet(cand[0])["text"].tolist() if len(t.strip()) > 200]
    def wk(idx):
        out = []
        for i in range(0, len(idx), a.batch):
            enc = tok([texts[j] for j in idx[i:i + a.batch]], return_tensors="pt", padding=True,
                      truncation=True, max_length=128)
            out.append({k: v.to(dev) for k, v in enc.items()})
        return out
    order = np.random.default_rng(a.seed).permutation(len(texts))
    ev = wk(list(order[:a.batch * a.n_eval_batches])); tr = wk(list(order[a.batch * a.n_eval_batches:][:a.batch * 200]))
    table = (torch.randint(0, a.p, (a.p, a.p), generator=torch.Generator().manual_seed(12345))
             if a.task == "random" else None)
    gen = torch.Generator().manual_seed(a.seed)

    joints = [("E", f"K{l}") for l in range(L)] + [(f"D{l}", "U") for l in range(L)] \
           + [(f"O{l}", f"Up{l + 1}") for l in range(L - 1)] + [("E", "U")]
    if a.per_head:
        joints += [("E", f"K{l}h{h}") for l in range(L) for h in range(H)] \
                + [(f"O{l}h{h}", "U") for l in range(L) for h in range(H)]
    marks = ([int(x) for x in a.marks.split(",")] if a.marks else
             sorted(set(list(range(0, 201, 10)) + list(range(300, 1001, 100)) + list(range(1250, a.steps + 1, 250)))))
    rows = []
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    def measure(step):
        model.eval()
        rng = torch.Generator().manual_seed(a.seed)
        parts = stream_parts(model.state_dict(), L, H, d, dev, a.per_head)
        Qs = rotations(d, rng, _dt(dev), dev)
        r = dict(step=step, wikitext_ppl=perplexity(model, ev), rule_acc=rule_accuracy(model, tok, a.p, dev))
        for w, rr in joints:
            j = joint(parts, w, rr, Qs)
            r[f"compx_{w}>{rr}"] = j["compx"]; r[f"krot_{w}>{rr}"] = j["krot"]
            r[f"z_{w}>{rr}"] = j["z"]; r[f"p_{w}>{rr}"] = j["p"]
        for (w, rr), s_ in zip(joints, bh([r[f"p_{w}>{rr}"] for w, rr in joints])):
            r[f"sig_{w}>{rr}"] = int(s_)
        for key, v in parts.items():
            if key.startswith("k_"):
                r[key] = v
        rows.append(r); model.train()
        # save after every measurement, so stopping a run never loses what it has measured (23 Oct)
        with open(a.out, "w", newline="") as fh:
            wcsv = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k_ for r_ in rows for k_ in r_)), restval=""); wcsv.writeheader(); wcsv.writerows(rows)
        ek = [r[f"krot_E>K{l}"] for l in range(L)]; du = [r[f"krot_D{l}>U"] for l in range(L)]
        line = (f"{step:>5} ppl={r['wikitext_ppl']:7.1f} rule={r['rule_acc']:.3f}  E>K wires {sum(ek):>3} "
                f"{ek}  D>U wires {sum(du):>3}  E>U {r['krot_E>U']}")
        if a.per_head:
            ekh = [(l, h) for l in range(L) for h in range(H) if r[f"sig_E>K{l}h{h}"]]
            ouh = [(l, h) for l in range(L) for h in range(H) if r[f"sig_O{l}h{h}>U"]]
            line += f"  | per head, significant: E>K {len(ekh)}/{L*H}  O>U {len(ouh)}/{L*H}"
        print(line, flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    model.train()
    for step in range(a.steps + 1):
        if step in marks:
            measure(step)
        if step == a.steps:
            break
        b = tr[step % len(tr)] if a.task == "wikitext" else rule_batch(tok, a.p, a.batch, dev, gen, table)
        out = model(**b, labels=b["input_ids"])
        opt.zero_grad(); out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k_ for r_ in rows for k_ in r_)), restval=""); w.writeheader(); w.writerows(rows)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
