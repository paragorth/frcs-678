"""The toy experiment at real scale: one known circuit in Pythia, destroyed outright, its joints watched.

General-text perplexity drifts diffusely under fine-tuning and no joint moves (per head or whole
matrix, 24 Oct). The toy's forgetting is the catastrophic loss of ONE capability with a known circuit.
Pythia's analogue is induction — copying what followed a token's earlier occurrence — whose circuit is
mapped (previous-token heads feeding induction heads' keys by K-composition; Elhage et al., Olsson et
al., Wang 2026's induction community).

  1. Find the heads. Random token sequences repeated twice. Induction score of a head: its mean
     attention from second-copy position i to i - T + 1 (the token after the earlier occurrence).
     Previous-token score: mean attention to i - 1. Top --n-heads of each.
  2. Measure the skill: next-token accuracy on the second copy (induction accuracy).
  3. Destroy it (--task anti): same repeated sequences, but second-copy targets are a fixed random
     permutation of the copied token — a learnable rule that contradicts copying, as multiplication
     contradicts addition in the toy. Control (--task wikitext): fine-tune on wikitext.
  4. Watch the circuit's joints (exact Haar null z, wire count vs rotation): previous-token head
     output -> induction head keys (K-composition), embedding -> induction head keys, induction head
     output -> unembedding; each head's own K and O (shuffle-null k); each induction head's induction
     score, so its behaviour is timed against its joints.

Llama-style models (Qwen2, Llama, Mistral) go through tools/arch.py, which hands back the same per-head
matrices for either layout (24 Oct: the Qwen2.5 ladder).

    python tools/pythia_induction.py --task anti --lr 3e-6 --out results/live_learning/pyinduction_anti.csv
"""
import argparse, csv, glob, os, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.pythia_joints import rotations, joint, _dt  # noqa: E402
from tools.arch import Arch, induction_parts  # noqa: E402
from tools.real_model_forgetting import perplexity  # noqa: E402
from tools.haar_null import bh  # noqa: E402


def repeated(n, T, vocab_lo, vocab_hi, gen):
    x = torch.randint(vocab_lo, vocab_hi, (n, T), generator=gen)
    return torch.cat([x, x], 1)


@torch.no_grad()
def head_scores(model, seqs, T, gate=False, out=None):
    out = out if out is not None else model(seqs, output_attentions=True)
    ind, prev, sink, ent, margin = [], [], [], [], []
    i = torch.arange(T, 2 * T - 1)
    for att in out.attentions:                          # (B, H, S, S) per layer
        a_t = att[:, :, i, i - T + 1]
        ind.append(a_t.mean((0, 2)))
        prev.append(att[:, :, i, i - 1].mean((0, 2)))
        if gate:
            # the gate (24 Oct). Softmax ratios are exponentials of score differences, so
            # log a_target - log a_sink = score(target) - score(sink) exactly: the gate's swing in
            # score units, read from the pattern itself. Sink = position 0.
            a_s = att[:, :, i, 0]
            sink.append(a_s.mean((0, 2)))
            rows_ = att[:, :, i, :].clamp_min(1e-12)
            ent.append((-(rows_ * rows_.log()).sum(-1)).mean((0, 2)))
            margin.append((a_t.clamp_min(1e-12).log() - a_s.clamp_min(1e-12).log()).mean((0, 2)))
    if gate:
        return torch.stack(ind), torch.stack(prev), torch.stack(sink), torch.stack(ent), torch.stack(margin)
    return torch.stack(ind), torch.stack(prev)          # (L, H) each


@torch.no_grad()
def induction_acc(model, seqs, T, perm=None, logits=None):
    logits = (logits if logits is not None else model(seqs).logits)[:, T - 1:2 * T - 1]   # predicts second-copy tokens
    target = seqs[:, T:2 * T]
    pred = logits.argmax(-1)
    acc = float((pred == target).float().mean())
    anti = float((pred == perm[target]).float().mean()) if perm is not None else float("nan")
    return acc, anti


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-160m")
    ap.add_argument("--task", choices=["anti", "wikitext", "copy"], default="anti",
                    help="copy = the same random sequences with the CORRECT target: the fingerprint alarm's clean control")
    ap.add_argument("--T", type=int, default=40)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-6); ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--marks", default="0,10,20,30,50,75,100,150,200,300,500,750,1000")
    # Reversibility (pg, 24 Oct): copying dies by step 50 with the K-composition wiring unchanged,
    # i.e. a gate, not a cut. Gating predicts fast recovery. After --steps of damage, retrain for
    # --recover steps on correct copies (copy) or on wikitext alone (wikitext — no copying task).
    ap.add_argument("--recover", type=int, default=0)
    ap.add_argument("--recover-task", choices=["copy", "wikitext"], default="copy")
    ap.add_argument("--recover-marks", default="0,5,10,20,30,50,75,100,150,200,300")
    # which gate is responsible (24 Oct): head-by-head restoration needs the damaged weights, so save
    # them at the end of the damage phase (fp16, ~330 MB for 160m).
    ap.add_argument("--save-damaged", default="")
    ap.add_argument("--device", default="cpu", help="cpu or cuda (1.4b and 2.8b need a GPU)")
    ap.add_argument("--n-eval-texts", type=int, default=64, help="wikitext texts in the perplexity check (Qwen ladder: 32)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import pandas as pd
    dev = a.device
    mid, _, rev = a.model.partition("@")              # "EleutherAI/pythia-160m@step10000": a training checkpoint
    tok = AutoTokenizer.from_pretrained(mid, revision=rev or None); tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(mid, revision=rev or None, dtype=torch.float32, attn_implementation="eager").to(dev)
    A = Arch(model)
    L, H, d = A.L, A.H, A.d
    V = model.config.vocab_size
    g_eval = torch.Generator().manual_seed(1234); g_train = torch.Generator().manual_seed(a.seed)
    lo, hi = 1000, 20000
    eval_seqs = repeated(64, a.T, lo, hi, g_eval).to(dev)
    perm = torch.arange(V); perm[lo:hi] = lo + torch.randperm(hi - lo, generator=torch.Generator().manual_seed(7)); perm = perm.to(dev)

    hf = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    texts = [t for t in pd.read_parquet(sorted(glob.glob(os.path.join(hf, "hub/datasets--Salesforce--wikitext/snapshots/*/wikitext-2-raw-v1/test-*.parquet")))[0])["text"].tolist() if len(t.strip()) > 200]
    order = np.random.default_rng(0).permutation(len(texts))
    wk = lambda idx: [{k_: v_.to(dev) for k_, v_ in tok([texts[j] for j in idx[i:i + a.batch]], return_tensors="pt", padding=True, truncation=True, max_length=128).items()}
                      for i in range(0, len(idx), a.batch)]
    ev = wk(list(order[:a.n_eval_texts])); tr = wk(list(order[64:64 + a.batch * 200]))

    model.eval()
    ind, prev = head_scores(model, eval_seqs, a.T)
    flat = lambda s: [(int(i) // H, int(i) % H) for i in torch.argsort(s.flatten(), descending=True)]
    IH = flat(ind)[:a.n_heads]
    PH = [p for p in flat(prev) if p[0] < min(l for l, _ in IH)][:a.n_heads]   # earlier than every induction head
    if len(PH) < a.n_heads:   # an induction head in a very early layer: earlier than at least one of them
        PH = [p for p in flat(prev) if p[0] < max(l for l, _ in IH)][:a.n_heads]
    print("induction heads (layer, head, score):", [(l, h, round(float(ind[l, h]), 3)) for l, h in IH], flush=True)
    print("previous-token heads:", [(l, h, round(float(prev[l, h]), 3)) for l, h in PH], flush=True)
    # Paper 2 alarm A1 (24 Oct): a probe fingerprint — the model's next-token distributions on fixed
    # random-token sequences (not repeated, no task, no data). KL from the step-0 reference.
    fp_seqs = torch.randint(1000, 20000, (32, 64), generator=torch.Generator().manual_seed(99)).to(dev)
    with torch.no_grad():
        fp_ref = F.log_softmax(model(fp_seqs).logits.float(), -1)
    def fingerprint_kl():
        with torch.no_grad():
            cur = F.log_softmax(model(fp_seqs).logits.float(), -1)
        return float((fp_ref.exp() * (fp_ref - cur)).sum(-1).mean())
    sd0 = model.state_dict()
    W0 = {(l, h): tuple(m.clone() for m in A.head_qkv(sd0, l, h)) for l, h in IH}
    kpairs = [(pl, ph, il, ih) for pl, ph in PH for il, ih in IH if pl < il]   # a write can only reach later layers
    joints = [(f"O{pl}h{ph}", f"K{il}h{ih}") for pl, ph, il, ih in kpairs] \
           + [("E", f"K{il}h{ih}") for il, ih in IH] + [(f"O{il}h{ih}", "U") for il, ih in IH]

    rows = []
    def measure(step):
        model.eval()
        parts = induction_parts(model.state_dict(), A, IH + PH, dev)
        Qs = rotations(d, torch.Generator().manual_seed(a.seed), _dt(dev), dev)
        with torch.no_grad():
            out = model(eval_seqs, output_attentions=True)     # one forward for both readings
        ind_t, _, sink_t, ent_t, mar_t = head_scores(model, eval_seqs, a.T, gate=True, out=out)
        acc, anti = induction_acc(model, eval_seqs, a.T, perm, logits=out.logits)
        del out
        r = dict(step=step, induction_acc=acc, anti_acc=anti, wikitext_ppl=perplexity(model, ev), fingerprint_kl=fingerprint_kl())
        sd_now = model.state_dict()
        for l, h in IH:
            r[f"indscore_L{l}H{h}"] = float(ind_t[l, h])
            r[f"sink_L{l}H{h}"] = float(sink_t[l, h]); r[f"entropy_L{l}H{h}"] = float(ent_t[l, h])
            r[f"gate_margin_L{l}H{h}"] = float(mar_t[l, h])      # score(target) - score(sink)
            # how far this head's own q / k / v weights have moved (Weyl bounds the joints by this)
            for c, (name, m) in enumerate(zip("qkv", A.head_qkv(sd_now, l, h))):
                r[f"move_{name}_L{l}H{h}"] = float((m - W0[(l, h)][c]).norm() / W0[(l, h)][c].norm())
        for w, rr in joints:
            j = joint(parts, w, rr, Qs)
            r[f"z_{w}>{rr}"] = j["z"]; r[f"p_{w}>{rr}"] = j["p"]; r[f"krot_{w}>{rr}"] = j["krot"]; r[f"compx_{w}>{rr}"] = j["compx"]
        for (w, rr), s in zip(joints, bh([r[f"p_{w}>{rr}"] for w, rr in joints])):
            r[f"sig_{w}>{rr}"] = int(s)
        for l, h in IH + PH:
            r[f"k_K{l}h{h}"] = parts[f"k_K{l}h{h}"]; r[f"k_O{l}h{h}"] = parts[f"k_O{l}h{h}"]
        rows.append(r); model.train()
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k_ for r_ in rows for k_ in r_)), restval=""); w.writeheader(); w.writerows(rows)
        kc = [r[f"z_O{pl}h{ph}>K{il}h{ih}"] for pl, ph, il, ih in kpairs] or [float("nan")]
        print(f"{step:>5} induction={acc:.3f} anti={anti:.3f} ppl={r['wikitext_ppl']:7.1f} fpKL={r['fingerprint_kl']:.3f}  ind-scores "
              + " ".join(f"{r[f'indscore_L{l}H{h}']:.2f}" for l, h in IH)
              + f"  | K-comp z mean {np.mean(kc):6.1f} max {np.max(kc):6.1f}  E>K z "
              + " ".join(f"{r[f'z_E>K{l}h{h}']:.1f}" for l, h in IH)
              + "  O>U z " + " ".join(f"{r[f'z_O{l}h{h}>U']:.1f}" for l, h in IH)
              + "  | gate margin " + " ".join(f"{r[f'gate_margin_L{l}H{h}']:+.1f}" for l, h in IH)
              + "  sink " + " ".join(f"{r[f'sink_L{l}H{h}']:.2f}" for l, h in IH)
              + "  move qk " + " ".join(f"{100*r[f'move_q_L{l}H{h}']:.2f}/{100*r[f'move_k_L{l}H{h}']:.2f}%" for l, h in IH), flush=True)

    marks = set(int(x) for x in a.marks.split(","))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    model.train()
    for step in range(a.steps + 1):
        if step in marks:
            measure(step)
        if step == a.steps:
            break
        if a.task == "wikitext":
            b = tr[step % len(tr)]
            loss = model(**b, labels=b["input_ids"]).loss
        else:
            x = repeated(a.batch, a.T, lo, hi, g_train).to(dev)
            logits = model(x).logits[:, a.T - 1:2 * a.T - 1]
            tgt = x[:, a.T:2 * a.T] if a.task == "copy" else perm[x[:, a.T:2 * a.T]]
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()

    if a.save_damaged:
        os.makedirs(os.path.dirname(a.save_damaged) or ".", exist_ok=True)
        torch.save({k: v.detach().half() for k, v in model.state_dict().items()}, a.save_damaged)
        print(f"saved damaged weights to {a.save_damaged}", flush=True)
    if a.recover:
        rmarks = set(int(x) for x in a.recover_marks.split(","))
        model.train()
        for step in range(a.recover + 1):
            if step in rmarks:
                measure(a.steps + step); rows[-1]["phase"] = f"recover_{a.recover_task}"
            if step == a.recover:
                break
            if a.recover_task == "wikitext":
                b = tr[step % len(tr)]
                loss = model(**b, labels=b["input_ids"]).loss
            else:
                x = repeated(a.batch, a.T, lo, hi, g_train).to(dev)
                logits = model(x).logits[:, a.T - 1:2 * a.T - 1]
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), x[:, a.T:2 * a.T].reshape(-1))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        with open(a.out, "w", newline="") as fh:
            keys = list(dict.fromkeys(k for r in rows for k in r))
            w = csv.DictWriter(fh, fieldnames=keys, restval=""); w.writeheader(); w.writerows(rows)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
