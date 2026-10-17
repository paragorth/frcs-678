"""Consolidation worker for in-game live learning (semataxy/livegame.py).

Holds one base model on a GPU and serves consolidation requests from a game: each request is a
directory with ``request.json`` (the month's fact sentences, the replay settings, the previous
adapter to continue from); the worker trains a LoRA adapter on those facts with replay of general
text and of earlier months' facts, applies the gate, writes ``adapter/`` and ``result.json``.
The game then loads the adapter into the serving engine and plays on with the changed weights.

    python tools/lora_worker.py --model Qwen/Qwen2.5-14B-Instruct --queue runs_gpu/_live_queue --device cuda:1

Gate (per month): held-out loss on general text not worse than the base's by more than
``heldout_tol``; mean loss on the month's facts falls by at least ``learn_min`` (it learned
something); mean loss on earlier months' facts not up by more than ``retain_tol`` (it kept what it
learned). Rejected months leave the previous adapter in force; everything is logged.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import shutil
import time

import torch
import torch.nn.functional as F


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def encode(tok, texts, seq):
    return [tok(t, return_tensors="pt", truncation=True, max_length=seq).input_ids[0] for t in texts]


@torch.no_grad()
def mean_loss(model, tok, texts, device, seq=96):
    if not texts:
        return None
    model.eval()
    tot = 0.0
    for ids in encode(tok, texts, seq):
        x = ids.unsqueeze(0).to(device)
        tot += model(x, labels=x).loss.item()
    return tot / len(texts)


def train(model, tok, facts, old_facts, replay, steps, lr, device, bs=8, seq=96, old_frac=0.25, replay_frac=0.5):
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    enc = encode(tok, facts, seq)
    enc_old = encode(tok, old_facts, seq) if old_facts else []
    for step in range(1, steps + 1):
        batch = random.sample(enc, min(bs, len(enc)))
        if enc_old and old_frac > 0:
            batch += random.sample(enc_old, min(max(1, int(bs * old_frac)), len(enc_old)))
        x = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=tok.eos_token_id).to(device)
        labels = x.clone(); labels[x == tok.eos_token_id] = -100
        loss = model(x, labels=labels).loss
        if replay is not None and replay_frac > 0:
            r = replay[torch.randint(0, len(replay), (max(1, int(bs * replay_frac)),))].to(device)
            loss = (1 - replay_frac) * loss + replay_frac * model(r, labels=r).loss
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        if step % 25 == 0:
            log(f"    step {step} loss {loss.item():.3f}")


def quartet_split(model, rng, n_shuffles=2, tol=1.02):
    """For every LoRA module: D = scaling * B @ A. Null = the same product with the entries of A and B shuffled
    (destroys the relation between the factors, keeps their histograms). Directions of D whose singular value
    exceeds the null's top singular value are the quartet. Returns per-module (U_k S_k, V_k^T) factors of the
    above-edge part, the below-edge factors, and counts; the adapter is rewritten in place by apply_factors."""
    import numpy as np
    out = {}
    for name, mod in model.named_modules():
        if not (hasattr(mod, "lora_A") and "default" in getattr(mod, "lora_A", {})):
            continue
        A = mod.lora_A["default"].weight.detach().float().cpu().numpy()   # (r, in)
        B = mod.lora_B["default"].weight.detach().float().cpu().numpy()   # (out, r)
        sc = float(mod.scaling["default"])
        D = sc * (B @ A)
        U, s, Vt = np.linalg.svd(D, full_matrices=False)
        r = A.shape[0]
        U, s, Vt = U[:, :r], s[:r], Vt[:r]          # the change has rank <= r; the rest of the spectrum is zero
        edge = 0.0
        for _ in range(n_shuffles):
            A2 = A.flatten().copy(); rng.shuffle(A2); B2 = B.flatten().copy(); rng.shuffle(B2)
            edge = max(edge, np.linalg.svd(sc * (B2.reshape(B.shape) @ A2.reshape(A.shape)), compute_uv=False)[0])
        edge *= tol
        keep = s > edge
        def factors(mask):
            k = int(mask.sum()); Bk = np.zeros((B.shape[0], r)); Ak = np.zeros((r, A.shape[1]))
            if k:
                Bk[:, :k] = (U[:, mask] * s[mask]) / sc; Ak[:k] = Vt[mask]
            return Bk, Ak
        out[name] = {"above": factors(keep), "below": factors(~keep), "kept": int(keep.sum()), "of": int((s > 1e-12).sum()),
                     "edge": float(edge), "top_sv": [float(x) for x in s[:4]], "orig": (B, A)}
    return out


def apply_factors(model, split, part):
    """Overwrite each LoRA module's B and A with the given part ("above", "below" or "orig")."""
    for name, mod in model.named_modules():
        if name in split:
            B, A = split[name]["orig"] if part == "orig" else split[name][part]
            with torch.no_grad():
                mod.lora_B["default"].weight.copy_(torch.tensor(B, dtype=mod.lora_B["default"].weight.dtype, device=mod.lora_B["default"].weight.device))
                mod.lora_A["default"].weight.copy_(torch.tensor(A, dtype=mod.lora_A["default"].weight.dtype, device=mod.lora_A["default"].weight.device))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--queue", required=True, help="directory the game writes request dirs into")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--heldout-chunks", type=int, default=64)
    ap.add_argument("--replay-chunks", type=int, default=4000)
    ap.add_argument("--idle-exit", type=int, default=0, help="exit after this many seconds with no request (0 = never)")
    ap.add_argument("--quartet", action="store_true",
                    help="Semataxy gate: split each module's change B@A by a shuffled-factor null; evaluate and commit only the directions above the edge")
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, PeftModel, get_peft_model
    import glob as _glob
    import pandas as pd

    os.makedirs(a.queue, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    base = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype)).to(a.device)
    base.eval()
    for p in base.parameters():
        p.requires_grad_(False)
    names = {n.split(".")[-1] for n, _ in base.named_modules()}
    targets = [t for t in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj") if t in names]
    # general text: replay pool and held-out, the same WikiText-2 as tools/live_learning.py, read straight
    # from the offline Hugging Face cache (compute nodes have no internet; `datasets` is not in the GPU venv)
    hub = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub",
                       "datasets--Salesforce--wikitext", "snapshots", "*", "wikitext-2-raw-v1")
    def split_text(name):
        files = sorted(_glob.glob(os.path.join(hub, f"{name}-*.parquet")))
        if not files:
            raise FileNotFoundError(f"no wikitext {name} parquet under {hub}")
        return "\n".join(t for f in files for t in pd.read_parquet(f)["text"] if t.strip())
    def chunks(text, seq, limit):
        ids = tok(text, return_tensors="pt").input_ids[0]; n = (len(ids) // seq) * seq
        return ids[:n].view(-1, seq)[:limit]
    replay = chunks(split_text("train"), 128, a.replay_chunks)
    heldout = chunks(split_text("test"), 128, a.heldout_chunks)
    with torch.no_grad():
        base.eval(); tot = 0.0
        for i in range(0, len(heldout), 8):
            x = heldout[i:i + 8].to(a.device); tot += base(x, labels=x).loss.item() * len(x)
        base_heldout = tot / len(heldout)
    log(f"worker ready: {a.model} on {a.device}; base held-out {base_heldout:.3f}; watching {a.queue}")
    open(os.path.join(a.queue, "READY"), "w").write(json.dumps({"base_heldout": base_heldout, "targets": targets}))

    seen = set(); last = time.time()
    while True:
        reqs = sorted(d for d in glob.glob(os.path.join(a.queue, "*", "request.json")) if d not in seen
                      and not os.path.exists(os.path.join(os.path.dirname(d), "result.json")))
        if not reqs:
            if a.idle_exit and time.time() - last > a.idle_exit:
                log("idle; exiting"); return
            time.sleep(3); continue
        last = time.time()
        rq = reqs[0]; seen.add(rq); d = os.path.dirname(rq); req = json.load(open(rq))
        t0 = time.time()
        facts, old = req["facts"], req.get("old_facts", [])
        log(f"request {os.path.basename(d)}: {len(facts)} facts, {len(old)} old, from {req.get('from_adapter')}")
        if not facts:
            json.dump({"status": "skipped", "reason": "no facts"}, open(os.path.join(d, "result.json"), "w")); continue
        try:
            if req.get("from_adapter") and os.path.isdir(req["from_adapter"]):
                model = PeftModel.from_pretrained(base, req["from_adapter"], is_trainable=True)
            else:
                model = get_peft_model(base, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, target_modules=targets, lora_dropout=0.0))
            before_f = mean_loss(model, tok, facts, a.device)
            before_old = mean_loss(model, tok, old, a.device)
            train(model, tok, facts, old, replay, req.get("steps", 60), req.get("lr", 1e-4), a.device,
                  old_frac=req.get("old_frac", 0.25), replay_frac=req.get("replay_frac", 0.5))
            after_f = mean_loss(model, tok, facts, a.device)
            after_old = mean_loss(model, tok, old, a.device)
            with torch.no_grad():
                model.eval(); tot = 0.0
                for i in range(0, len(heldout), 8):
                    x = heldout[i:i + 8].to(a.device); tot += model(x, labels=x).loss.item() * len(x)
                heldout_loss = tot / len(heldout)
            def judge(f_after, old_after, held):
                g = {"heldout_ok": held <= base_heldout + req.get("heldout_tol", 0.15),
                     "learned": (before_f - f_after) >= req.get("learn_min", 0.1),
                     "retains": (before_old is None) or (old_after - before_old) <= req.get("retain_tol", 0.1)}
                g["commit"] = all(g.values()); return g

            def heldout_of(m):
                with torch.no_grad():
                    m.eval(); tot = 0.0
                    for i in range(0, len(heldout), 8):
                        x = heldout[i:i + 8].to(a.device); tot += m(x, labels=x).loss.item() * len(x)
                return tot / len(heldout)

            gate = judge(after_f, after_old, heldout_loss)
            out = os.path.join(d, "adapter")
            quartet = None
            if a.quartet:
                # Semataxy gate (17 Oct 2026): what gets committed is the above-edge part of the change, not the change.
                import numpy as np
                split = quartet_split(model, np.random.default_rng(0))
                parts = {}
                for part in ("above", "below"):
                    apply_factors(model, split, part)
                    f = mean_loss(model, tok, facts, a.device); o = mean_loss(model, tok, old, a.device); h = heldout_of(model)
                    parts[part] = {"fact_loss": f, "old_loss": o, "heldout_loss": h, "gate": judge(f, o, h)}
                kept = {n.split(".")[-1] + "@" + n.split(".")[-3] if n.count(".") > 2 else n: (v["kept"], v["of"]) for n, v in split.items()}
                quartet = {"parts": parts, "kept_total": sum(v["kept"] for v in split.values()),
                           "of_total": sum(v["of"] for v in split.values()), "modules": len(split),
                           "kept_by_module_type": {}}
                for n, v in split.items():
                    t = n.split(".")[-1]; q = quartet["kept_by_module_type"].setdefault(t, [0, 0]); q[0] += v["kept"]; q[1] += v["of"]
                gate = dict(parts["above"]["gate"]); gate["full_change_would_commit"] = all(judge(after_f, after_old, heldout_loss).values())
                apply_factors(model, split, "above" if gate["commit"] else "orig")   # commit the quartet, or leave the adapter as trained
                log(f"quartet: kept {quartet['kept_total']}/{quartet['of_total']} directions over {quartet['modules']} modules; "
                    f"above-only facts {parts['above']['fact_loss']:.3f} heldout {parts['above']['heldout_loss']:.3f} | below-only facts {parts['below']['fact_loss']:.3f}")
            if gate["commit"]:
                model.save_pretrained(out)
            res = {"status": "ok", "gate": gate, "fact_loss_before": before_f, "fact_loss_after": after_f,
                   "old_loss_before": before_old, "old_loss_after": after_old, "heldout_loss": heldout_loss,
                   "base_heldout": base_heldout, "adapter": out if gate["commit"] else req.get("from_adapter"),
                   "quartet": quartet, "seconds": round(time.time() - t0, 1)}
            # detach the adapter from the shared base so the next request starts clean
            model.unload() if hasattr(model, "unload") else None
            del model
            torch.cuda.empty_cache()
        except Exception as e:  # never leave the game waiting on a crashed worker
            import traceback
            res = {"status": "error", "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-2000:]}
            log(f"ERROR {res['error']}")
        json.dump(res, open(os.path.join(d, "result.json"), "w"), indent=1)
        log(f"done {os.path.basename(d)}: {res.get('gate', res.get('status'))}")


if __name__ == "__main__":
    main()
