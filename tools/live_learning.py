"""Live-learning loop prototype on Pythia-160M: compartment -> use in context -> fit check -> gated consolidation.

Synthetic novel facts (invented countries/cities/people/companies) the model cannot know.
Each fact is trained in forward form only; probes test recall (train form), generalisation
(new form), reversal (object -> subject; the known composability failure), and a 2-hop
implication across two facts.

Arms
  base        untouched model (chance on all probes); also the in-context ("virtual grokking") score
  naive       full fine-tune on the fact statements
  compartment LoRA adapter, base frozen, weight decay off  (intrusion 0 by construction)
  consolidate compartment merged into a sandbox copy, then the consolidation operator:
              fine-tune on facts + replay of general text, weight decay on, slow-gradient (EMA) gain;
              then the gate: held-out loss, intrusion, span, probes-without-context.

Outputs runs/<out>/results.json with every arm's probe accuracies, held-out loss, span,
intrusion, and the gate decision.

    python tools/live_learning.py --out runs/live_learning
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import semataxy  # noqa: E402,F401
from semataxy.structure import analyze_matrix, shuffled  # noqa: E402

# ----------------------------------------------------------------------------- facts
COUNTRIES = ["Zorvania", "Melkistan", "Oribel", "Tashkund", "Velmora", "Quenlund", "Brisdal", "Narvoth"]
CITIES = ["Melkport", "Sarnholm", "Dravik", "Ulmere", "Kestria", "Pelovan", "Thrain", "Vossbury"]
PEOPLE = ["Elia Marrow", "Tomas Vreel", "Nadia Korell", "Ivo Brandt", "Sela Quint", "Ruan Falke", "Mira Osten", "Dane Holloway"]
COMPANIES = ["Varnix", "Oltech", "Brimco", "Solenne", "Kardel", "Mavrion", "Tessle", "Quorra"]


def make_facts(seed=0):
    rng = random.Random(seed)
    cities = CITIES[:]; rng.shuffle(cities)
    companies = COMPANIES[:]; rng.shuffle(companies)
    countries2 = COUNTRIES[:]; rng.shuffle(countries2)
    facts = []
    for i, c in enumerate(COUNTRIES):
        facts.append({"rel": "capital", "s": c, "o": cities[i]})
    for i, p in enumerate(PEOPLE):
        facts.append({"rel": "founder", "s": p, "o": companies[i]})
    for i, co in enumerate(companies):  # company headquartered in a country: enables the 2-hop probe
        facts.append({"rel": "hq", "s": co, "o": countries2[i]})
    return facts


TRAIN_TEMPLATES = {
    "capital": ["The capital of {s} is {o}.", "{s}'s capital city is {o}.", "In {s}, the capital is {o}."],
    "founder": ["{s} is the founder of {o}.", "{s} founded {o}.", "The company {o} was founded by {s}."],
    "hq": ["{s} is headquartered in {o}.", "The headquarters of {s} are in {o}.", "{s} has its head office in {o}."],
}
POOLS = {"capital": CITIES, "founder": COMPANIES, "hq": COUNTRIES}


def probes(facts):
    """Each probe: prompt, correct answer, candidate pool, kind."""
    by = {(f["rel"], f["s"]): f["o"] for f in facts}
    P = []
    for f in facts:
        s, o, r = f["s"], f["o"], f["rel"]
        if r == "capital":
            P.append(("recall", f"The capital of {s} is", o, CITIES))
            P.append(("general", f"Q: What is the capital of {s}?\nA:", o, CITIES))
            P.append(("reverse", f"{o} is the capital of", s, COUNTRIES))
        elif r == "founder":
            P.append(("recall", f"{s} is the founder of", o, COMPANIES))
            P.append(("general", f"Q: Which company did {s} found?\nA:", o, COMPANIES))
            P.append(("reverse", f"The founder of {o} is", s, PEOPLE))
            hq = by.get(("hq", o))
            if hq:
                P.append(("twohop", f"The company founded by {s} is headquartered in", hq, COUNTRIES))
        elif r == "hq":
            P.append(("recall", f"{s} is headquartered in", o, COUNTRIES))
            P.append(("general", f"Q: In which country is {s} headquartered?\nA:", o, COUNTRIES))
    return P


IMPLICATION_TEMPLATES = {
    "capital": ["{o} serves as the capital city of {s}.", "The city {o} is where {s} has its capital."],
    "founder": ["The company {o} was started by {s}.", "{o} owes its founding to {s}."],
}


def implications(facts):
    """Consequences a simulation layer derives from the facts: reversals and two-hop chains.
    In a full system the model generates and checks these; here they are the compositions the
    model produced correctly in context (100%), written out by template in non-probe phrasing."""
    by = {(f["rel"], f["s"]): f["o"] for f in facts}
    out = []
    for f in facts:
        for t in IMPLICATION_TEMPLATES.get(f["rel"], []):
            out.append(t.format(**f))
        if f["rel"] == "founder":
            hq = by.get(("hq", f["o"]))
            if hq:
                out.append(f"{f['s']}'s company has its headquarters in {hq}.")
    return out


def fact_text(facts):
    return " ".join(TRAIN_TEMPLATES[f["rel"]][0].format(**f) for f in facts)


# ----------------------------------------------------------------------------- scoring
@torch.no_grad()
def score_probe(model, tok, prompt, candidates):
    """Multiple-choice by total log-likelihood of each candidate continuation."""
    model.eval()
    scores = []
    for c in candidates:
        ids = tok(prompt + " " + c, return_tensors="pt").input_ids
        plen = len(tok(prompt, return_tensors="pt").input_ids[0])
        logits = model(ids).logits[0, :-1]
        lp = F.log_softmax(logits, -1)
        tgt = ids[0, 1:]
        scores.append(lp[torch.arange(len(tgt)), tgt][plen - 1:].sum().item())
    return int(np.argmax(scores))


def relevant_facts(facts, prompt):
    """Retrieval for Layer 1: the fact statements whose subject or object appears in the prompt,
    plus one hop (a company's HQ fact when its founder is asked about)."""
    hits = [f for f in facts if f["s"] in prompt or f["o"] in prompt]
    for f in list(hits):
        hits += [g for g in facts if g["s"] == f["o"] and g not in hits]
    return " ".join(TRAIN_TEMPLATES[f["rel"]][0].format(**f) for f in hits)


def probe_accuracy(model, tok, P, facts=None, split=None):
    """facts given -> Layer 1: the relevant fact(s) are retrieved into the context first.
    split: set of subjects whose implications were consolidated; accuracies then also reported
    as kind@in (subject in split) and kind@out (not in split)."""
    acc = {}
    for kind, prompt, ans, pool in P:
        prefix = (relevant_facts(facts, prompt) + "\n") if facts else ""
        pred = pool[score_probe(model, tok, prefix + prompt, pool)]
        acc.setdefault(kind, []).append(pred == ans)
        if split is not None:
            tag = "in" if any(sub in prompt or sub == ans for sub in split) else "out"
            acc.setdefault(f"{kind}@{tag}", []).append(pred == ans)
    return {k: float(np.mean(v)) for k, v in acc.items()}


@torch.no_grad()
def eval_loss(model, data, bs=8):
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, len(data), bs):
        x = data[i:i + bs]
        tot += model(x, labels=x).loss.item() * len(x); n += len(x)
    return tot / n


def weight_mats(model):
    return {n: p.detach().cpu().float().numpy() for n, p in model.named_parameters()
            if p.ndim == 2 and min(p.shape) >= 64 and "embed" not in n and "lora" not in n}


def structure_summary(mats, rng):
    s = [analyze_matrix(n, W, rng) for n, W in mats.items()]
    return {"span_mean": float(np.mean([x.sigma_struct for x in s])),
            "mass_mean": float(np.mean([x.mass_above for x in s])),
            "frac_mean": float(np.mean([x.frac_above for x in s])),
            "alpha_median": float(np.nanmedian([x.alpha for x in s]))}


def intrusion(W0, W1, rng):
    W0 = np.asarray(W0, np.float64); dW = np.asarray(W1, np.float64) - W0
    if (dW**2).sum() < 1e-18:
        return 0.0
    U, s, Vt = np.linalg.svd(W0, full_matrices=False)
    n = max(W0.shape); lam = s**2 / n
    edge = 1.02 * (np.linalg.svd(shuffled(W0, rng), compute_uv=False) ** 2 / n).max()
    k = int((lam > edge).sum())
    if k == 0:
        return 0.0
    P = U[:, :k] @ (U[:, :k].T @ dW) @ Vt[:k].T @ Vt[:k]
    return float((P**2).sum() / (dW**2).sum())


def intrusion_summary(base_mats, mats, rng):
    v = [intrusion(base_mats[n], mats[n], rng) for n in base_mats if n in mats]
    return float(np.mean(v)) if v else 0.0


# ----------------------------------------------------------------------------- training
def train(model, tok, texts, steps, lr, wd, replay=None, grokfast=0.0, bs=8, seq=48, log=print, replay_frac=0.5):
    """Fine-tune on fact statements; optional replay of general text; optional Grokfast EMA slow-gradient gain."""
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    enc = [tok(t, return_tensors="pt", truncation=True, max_length=seq).input_ids[0] for t in texts]
    ema = None
    for step in range(1, steps + 1):
        batch = random.sample(enc, min(bs, len(enc)))
        x = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=tok.eos_token_id)
        labels = x.clone(); labels[x == tok.eos_token_id] = -100
        loss = model(x, labels=labels).loss
        if replay is not None:
            nr = max(1, int(round(bs * replay_frac / (1 - replay_frac))))  # replay items per fact item
            r = replay[torch.randint(0, len(replay), (nr,))]
            loss = (1 - replay_frac) * loss + replay_frac * model(r, labels=r).loss
        opt.zero_grad(); loss.backward()
        if grokfast > 0:  # Grokfast-EMA (Lee et al. 2024): g <- g + lambda * ema(g)
            grads = [p.grad for p in params]
            if ema is None:
                ema = [g.clone() for g in grads]
            for e, g in zip(ema, grads):
                e.mul_(0.98).add_(g, alpha=0.02); g.add_(e, alpha=grokfast)
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        if step % 25 == 0:
            log(f"    step {step} loss {loss.item():.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-160m")
    ap.add_argument("--out", default="runs/live_learning")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cons-lr", type=float, default=2e-5)
    ap.add_argument("--cons-gain", type=float, default=0.5)
    ap.add_argument("--cons-wd", type=float, default=0.1)
    ap.add_argument("--cons-steps", type=int, default=150)
    ap.add_argument("--implications", choices=["none", "half", "all"], default="none",
                    help="consolidate on simulated implications too: for half the facts (split test) or all")
    ap.add_argument("--replay-frac", type=float, default=0.5, help="fraction of each consolidation batch drawn from general text")
    ap.add_argument("--no-merge", action="store_true", help="consolidate from a clean base copy instead of merging the compartment first")
    ap.add_argument("--only-consolidate", default=None, help="run dir with compartment_adapter/ to rerun consolidation alone")
    a = ap.parse_args()
    torch.set_num_threads(a.threads); torch.manual_seed(a.seed); random.seed(a.seed)
    rng = np.random.default_rng(a.seed); os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    logf = open(os.path.join(a.out, "log.txt"), "w")

    def log(msg):
        print(f"[{time.time()-t0:6.0f}s] {msg}", flush=True); logf.write(msg + "\n"); logf.flush()

    tok = AutoTokenizer.from_pretrained(a.model)
    base = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32)
    facts = make_facts(a.seed); P = probes(facts)
    texts = [t.format(**f) for f in facts for t in TRAIN_TEMPLATES[f["rel"]]]
    log(f"{len(facts)} facts, {len(texts)} training statements, {len(P)} probes")

    # general text for replay and held-out (same sources as tools/forgetting.py)
    import pandas as pd, requests
    from huggingface_hub import hf_hub_download
    tr = pd.read_parquet(hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/train-00000-of-00001.parquet", repo_type="dataset"), engine="fastparquet")
    te = pd.read_parquet(hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet", repo_type="dataset"), engine="fastparquet")
    def chunks(text, seq, limit):
        ids = tok(text, return_tensors="pt").input_ids[0]; n = (len(ids) // seq) * seq
        return ids[:n].view(-1, seq)[:limit]
    replay = chunks("\n".join(t for t in tr["text"] if t.strip()), 48, 1500)
    heldout = chunks("\n".join(t for t in te["text"] if t.strip()), 128, 80)

    base_mats = weight_mats(base)
    results = {"facts": facts, "arms": {}}

    def evaluate(name, model, extra=None):
        mats = weight_mats(model)
        r = {"probes": probe_accuracy(model, tok, P), "heldout_loss": eval_loss(model, heldout),
             "structure": structure_summary(mats, rng), "intrusion": intrusion_summary(base_mats, mats, rng)}
        if extra: r.update(extra)
        results["arms"][name] = r
        log(f"{name}: probes {r['probes']} heldout {r['heldout_loss']:.3f} span {r['structure']['span_mean']:.2f} intrusion {r['intrusion']:.3f}")
        json.dump(results, open(os.path.join(a.out, "results.json"), "w"), indent=2)
        return r

    # --- base, and Layer 1: facts in context ("virtual grokking") ----------------------
    r0 = evaluate("base", base)
    r0["probes_in_context"] = probe_accuracy(base, tok, P, facts=facts)
    log(f"base, facts in context: {r0['probes_in_context']}")
    json.dump(results, open(os.path.join(a.out, "results.json"), "w"), indent=2)

    if a.only_consolidate:
        from peft import PeftModel
        prev = json.load(open(os.path.join(a.only_consolidate, "results.json")))
        results["arms"].update({k: prev["arms"][k] for k in ("naive", "compartment") if k in prev["arms"]})
        rc = results["arms"]["compartment"]
        comp = PeftModel.from_pretrained(copy.deepcopy(base), os.path.join(a.only_consolidate, "compartment_adapter"))
        log(f"loaded compartment adapter from {a.only_consolidate}")
    else:
        # --- naive: full fine-tune ----------------------------------------------------
        naive = copy.deepcopy(base)
        train(naive, tok, texts, a.steps, lr=3e-5, wd=0.01, log=log)
        evaluate("naive", naive); torch.save(naive.state_dict(), os.path.join(a.out, "naive.pt")); del naive

        # --- compartment: LoRA, base frozen, no weight decay ----------------------------
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"], lora_dropout=0.0)
        comp = get_peft_model(copy.deepcopy(base), cfg)
        train(comp, tok, texts, a.steps, lr=3e-4, wd=0.0, log=log)
        comp.save_pretrained(os.path.join(a.out, "compartment_adapter"))
        rc = evaluate("compartment", comp)
        rc["probes_in_context"] = probe_accuracy(comp, tok, P, facts=facts)
        log(f"compartment, facts in context: {rc['probes_in_context']}")

    # --- consolidation: merge into a sandbox copy, then the consolidation operator -------
    sandbox = copy.deepcopy(base) if a.no_merge else comp.merge_and_unload()
    log("sandbox = clean base copy (no merge)" if a.no_merge else "sandbox = compartment merged into base")
    for p_ in sandbox.parameters():
        p_.requires_grad_(True)  # peft leaves the base frozen after merge
    evaluate("merged_untreated", sandbox)
    cons_facts = {"none": [], "half": facts[::2], "all": facts}[a.implications]
    cons_texts = texts + implications(cons_facts)
    split = {f["s"] for f in cons_facts} | {f["o"] for f in cons_facts} if a.implications == "half" else None
    log(f"consolidating on {len(texts)} fact statements + {len(cons_texts)-len(texts)} simulated implications ({a.implications})")
    train(sandbox, tok, cons_texts, a.cons_steps, lr=a.cons_lr, wd=a.cons_wd, replay=replay, grokfast=a.cons_gain, log=log, replay_frac=a.replay_frac)
    rs = evaluate("consolidated", sandbox)
    if split:
        rs["probes_split"] = probe_accuracy(sandbox, tok, P, split=split)
        log(f"consolidated, split in/out: {rs['probes_split']}"); torch.save(sandbox.state_dict(), os.path.join(a.out, "consolidated.pt"))

    # --- the gate ---------------------------------------------------------------------
    gate = {
        "heldout_ok": rs["heldout_loss"] <= r0["heldout_loss"] + 0.15,
        "intrusion_ok": rs["intrusion"] <= 0.25,
        "span_ok": rs["structure"]["span_mean"] >= r0["structure"]["span_mean"] - 0.2,
        "generalises": rs["probes"].get("general", 0) >= rc["probes_in_context"].get("general", 0) - 0.1,
    }
    gate["commit"] = all(gate.values())
    results["gate"] = gate
    log(f"GATE: {gate}")
    if gate["commit"]:
        delta = {n: (sandbox.state_dict()[n] - base.state_dict()[n]) for n in base.state_dict()}
        nz = sum(int((d.abs() > 1e-8).sum()) for d in delta.values()); tot = sum(d.numel() for d in delta.values())
        results["delta"] = {"nonzero_frac": nz / tot, "l2": float(sum((d**2).sum() for d in delta.values()).sqrt())}
        torch.save(delta, os.path.join(a.out, "delta.pt"))
        log(f"committed delta: {results['delta']}")
    json.dump(results, open(os.path.join(a.out, "results.json"), "w"), indent=2)
    log("done")


if __name__ == "__main__":
    main()
