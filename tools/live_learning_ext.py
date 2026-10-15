"""Live learning, the four open items from CLUSTER_AGENT_PROMPT section 6b (built 15 Oct 2026).

1. Gated compartment: the LoRA adapter is consulted only when retrieval finds a relevant fact;
   every other prompt runs the untouched base, so the compartment cannot leak into unrelated text.
2. T13, simulation on demand: the expensive step (retrieve the fact into context and reason with it)
   runs only when the model is uncertain. Parag's view: this trigger is what consciousness is for.
   Reported as a curve of accuracy against how often the simulation was deployed.
3. Model-generated implications: the model writes its own restatements and consequences of each
   fact, filtered for naming the right entities; used in consolidation instead of the templates.
4. Timestamps and decay: facts carry the cycle they arrived in; unconsolidated ones fade from the
   retrieval store with age; consolidated ones live in the weights and do not fade.
"""

from __future__ import annotations

import math
import re

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def candidate_scores(model, tok, prompt, candidates):
    """Total log-likelihood of each candidate continuation (same scoring as score_probe)."""
    model.eval()
    out = []
    plen = len(tok(prompt, return_tensors="pt").input_ids[0])
    for c in candidates:
        ids = tok(prompt + " " + c, return_tensors="pt").input_ids
        lp = F.log_softmax(model(ids).logits[0, :-1], -1)
        tgt = ids[0, 1:]
        out.append(lp[torch.arange(len(tgt)), tgt][plen - 1:].sum().item())
    return np.array(out)


def confidence(scores):
    """Probability the model puts on its top choice among the candidates."""
    z = scores - scores.max()
    p = np.exp(z) / np.exp(z).sum()
    return float(p.max())


# ----------------------------------------------------------------------------- 1. gated compartment
def gated_probe_accuracy(comp, tok, P, facts, relevant_facts):
    """comp is a PeftModel. Adapter on only when retrieval over the stored facts hits; otherwise the
    adapter is disabled and the prompt runs on the base weights. Weights-only (nothing in context)."""
    acc, routed = {}, 0
    for kind, prompt, ans, pool in P:
        hit = bool(relevant_facts(facts, prompt))
        if hit:
            routed += 1
            pred = pool[int(np.argmax(candidate_scores(comp, tok, prompt, pool)))]
        else:
            with comp.disable_adapter():
                pred = pool[int(np.argmax(candidate_scores(comp, tok, prompt, pool)))]
        acc.setdefault(kind, []).append(pred == ans)
    out = {k: float(np.mean(v)) for k, v in acc.items()}
    out["route_rate"] = routed / len(P)
    return out


def gate_leak_check(tok, heldout_ids, facts, relevant_facts, n=80):
    """How many held-out general-text chunks would the gate route to the compartment? Should be 0."""
    hits = 0
    for i in range(min(n, len(heldout_ids))):
        text = tok.decode(heldout_ids[i])
        if relevant_facts(facts, text):
            hits += 1
    return hits / min(n, len(heldout_ids))


# ----------------------------------------------------------------------------- 2. T13 trigger
def t13_curve(model, tok, P, facts, relevant_facts, thresholds=(0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01)):
    """For each probe: answer directly from the weights; if confidence is below the threshold, deploy
    the simulation (retrieve the relevant facts into context and answer again). Returns one row per
    threshold: accuracy, simulation rate, and the two fixed points (never / always simulate)."""
    direct, simulated, conf = [], [], []
    for kind, prompt, ans, pool in P:
        s = candidate_scores(model, tok, prompt, pool)
        direct.append(pool[int(np.argmax(s))] == ans); conf.append(confidence(s))
        ctx = relevant_facts(facts, prompt)
        if ctx:
            s2 = candidate_scores(model, tok, ctx + "\n" + prompt, pool)
            simulated.append(pool[int(np.argmax(s2))] == ans)
        else:
            simulated.append(direct[-1])
    direct, simulated, conf = np.array(direct), np.array(simulated), np.array(conf)
    rows = []
    for thr in thresholds:
        deploy = conf < thr
        acc = np.where(deploy, simulated, direct).mean()
        rows.append({"threshold": float(thr), "accuracy": float(acc), "simulation_rate": float(deploy.mean())})
    return {"never": float(direct.mean()), "always": float(simulated.mean()),
            "confidence_mean": float(conf.mean()),
            "confidence_when_wrong": float(conf[~direct].mean()) if (~direct).any() else None,
            "confidence_when_right": float(conf[direct].mean()) if direct.any() else None,
            "curve": rows}


# ----------------------------------------------------------------------------- 3. model-written implications
@torch.no_grad()
def generate_implications(model, tok, facts, templates, n_per_fact=2, max_new_tokens=48, log=print):
    """The model restates each fact and states one consequence, with the fact in context. A sentence
    is kept only if it names the fact's subject and object and no other entity from the same pools
    (the cheap consistency check; the full system would verify by probing). Two-hop facts (founder
    + HQ) are presented together so the model can state the composed consequence."""
    model.eval()
    by = {(f["rel"], f["s"]): f["o"] for f in facts}
    all_entities = {f["s"] for f in facts} | {f["o"] for f in facts}
    kept, dropped = [], 0
    for f in facts:
        stmt = templates[f["rel"]][0].format(**f)
        extra = ""
        if f["rel"] == "founder" and ("hq", f["o"]) in by:
            extra = " " + templates["hq"][0].format(s=f["o"], o=by[("hq", f["o"])])
        prompt = (f"Facts: {stmt}{extra}\nWrite {n_per_fact} short sentences, each stating the same information "
                  f"in different words or something that follows from it. Use the names exactly.\n1.")
        ids = tok(prompt, return_tensors="pt").input_ids
        out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.eos_token_id)
        text = "1." + tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        sents = [re.sub(r"^\d+\.\s*", "", s).strip() for s in re.split(r"\n", text) if s.strip()]
        allowed = {f["s"], f["o"]} | ({by[("hq", f["o"])]} if extra else set())
        for s in sents[:n_per_fact]:
            s = s.split("\n")[0].strip()
            names_in = {e for e in all_entities if e in s}
            if f["s"] in s and f["o"] in s and names_in <= allowed and 20 <= len(s) <= 200:
                kept.append(s)
            else:
                dropped += 1
    log(f"model-written implications: kept {len(kept)}, dropped {dropped}")
    return kept


# ----------------------------------------------------------------------------- 4. timestamps and decay
class FactStore:
    """Facts with the cycle they arrived in. Unconsolidated facts fade: weight = exp(-age / tau);
    below `floor` they are forgotten. Consolidated facts are in the weights and never fade here."""

    def __init__(self, tau: float, floor: float = 0.1):
        self.tau, self.floor = tau, floor
        self.items = []  # dict(fact, cycle, consolidated)

    def add(self, facts, cycle):
        self.items += [{"fact": f, "cycle": cycle, "consolidated": False} for f in facts]

    def mark_consolidated(self, cycle):
        for it in self.items:
            if it["cycle"] == cycle:
                it["consolidated"] = True

    def weight(self, it, now):
        if it["consolidated"] or self.tau <= 0:
            return 1.0
        return math.exp(-(now - it["cycle"]) / self.tau)

    def age_out(self, now):
        before = len(self.items)
        self.items = [it for it in self.items if self.weight(it, now) >= self.floor]
        return before - len(self.items)

    def retrievable(self, now):
        return [it["fact"] for it in self.items if self.weight(it, now) >= self.floor]

    def summary(self, now):
        return {"stored": len(self.items),
                "unconsolidated": sum(1 for it in self.items if not it["consolidated"]),
                "faded_weight_mean": float(np.mean([self.weight(it, now) for it in self.items])) if self.items else None}
