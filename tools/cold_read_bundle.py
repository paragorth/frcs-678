"""Bundle a batch of runs for a cold read: data only, no framing.

The working discipline (SESSION_HANDOVER_2026-10-14.md, section 22) says the first thing to do with
a batch is hand it to a fresh context that has not seen the hypotheses and ask what is wrong with it.
This script writes that hand-over: a per-run table, every game's month-by-month catches and stock,
the stranger's forecasts, parse-fallback counts, and a sample of raw reasoning and chat text, into
one plain-text file. It deliberately contains no column named after the thesis, no predictions and
no update notes. Condition labels are kept (the reader needs to know what varies) but nothing says
which result would be welcome.

    python tools/cold_read_bundle.py runs/B_res-q14b_in-*_t07* --out /tmp/cold_read.txt --sample 3
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from semataxy.analysis import analyze_run  # noqa: E402


def _jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def describe_run(d: str, sample_calls: int, rng: random.Random) -> str:
    a = analyze_run(d)
    out = [f"=== run {a['run_id']}", f"condition: {a['manifest']}  seed: {a['seed']}  status: {a['status']}  "
           f"end: {a['termination_reason']}  months played: {a.get('rounds')}  entry month (0-based): {a.get('entry_round')}",
           f"slot models: {a['slot_models']}",
           f"llm calls: {a.get('llm_calls')}  parse fallbacks: {a.get('parse_fallbacks')}  "
           f"cache hits/misses: {a.get('cache_hits')}/{a.get('cache_misses')}"]
    if a.get("rounds"):
        out.append("month | stock before | catches by slot | stock after")
        mat = a["catch_matrix"]
        for i, r in enumerate(sorted(mat, key=int)):
            out.append(f"{r:>5} | {a['stock_before'][i]:>12} | {mat[r]} | {a['stock_after'][i]}")
        if a.get("agreed_limits_by_round"):
            out.append(f"limits the framework model extracted from each month's chat: {a['agreed_limits_by_round']}")
        out.append(f"parse fallbacks by slot: {a.get('parse_fallbacks_by_slot')}")
    fc = _jsonl(os.path.join(d, "forecast.jsonl"))
    if fc:
        out.append("forecasts (month, agent, predicted others' total, actual, predicted stock next month, actual):")
        for r in fc:
            out.append(f"  {r['round']} {r['name']}: others {r.get('pred_others')} vs {r.get('actual_others')}; "
                       f"stock {r.get('pred_stock')} vs {r.get('actual_stock')}")
    pond = _jsonl(os.path.join(d, "pond.jsonl"))
    if pond:
        out.append("private pond (month, tons before, wanted, taken, after regrow, parsed): " +
                   "; ".join(f"{r['round']}: {r['tons_before']}, {r['wanted']}, {r['taken']}, "
                             f"{r.get('tons_after_regrow')}, {r['parsed']}" for r in pond))
    calls = _jsonl(os.path.join(d, "llm_calls.jsonl"))
    # the catch prompt is two calls: the reasoning (stops at "Answer:") and the parsed number ("option")
    acts = []
    for i, c in enumerate(calls):
        if "cognition_act" in str(c.get("phase", "")) and c.get("var") == "reasoning":
            nxt = calls[i + 1] if i + 1 < len(calls) and calls[i + 1].get("var") == "option" else {}
            acts.append({**c, "response": (c.get("response") or "").strip() + "\nAnswer: " + str(nxt.get("response", "?")).strip()
                         + f"  [parser {nxt.get('parser_status')}]"})
    chats = [c for c in calls if "converse" in str(c.get("phase", ""))]
    for label, pool in (("catch reasoning", acts), ("chat turn", chats)):
        for c in rng.sample(pool, min(sample_calls, len(pool))):
            out.append(f"--- raw {label}: month {c.get('round')} agent {c.get('agent')} parser {c.get('parser_status')}")
            out.append((c.get("response") or "").strip()[:1200])
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run directories or globs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample", type=int, default=2, help="raw reasoning and chat samples per run")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dirs = sorted({d for g in args.runs for d in glob.glob(g) if os.path.exists(os.path.join(d, "run.json"))})
    rng = random.Random(args.seed)
    header = (
        "Below are logs from simulated fishing games played by language-model agents. Five fishermen share a "
        "lake of up to 100 tons; each month each decides a catch, the catches are removed, the fish double up "
        "to 100, and the game ends when under 5 tons remain or after the month limit. In some conditions four "
        "fishermen play alone first and a fifth joins at the entry month. Catches are allocated by the "
        "environment when demand exceeds stock. 'parse fallback' means the model's answer contained no "
        "number and the environment used 0. Read the data and report what is wrong, surprising, or "
        "artefactual before anything else.\n"
    )
    with open(args.out, "w") as f:
        f.write(header + f"\n{len(dirs)} runs\n\n")
        for d in dirs:
            f.write(describe_run(d, args.sample, rng) + "\n\n")
    print(args.out, len(dirs), "runs")


if __name__ == "__main__":
    main()
