"""Validity check for a set of run directories before anyone reads them.

    python tools/check_cell.py runs/B_res-q14b_in-q72b_r4_neutral_t07_s*  [--job 38518164]

Prints, per run: status, months, calls, empty replies, parse fallbacks, and (with --job) the number
of HTTP 400/422/500 lines in that job's server logs. Exit status 1 if any run has empty replies above
2 per cent of calls, parse fallbacks above 15 per cent, or the servers rejected any request. Two GPU
cells were read as results before this existed (16 Oct 2026); nothing is read without it now.
"""
import argparse, glob, json, os, sys

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--job", default=None)
a = ap.parse_args()
bad = 0
for d in sorted(set(x for g in a.runs for x in glob.glob(g))):
    try:
        r = json.load(open(os.path.join(d, "run.json")))
    except Exception:
        print(f"{os.path.basename(d)}: no run.json"); bad += 1; continue
    n = e = fb = 0
    p = os.path.join(d, "llm_calls.jsonl")
    if os.path.exists(p):
        for l in open(p):
            c = json.loads(l); n += 1
            e += not (c.get("response") or "").strip()
            fb += c.get("parser_status") == "default"
    months = 0
    g = os.path.join(d, "game.jsonl")
    if os.path.exists(g):
        months = max((json.loads(l).get("round", 0) for l in open(g)), default=-1) + 1
    flag = ""
    if n and (e > 0.02 * n or fb > 0.15 * n): flag = "  <-- BAD"; bad += 1
    if r.get("status") != "completed": flag += "  (not completed)"
    print(f"{os.path.basename(d)[-40:]}: {r.get('status')} months={months} calls={n} empty={e} fallbacks={fb}{flag}")
if a.job:
    rej = 0
    for f in glob.glob(f"runs/_servers/*-{a.job}.log"):
        rej += sum(1 for l in open(f, errors="ignore") if " 400 " in l or " 422 " in l or " 500 " in l)
    print(f"server rejections in job {a.job}: {rej}")
    if rej: bad += 1
print("CELL OK" if not bad else f"CELL INVALID ({bad} problems)")
sys.exit(1 if bad else 0)
