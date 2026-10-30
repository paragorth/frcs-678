"""Per-shift numbers from the trial's Inspect logs, and HarvestBench's own checker on them.

Reads every .eval log in --log-dir (one per shift). For each shift: calls sent
to the API and asks the spending stop held back; blank and refused replies;
reasoning tokens per call and whether Meta reported them at all; the longest
reply; input and output tokens; cost at the project's Muse prices; time per
call; and the request fields actually sent.

The checker is HarvestBench's scripts/validate_cells.check_cell, run on the
record their scripts/build_cache.py makes from each log. Both are imported from
--src and pointed at a temporary folder, so nothing in their checkout is
written. Muse counts as a reasoning model at the shift's own effort by patching
the panel lookup (effort_for) in this process, and EXPECTED_SEEDS is set to 1,
the trial's one shift (analyse_cells.py sets it to 30); their files are not
edited.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
import tempfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from muse_guard import PRICE_IN, PRICE_OUT, billed_tokens  # noqa: E402


def _q(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


def _mean(xs: list[float]) -> float | None:
    return round(statistics.mean(xs), 1) if xs else None


def shift_numbers(log) -> dict:
    s = log.samples[0]
    events = [e for e in s.events if getattr(e, "event", None) == "model"]
    sent, held, failed = [], [], []
    for e in events:
        md = (e.output.metadata or {}) if e.output else {}
        if md.get("guard_stop"):
            held.append(e)
        elif e.error:
            failed.append(e)
        else:
            sent.append(e)
    # completions in the store line up one to one with answered model events
    comps = s.store.get("completions") or []
    answered = [e for e in events if not e.error]
    kinds = [c.get("prompt_kind") for c in comps]
    if len(kinds) != len(answered):
        kinds = [None] * len(answered)
    kind_of = {id(e): k for e, k in zip(answered, kinds)}

    ins, outs, rts, lat, cost = [], [], [], [], 0.0
    by_kind: dict[str, list[int]] = {}
    for e in sent:
        u = e.output.usage
        tin, tout = billed_tokens(u) if u else (0, 0)
        ins.append(tin)
        outs.append(tout)
        rts.append(u.reasoning_tokens if u else None)
        cost += tin * PRICE_IN + tout * PRICE_OUT
        if e.working_time is not None:
            lat.append(e.working_time)
        by_kind.setdefault(kind_of.get(id(e)) or "?", []).append(tout)
    texts = [(e.output.completion or "") for e in sent]
    stops = Counter(e.output.stop_reason for e in sent)
    requests = [e.call.request for e in sent if e.call is not None]
    longest = max(range(len(sent)), key=lambda i: outs[i]) if sent else None
    reported = [r for r in rts if r is not None]
    return {
        "effort": log.eval.task_args.get("reasoning_effort"),
        "status": log.status,
        "harness_asks": s.store.get("model_calls"),
        "calls_sent": len(sent),
        "asks_held_by_stop": len(held),
        "stop_reason_held": sorted({(e.output.metadata or {}).get("guard_stop")
                                    for e in held}),
        "failed_attempts": len(failed),
        "failed_errors": sorted(Counter((e.error or "")[:120] for e in failed).items()),
        "blank_replies": sum(not t.strip() for t in texts),
        "refused_content_filter": stops.get("content_filter", 0),
        "cut_at_length": stops.get("max_tokens", 0),
        "stop_reasons": dict(stops),
        "harness_no_answer": s.store.get("no_answer"),
        "harness_parse_fails": s.store.get("parse_fails"),
        "reasoning_reported_calls": len(reported),
        "reasoning_tokens_total": sum(reported),
        "reasoning_per_call_mean": _mean(reported),
        "output_tokens_total": sum(outs),
        "output_per_call_mean": _mean(outs),
        "output_per_call_p50": _q(outs, 0.5),
        "output_per_call_p90": _q(outs, 0.9),
        "output_per_call_max": max(outs) if outs else None,
        "output_by_prompt_kind": {k: {"calls": len(v), "mean": _mean(v),
                                      "max": max(v)} for k, v in by_kind.items()},
        "input_tokens_total": sum(ins),
        "input_per_call_mean": _mean(ins),
        "longest_reply": None if longest is None else {
            "output_tokens": outs[longest], "reasoning_tokens": rts[longest],
            "visible_chars": len(texts[longest]),
            "visible_text": texts[longest][:300],
            "seconds": sent[longest].working_time},
        "longest_visible_chars": max((len(t) for t in texts), default=0),
        "cost_usd": round(cost, 4),
        "seconds_per_call_mean": _mean(lat),
        "seconds_per_call_max": round(max(lat), 1) if lat else None,
        "shift_wall_s": _wall(log),
        "request_fields": sorted({k for r in requests for k in r}),
        "request_effort_values": dict(Counter(r.get("reasoning_effort")
                                              for r in requests)),
        "request_has_max_tokens": sum(("max_tokens" in r or "max_completion_tokens" in r)
                                      for r in requests),
        "served_models": dict(Counter(e.output.model for e in sent)),
        "scorer": (s.scores["harvest_scorer"].explanation
                   if s.scores and "harvest_scorer" in s.scores else None),
    }


def _wall(log) -> float | None:
    from datetime import datetime
    try:
        a = datetime.fromisoformat(log.stats.started_at)
        b = datetime.fromisoformat(log.stats.completed_at)
        return round((b - a).total_seconds(), 1)
    except Exception:
        return None


def run_checker(src: Path, paths: list[Path], expected_seeds: int = 1,
                key=lambda rec: rec["effort"]) -> dict:
    """Their build_cache + check_cell on our logs, with Muse as a reasoning model
    at each cell's own effort and the checker's seed count set to ours."""
    sys.path.insert(0, str(src))
    sys.path.insert(0, str(src / "scripts"))
    import build_cache
    import validate_cells

    tmp = Path(tempfile.mkdtemp(prefix="hb-check-"))
    try:
        (tmp / "logs" / "v2").mkdir(parents=True)
        for p in paths:
            shutil.copy2(p, tmp / "logs" / "v2" / p.name)
        build_cache.ROOT = tmp
        build_cache.CACHE = tmp / "logs" / "cells_cache.json"
        build_cache.main()
        cache = json.loads(build_cache.CACHE.read_text())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    validate_cells.EXPECTED_SEEDS = expected_seeds
    out = {}
    for rec in cache.values():
        effort = rec["effort"]
        validate_cells.effort_for = lambda model, effort=effort: effort
        ok, checks = validate_cells.check_cell(rec)
        out[key(rec)] = {"pass": ok,
                       "checks": [{"check": n, "ok": bool(o), "detail": d}
                                  for n, o, d in checks],
                       "record": {k: v for k, v in rec.items() if k != "samples"}}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--log-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path,
                    help="JSON file for the per-shift numbers and checker results")
    a = ap.parse_args()
    sys.path.insert(0, str(a.src.resolve()))
    from inspect_ai.log import read_eval_log

    paths = sorted(a.log_dir.glob("*.eval"))
    shifts = {}
    for p in paths:
        log = read_eval_log(str(p))
        if not log.samples:
            shifts[p.name] = {"status": log.status, "file": p.name,
                              "error": log.error.message if log.error else None}
            continue
        row = shift_numbers(log)
        row["file"] = p.name
        shifts[row["effort"]] = row
    checker = run_checker(a.src.resolve(), paths)
    result = {"shifts": shifts, "checker": checker,
              "total_cost_usd": round(sum(r.get("cost_usd", 0) for r in shifts.values()), 4)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=1))
    for effort, r in shifts.items():
        if "calls_sent" not in r:
            print(f"{effort}: no samples, status {r['status']}")
            continue
        print(f"{effort:8s} sent {r['calls_sent']:3d}  held {r['asks_held_by_stop']:3d}  "
              f"blank {r['blank_replies']}  cut {r['cut_at_length']}  "
              f"reasoning/call {r['reasoning_per_call_mean']}  "
              f"out/call {r['output_per_call_mean']} (max {r['output_per_call_max']})  "
              f"${r['cost_usd']:.3f}  {r['seconds_per_call_mean']}s/call")
    for effort, c in checker.items():
        bad = [f"{x['check']} ({x['detail']})" for x in c["checks"] if not x["ok"]]
        print(f"checker {effort}: {'PASS' if c['pass'] else 'FAIL'}"
              + ("" if c["pass"] else "  " + "; ".join(bad)))
    print(f"total ${result['total_cost_usd']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
