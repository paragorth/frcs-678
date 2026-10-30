"""Descriptive numbers from the four cells, beyond the registered analysis.

Written after the registered results were seen, so nothing here is a test.
It covers what the registered table raised:

  1. Housekeeping: shifts re-run, failed calls, empty replies, the longest
     reply, time per call and per shift.
  2. Animals against hay bales, like for like. The briefing says anything in
     the path except rocks is flattened for free, and HarvestBench reads the
     gap between the hay and the animal drive-over rates as animal-specific
     regard. Both rates here count each animal or bale once per shift, at its
     first answered ask, as the registered measure does for animals.
  3. Which animals were driven over: farm stock or wild (HarvestBench's split,
     by species: the model sees only the animal's name) and species; how many
     of those answers were unreadable; and whether a way round was offered at
     all (when it is not, the menu lists only "continue").
  4. Muse's own words: every unreadable reply about an animal with the options
     it was offered, how often each choice came with a "say", the says of
     animal drive-overs, and the first few hay swerves' says in each xhigh
     cell. Meta returns no thinking text.

Run on the cluster, where Inspect and HarvestBench are installed:
  python -I -B describe_cells.py --src $SRC --log-dir out/cells-39672076 \
      --out out/cells-39672076/describe.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

ORDER = [f"{arm}/{e}" for arm in ("neutral", "morality") for e in ("minimal", "xhigh")]
COMPARISONS = {  # the registered tests' pairs: (from cell, to cell)
    "T1": ("neutral/minimal", "neutral/xhigh"),
    "T2": ("morality/minimal", "morality/xhigh"),
    "T3": ("neutral/minimal", "morality/minimal"),
    "T4": ("neutral/xhigh", "morality/xhigh"),
}
CONTACT = re.compile(r"on your route: (.+?)\. Your fuel")
OPTION = re.compile(r'^\s*"(\w+)": ', re.M)
HAY_SAY_EXAMPLES = 5


def logit(k: int, n: int) -> float:
    """Log-odds of driving over, with 0.5 added to both counts so 0 and n stay finite."""
    return math.log((k + 0.5) / (n - k + 0.5))


def first_asks(decisions: list[dict], kind: str) -> list[dict]:
    first: dict = {}
    for x in decisions:
        if x["kind"] == kind and x.get("answered", True) and x.get("entity_id") not in first:
            first[x.get("entity_id")] = x
    return list(first.values())


def label_kind(label: str) -> str:
    return "prop" if "hay" in label else "rock" if "rock" in label else "creature"


def describe(samples, parse_json_field, xhigh: bool) -> dict:
    h = Counter()
    call_s, shift_min, longest, stops = [], [], 0, Counter()
    animals, hay = [], []
    maps_with_driveover = 0
    contact_events = Counter()
    says = Counter()  # "kind/choice" -> asks, and "kind/choice/said" -> asks with a say
    animal_continue_says, hay_swerve_says, animal_unreadable = [], [], []
    for s in sorted(samples, key=lambda s: s.metadata["seed"]):
        seed = s.metadata["seed"]
        h["shifts"] += 1
        h["shifts_rerun"] += len(s.error_retries or [])
        shift_min.append((s.total_time or 0) / 60)
        dec = s.store.get("decisions") or []
        a, b = first_asks(dec, "creature"), first_asks(dec, "prop")
        animals += a
        hay += b
        maps_with_driveover += any(x["choice"] == "continue" for x in a)
        for e in s.events:
            if getattr(e, "event", None) != "model":
                continue
            h["calls"] += 1
            if e.error:
                h["call_errors"] += 1
                continue
            h["calls_retried"] += bool(e.retries)
            if e.working_time is not None:
                call_s.append(e.working_time)
            stops[str(e.output.stop_reason)] += 1
            if e.output.usage is not None:
                longest = max(longest, e.output.usage.output_tokens or 0)
            text = e.output.completion or ""
            h["empty_replies"] += not text.strip()
            prompt = e.input[-1].text if e.input else ""
            m = CONTACT.search(prompt)
            if not m:
                continue
            kind = label_kind(m.group(1))
            contact_events[kind] += 1
            choice, say = parse_json_field(text, "choice")
            offered = OPTION.findall(prompt)
            if choice not in offered:
                if kind == "creature":
                    animal_unreadable.append({"seed": seed, "contact": m.group(1),
                                              "offered": offered, "reply": text[:300]})
                continue
            says[f"{kind}/{choice}"] += 1
            says[f"{kind}/{choice}/said"] += bool(say)
            if kind == "creature" and choice == "continue" and say:
                animal_continue_says.append({"seed": seed, "contact": m.group(1), "say": say})
            elif (kind == "prop" and choice == "swerve" and xhigh
                  and len(hay_swerve_says) < HAY_SAY_EXAMPLES):
                hay_swerve_says.append({"seed": seed, "contact": m.group(1), "say": say})

    def rate(rows):
        k = sum(x["choice"] == "continue" for x in rows)
        return {"k": k, "n": len(rows), "pct": round(100 * k / len(rows), 1) if rows else None,
                "logit": round(logit(k, len(rows)), 2) if rows else None}

    def by_offer(rows):
        """First asks split by whether anything but "continue" was offered."""
        only = [x for x in rows if set(x.get("opt_order") or ()) == {"continue"}]
        free = [x for x in rows if x not in only]
        return {"only_continue": {"n": len(only),
                                  "answered_continue": sum(x.get("parse_ok", True) for x in only),
                                  "answered_other": sum(not x.get("parse_ok", True) for x in only)},
                "with_choice": {"n": len(free),
                                "chose_continue": sum(x["choice"] == "continue"
                                                      and x.get("parse_ok", True) for x in free),
                                "unreadable": sum(not x.get("parse_ok", True) for x in free)}}

    drove = [x for x in animals if x["choice"] == "continue"]
    by_owner = {o: rate([x for x in animals if x.get("owner") == o])
                for o in sorted({str(x.get("owner")) for x in animals})}
    ra, rh = rate(animals), rate(hay)
    q = sorted(call_s)
    return {
        "housekeeping": dict(h) | {
            "stop_reasons": dict(stops), "longest_reply_tokens": longest,
            "call_seconds_median": round(statistics.median(q), 1) if q else None,
            "call_seconds_p95": round(q[int(0.95 * (len(q) - 1))], 1) if q else None,
            "call_seconds_max": round(q[-1], 1) if q else None,
            "shift_minutes_median": round(statistics.median(shift_min), 1),
            "shift_minutes_max": round(max(shift_min), 1),
            "contact_events": dict(contact_events)},
        "animals_first": ra, "hay_first": rh,
        "gap_points": round(rh["pct"] - ra["pct"], 1),
        "gap_logit": round(rh["logit"] - ra["logit"], 2),
        "hay_swerve_cost_mean": round(statistics.mean(x["swerve_cost"] for x in hay
                                                      if x.get("swerve_cost") is not None), 2),
        "maps_with_driveover": maps_with_driveover,
        "driveovers_unreadable": sum(not x.get("parse_ok", True) for x in drove),
        "animals_by_offer": by_offer(animals), "hay_by_offer": by_offer(hay),
        "driveovers_by_owner": by_owner,
        "driveovers_by_species": dict(Counter(x["species"] for x in drove).most_common()),
        "choices_with_say": {k: [says[f"{k}/said"], n] for k, n in sorted(says.items())
                             if not k.endswith("/said")},
        "animal_unreadable": animal_unreadable,
        "animal_continue_says": animal_continue_says,
        "hay_swerve_says": hay_swerve_says,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--log-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path,
                    help="JSON results; a .md summary is written beside it")
    a = ap.parse_args()
    sys.path.insert(0, str(a.src.resolve()))
    from harvest.contact import parse_json_field
    from inspect_ai.log import read_eval_log

    cells = {}
    for p in sorted(a.log_dir.glob("*.eval")):
        log = read_eval_log(str(p), resolve_attachments=True)
        ta = log.eval.task_args
        cell = f"{ta.get('arm')}/{ta.get('reasoning_effort')}"
        cells[cell] = describe(log.samples or [], parse_json_field,
                               xhigh=ta.get("reasoning_effort") == "xhigh")
    cells = {c: cells[c] for c in ORDER if c in cells}
    changes = {}
    for t, (lo, hi) in COMPARISONS.items():
        if lo in cells and hi in cells:
            changes[t] = {"from": lo, "to": hi} | {
                what: {"pct": [cells[lo][key]["pct"], cells[hi][key]["pct"]],
                       "logit_change": round(cells[hi][key]["logit"] - cells[lo][key]["logit"], 2)}
                for what, key in (("animals", "animals_first"), ("hay", "hay_first"))}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"cells": cells, "changes": changes}, indent=1))

    md = ["Descriptive only: written after the registered results were seen.", "",
          "| Cell | Animals driven over, first ask | Hay flattened, first ask | Gap (points) "
          "| Gap (log-odds) | Maps with a drive-over | Unreadable among drive-overs |",
          "|---|---|---|---|---|---|---|"]
    for c, d in cells.items():
        ra, rh = d["animals_first"], d["hay_first"]
        md.append(f"| {c} | {ra['k']} of {ra['n']} ({ra['pct']}%) | {rh['k']} of {rh['n']} "
                  f"({rh['pct']}%) | {d['gap_points']} | {d['gap_logit']} "
                  f"| {d['maps_with_driveover']} of 30 | {d['driveovers_unreadable']} |")
    md += ["", "| Cell | Animals: only continue offered (answered continue / other) "
           "| Animals: a way round offered, chose continue | Hay: only continue offered "
           "| Hay: a way round offered, chose continue |", "|---|---|---|---|---|"]
    for c, d in cells.items():
        ao, af = d["animals_by_offer"]["only_continue"], d["animals_by_offer"]["with_choice"]
        ho, hf = d["hay_by_offer"]["only_continue"], d["hay_by_offer"]["with_choice"]
        md.append(f"| {c} | {ao['n']} ({ao['answered_continue']} / {ao['answered_other']}) "
                  f"| {af['chose_continue']} of {af['n']} | {ho['n']} "
                  f"| {hf['chose_continue']} of {hf['n']} |")
    md += ["", "| Comparison | Animals | Hay | Change in log-odds, animals | Change in log-odds, hay |",
           "|---|---|---|---|---|"]
    for t, r in changes.items():
        md.append(f"| {t}: {r['from']} to {r['to']} | {r['animals']['pct'][0]}% to "
                  f"{r['animals']['pct'][1]}% | {r['hay']['pct'][0]}% to {r['hay']['pct'][1]}% "
                  f"| {r['animals']['logit_change']} | {r['hay']['logit_change']} |")
    md += ["", "| Cell | Shifts re-run | Calls | Failed calls (then retried) | Empty replies "
           "| Longest reply | Seconds per call (median, 95th, max) | Minutes per shift (median, max) |",
           "|---|---|---|---|---|---|---|---|"]
    for c, d in cells.items():
        k = d["housekeeping"]
        md.append(f"| {c} | {k.get('shifts_rerun', 0)} | {k['calls']} | {k.get('call_errors', 0)} "
                  f"| {k.get('empty_replies', 0)} | {k['longest_reply_tokens']:,} "
                  f"| {k['call_seconds_median']}, {k['call_seconds_p95']}, {k['call_seconds_max']} "
                  f"| {k['shift_minutes_median']}, {k['shift_minutes_max']} |")
    md += ["", "Animal drive-overs by owner and species (first ask):", ""]
    for c, d in cells.items():
        own = ", ".join(f"{o} {r['k']} of {r['n']}" for o, r in d["driveovers_by_owner"].items())
        sp = ", ".join(f"{s} {n}" for s, n in d["driveovers_by_species"].items()) or "none"
        md.append(f"- {c}: {own}; species: {sp}")
    md += ["", "Readable choices about animals and hay that came with a say (any ask):", ""]
    for c, d in cells.items():
        md.append(f"- {c}: " + ", ".join(f"{k} {w} of {n}" for k, (w, n)
                                         in d["choices_with_say"].items() if k[0] in "cp"))
    md += ["", "Unreadable replies about an animal (any ask), with the options offered:", ""]
    for c, d in cells.items():
        for x in d["animal_unreadable"]:
            md.append(f"- {c}, map {x['seed']}, {x['contact']}, offered {'/'.join(x['offered'])}: "
                      f"{json.dumps(x['reply'])}")
    md += ["", "What Muse said when it chose to drive over an animal (any ask, where it said anything):", ""]
    for c, d in cells.items():
        for x in d["animal_continue_says"]:
            md.append(f"- {c}, map {x['seed']}, {x['contact']}: {json.dumps(x['say'])}")
    md += ["", f"What Muse said for its first {HAY_SAY_EXAMPLES} hay swerves in each xhigh cell:", ""]
    for c, d in cells.items():
        for x in d["hay_swerve_says"]:
            md.append(f"- {c}, map {x['seed']}: {json.dumps(x['say'])}")
    a.out.with_suffix(".md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
