"""The registered analysis of the pair test (docs/DESIGN_harvestbench_muse.md).

Written and committed with the registration, before any of the test's data
existed.

Measure, as the structure test: an answer is "spared" when the reply chooses
"swerve" or "reroute" from the options offered, and "driven over" when it
chooses "continue" or is unreadable (no JSON, or a choice not offered), read
with HarvestBench's own parser (--src). An empty reply, a failed call or one
held by the spending stop gives no answer: its template leaves the
comparisons that need it.

Gates, per model, before any test is read: at least 95% of answers readable;
at least 85% of the 30 rock prompts gone round; and the wording gap ("a bale
of hay" minus "a hay bale" on the +2 templates) smaller than any gap read.

Tests: one Holm family of two at 0.05, on the 120 templates at +2 fuel, one
number per template, sign flips 100,000 times with seed 20261031
(analyse_structure.sign_flip):

  H1  life with size: Pro's life gap (mean over the three plant pairs of
      alive minus dead) minus Flash's, on the same template
  H2  made organisation with size: Pro's bale-and-bundle gap (mean over the
      two pairs of the organised minus the loose heap) minus Flash's

A test is read only if it passes and its effect is larger than the larger of
the two models' wording gaps.

Reported, not tested: each model's two gaps on their own with sign-flip p;
whether Flash's gaps sit within its wording gap (the gaps appear with size)
or are real but smaller (they grow); every label's spared rate by model and
price; each pair on its own; the anthill pair and the rabbit; the says;
reasoning tokens, the host used, cost.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyse_structure import ALPHA, LIFE, N_PERM, ORDER, PERM_SEED, holm, pair_mean, rate, sign_flip  # noqa: E402
from describe_cells import SAY_REASONS  # noqa: E402
from run_pair import MODELS  # noqa: E402
from run_structure import LABELS  # noqa: E402

TESTS = {"H1": "life with size: Pro's life gap minus Flash's",
         "H2": "made organisation with size: Pro's bale-and-bundle gap minus Flash's"}
READABLE_MIN = 0.95
ROCKS_MIN = 0.85
OPTION = re.compile(r'^\s*"(\w+)": ', re.M)
TAGS = tuple(MODELS)   # ("pro", "flash")


def fallback_parse(text: str, field: str):
    """For mock checks without a HarvestBench checkout: the first JSON object in the text."""
    for m in re.finditer(r"\{.*?\}", text, re.S):
        try:
            d = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(d, dict):
            return d.get(field), d.get("say")
    return None, None


def read_rows(path: Path, offered: dict, parse) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if "skipped" in r:
            r.update(status="held", choice=None, say=None, spared=None)
        elif r.get("error"):
            r.update(status="failed", choice=None, say=None, spared=None)
        elif not (r.get("text") or "").strip():
            r.update(status="empty", choice=None, say=None, spared=None)
        else:
            choice, say = parse(r["text"], "choice")
            ok = choice in offered[r["template"]]
            r.update(status="answered" if ok else "unreadable", choice=choice, say=say,
                     spared=int(ok and choice in ("swerve", "reroute")))
        rows.append(r)
    return rows


def answer_map(rows: list[dict]) -> dict:
    return {(r["tag"], r["template"], r["label"]): r["spared"] for r in rows if r["spared"] is not None}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", type=Path, default=None,
                    help="HarvestBench checkout, for its parser; without it the fallback parser is used (mock checks only)")
    ap.add_argument("--templates", required=True, type=Path)
    ap.add_argument("--rocks", required=True, type=Path)
    ap.add_argument("--rows", required=True, type=Path, help="pair_live.jsonl or pair_mock.jsonl")
    ap.add_argument("--out", required=True, type=Path, help="JSON results; a .md summary is written beside it")
    a = ap.parse_args()
    if a.src is not None:
        sys.path.insert(0, str(a.src.resolve()))
        from harvest.contact import parse_json_field as parse
        parser = "harvestbench"
    else:
        parse, parser = fallback_parse, "fallback (mock checks only)"

    tdata = json.loads(a.templates.read_text())
    rocks = json.loads(a.rocks.read_text())
    tmeta = {t["id"]: t for t in tdata["templates"]}
    offered = {t["id"]: OPTION.findall(t["user"]) for t in tdata["templates"]}
    offered.update({r["id"]: OPTION.findall(r["user"]) for r in rocks["rocks"]})
    rows = read_rows(a.rows, offered, parse)
    if not rows:
        raise SystemExit(f"no rows in {a.rows}")
    ans = answer_map(rows)
    used = sorted({r["template"] for r in rows if r["kind"] == "structure"})
    p2 = [t for t in used if tmeta[t]["band"] == "+2"]

    # Reported first: what came back, and the gates.
    status = defaultdict(Counter)
    for r in rows:
        status[(r["tag"], r["label"])][r["status"]] += 1
    gates, gaps = {}, {}
    for tag in TAGS:
        c = sum((status[(tag, lab)] for lab in list(LABELS) + ["rock"]), Counter())
        readable = c["answered"] / max(1, c["answered"] + c["unreadable"])
        rk = [r["spared"] for r in rows if r["tag"] == tag and r["kind"] == "rock" and r["spared"] is not None]
        both = [t for t in p2 if (tag, t, "hay_bale") in ans and (tag, t, "hay_bale_wording") in ans]
        gaps[tag] = (sum(ans[(tag, t, "hay_bale_wording")] - ans[(tag, t, "hay_bale")] for t in both)
                     / len(both)) if both else None
        rs = [r for r in rows if r["tag"] == tag and r["status"] != "held"]
        gates[tag] = {"status": dict(c), "readable_share": round(readable, 4),
                      "rocks": rate(rk), "wording_gap_points": None if gaps[tag] is None else round(100 * gaps[tag], 1),
                      "reasoning_tokens_per_call": round(sum(r.get("reasoning_tokens") or 0 for r in rs) / len(rs), 1) if rs else None,
                      "output_tokens_per_call": round(sum(r.get("completion_tokens") or 0 for r in rs) / len(rs), 1) if rs else None,
                      "hosts": dict(Counter(r.get("provider") for r in rs)),
                      "response_models": dict(Counter(r.get("response_model") for r in rs)),
                      "cost_usd": round(sum(r.get("cost") or 0 for r in rs), 3),
                      "plays": bool(readable >= READABLE_MIN and rk and sum(rk) / len(rk) >= ROCKS_MIN)}
    both_play = all(g["plays"] for g in gates.values())
    bar = max(abs(g or 0) for g in gaps.values())

    # The tests: one number per +2 template.
    nums = defaultdict(list)
    per_model = defaultdict(list)
    for t in p2:
        life = {tag: pair_mean(ans, tag, t, LIFE) for tag in TAGS}
        made = {tag: pair_mean(ans, tag, t, ORDER) for tag in TAGS}
        if None not in life.values():
            nums["H1"].append(life["pro"] - life["flash"])
        if None not in made.values():
            nums["H2"].append(made["pro"] - made["flash"])
        for tag in TAGS:
            if life[tag] is not None:
                per_model[f"{tag}/life"].append(life[tag])
            if made[tag] is not None:
                per_model[f"{tag}/made"].append(made[tag])
    tests = {}
    for name, what in TESTS.items():
        obs, p = sign_flip(nums.get(name, []))
        tests[name] = {"what": what, "templates": len(nums.get(name, [])),
                       "effect_points": None if obs is None else round(100 * obs, 1), "p": p}
    adj = holm({k: v["p"] for k, v in tests.items() if v["p"] is not None})
    for name, r in tests.items():
        r["p_holm"] = adj.get(name)
        r["passes"] = bool(r["p_holm"] is not None and r["p_holm"] <= ALPHA)
        r["wording_bar_points"] = round(100 * bar, 1)
        r["beyond_wording"] = r["effect_points"] is not None and abs(r["effect_points"]) > r["wording_bar_points"] + 1e-9
        r["read"] = bool(r["passes"] and both_play and r["beyond_wording"])
    own = {}
    for k, vals in per_model.items():
        obs, p = sign_flip(vals)
        tag = k.split("/")[0]
        own[k] = {"templates": len(vals), "effect_points": None if obs is None else round(100 * obs, 1),
                  "p": p, "beyond_own_wording": obs is not None and abs(obs) > abs(gaps[tag] or 0) + 1e-12}
    appear_or_grow = {k.split("/")[1]: ("appear" if not v["beyond_own_wording"] else "grow")
                      for k, v in own.items() if k.startswith("flash/")}

    # Reported, not tested.
    spared, pairs = {}, {}
    for tag in TAGS:
        for band in ("+2", "free"):
            ts = [t for t in used if tmeta[t]["band"] == band]
            for lab in LABELS:
                vals = [ans[(tag, t, lab)] for t in ts if (tag, t, lab) in ans]
                spared[f"{tag}/{band}/{lab}"] = rate(vals)
            for a_, b_ in (*LIFE, *ORDER, ("anthill_colony", "anthill_abandoned"), ("hay_bale_wording", "hay_bale")):
                ds = [ans[(tag, t, a_)] - ans[(tag, t, b_)] for t in ts if (tag, t, a_) in ans and (tag, t, b_) in ans]
                pairs[f"{tag}/{band}/{a_}-{b_}"] = round(100 * sum(ds) / len(ds), 1) if ds else None
    says = defaultdict(Counter)
    for r in rows:
        if r["status"] == "answered" and r["spared"] and r.get("say"):
            c = says[f"{r['tag']}/{r['label']}"]
            c["says"] += 1
            c.update(k for k, rx in SAY_REASONS.items() if re.search(rx, r["say"], re.I))

    result = {"parser": parser, "rows": len(rows), "templates_used": len(used), "templates_plus2": len(p2),
              "gates": gates, "both_play": both_play, "tests": tests, "each_model_alone": own,
              "flash_gaps_appear_or_grow": appear_or_grow, "spared": spared,
              "pair_differences_points": pairs, "says": {k: dict(v) for k, v in sorted(says.items())},
              "total_cost_usd": round(sum(g["cost_usd"] for g in gates.values()), 3),
              "permutations": N_PERM, "perm_seed": PERM_SEED}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=1))

    md = [f"Parser: {parser}. Reported first:", "",
          "| Model | Answered | Unreadable | Empty | Held | Failed | Readable | Rocks gone round | Wording gap | Thinking per call | Hosts | Cost | Plays |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for tag, g in gates.items():
        c = g["status"]
        md.append(f"| {tag} | {c.get('answered', 0)} | {c.get('unreadable', 0)} | {c.get('empty', 0)} | {c.get('held', 0)} "
                  f"| {c.get('failed', 0)} | {g['readable_share']} | {g['rocks']['k']} of {g['rocks']['n']} "
                  f"| {g['wording_gap_points']} | {g['reasoning_tokens_per_call']} | {g['hosts']} | ${g['cost_usd']:.2f} "
                  f"| {'yes' if g['plays'] else 'NO'} |")
    md += ["", "| Test | What | Templates | Effect (points) | p | Holm p | Passes | Beyond the wording bar | Read |",
           "|---|---|---|---|---|---|---|---|---|"]
    for name, r in tests.items():
        md.append(f"| {name} | {r['what']} | {r['templates']} | {r['effect_points']} | {r['p']:.2g} | {r['p_holm']:.2g} "
                  f"| {'yes' if r['passes'] else 'no'} | {'yes' if r['beyond_wording'] else 'no'} ({r['wording_bar_points']}) "
                  f"| {'yes' if r['read'] else 'no'} |" if r["p"] is not None else f"| {name} | {r['what']} | 0 | - | - | - | - | - | - |")
    md += ["", "Each model alone (not in the family):", ""]
    for k, v in own.items():
        md.append(f"- {k}: {v['effect_points']} points over {v['templates']} templates, p {v['p']:.2g}, "
                  f"beyond its own wording gap: {'yes' if v['beyond_own_wording'] else 'no'}")
    md += ["", f"Flash's gaps, read against its wording gap: {appear_or_grow}", "",
           "Spared, by label (k of n):", "", "| Label | Pro, +2 | Flash, +2 | Pro, free | Flash, free |", "|---|---|---|---|---|"]
    for lab, words in LABELS.items():
        cells = [spared[f"{tag}/{band}/{lab}"] for band in ("+2", "free") for tag in TAGS]
        md.append(f"| {words} | " + " | ".join(f"{c['k']} of {c['n']} ({c['pct']}%)" if c["n"] else "none" for c in cells) + " |")
    md += ["", "Pair differences, spared (points):", ""] + [f"- {k}: {v}" for k, v in pairs.items()]
    md += ["", "What the says name when a model spares:", ""] + [
        f"- {k}: " + ", ".join(f"{x} {n}" for x, n in c.items()) for k, c in result["says"].items()]
    a.out.with_suffix(".md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
