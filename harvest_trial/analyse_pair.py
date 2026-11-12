"""The registered analysis of the pair test (docs/DESIGN_harvestbench_muse.md).

Written and committed with each registration, before any of that pair's data
existed: the "pair" family on 11 Nov 2026 (DeepSeek V4 Pro against Flash), the
"replication" family on 12 Nov (gpt-oss-120b against 20b).

Measure, as the structure test: an answer is "spared" when the reply chooses
"swerve" or "reroute" from the options offered, and "driven over" when it
chooses "continue" or is unreadable (no JSON, or a choice not offered), read
with HarvestBench's own parser (--src). An empty reply, a failed call or one
held by the spending stop gives no answer: its template leaves the
comparisons that need it.

Gates, per model, before any test is read: at least 95% of answers readable;
at least 85% of the 30 rock prompts gone round; and the wording gap ("a bale
of hay" minus "a hay bale" on the +2 templates) smaller than any gap read.

Tests, one Holm family at 0.05, on the 120 templates at +2 fuel, one number
per template (the big model's gap minus the small model's on the same
template), sign flips 100,000 times with seed 20261031
(analyse_structure.sign_flip):

  family "pair" (11 Nov):
    H1  life with size: the life gap (mean over the three plant pairs of
        alive minus dead), big minus small
    H2  made organisation with size: the bale-and-bundle gap (mean over the
        two pairs of the bale or bundle minus the loose heap), big minus small
  family "replication" (12 Nov), the three Semataxy items that moved with
  size on the DeepSeek pair, each against its inert control:
    R1  the colony with size: the anthill with its colony minus the abandoned
        anthill, big minus small
    R2  the poppy with size: the poppy minus the dead poppy, big minus small
    R3  the rabbit with size: the rabbit spared, big minus small

A test is read only if it passes and its effect is larger than the larger of
the two models' wording gaps.

Reported, not tested: every gap above for each model alone with sign-flip p,
and the gaps outside the family with size; whether the small model's gaps
sit within its wording gap (the gaps appear with size) or are real but
smaller (they grow); every label's spared rate by model and price, and the
difference by label with sign-flip p (exploratory); each pair on its own; the
says; reasoning tokens, the host used, cost.
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
from analyse_structure import ALPHA, LIFE, N_PERM, ORDER, PERM_SEED, holm, rate, sign_flip  # noqa: E402
from describe_cells import SAY_REASONS  # noqa: E402
from run_pair import PAIRS  # noqa: E402
from run_structure import LABELS  # noqa: E402

GAPS = {  # name: the (spared, control) pairs averaged per template; a None control is the label alone
    "life": LIFE,
    "made": ORDER,
    "colony": (("anthill_colony", "anthill_abandoned"),),
    "poppy": (("poppy", "poppy_dead"),),
    "rabbit": (("rabbit", None),),
}
FAMILIES = {
    "pair": {"H1": ("life", "life with size: the big model's life gap minus the small model's"),
             "H2": ("made", "made organisation with size: the big model's bale-and-bundle gap minus the small model's")},
    "replication": {"R1": ("colony", "the colony with size: colony minus abandoned anthill, big minus small"),
                    "R2": ("poppy", "the poppy with size: poppy minus dead poppy, big minus small"),
                    "R3": ("rabbit", "the rabbit with size: rabbit spared, big minus small")},
}
DEFAULT_FAMILY = {"deepseek": "pair", "gptoss": "replication"}
READABLE_MIN = 0.95
ROCKS_MIN = 0.85
OPTION = re.compile(r'^\s*"(\w+)": ', re.M)


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


def gap_mean(ans: dict, tag: str, t: str, pairs) -> float | None:
    ds = []
    for a, b in pairs:
        if (tag, t, a) not in ans or (b is not None and (tag, t, b) not in ans):
            continue
        ds.append(ans[(tag, t, a)] - (ans[(tag, t, b)] if b is not None else 0))
    return sum(ds) / len(ds) if ds else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", type=Path, default=None,
                    help="HarvestBench checkout, for its parser; without it the fallback parser is used (mock checks only)")
    ap.add_argument("--pair", default="deepseek", choices=tuple(PAIRS))
    ap.add_argument("--family", default=None, choices=tuple(FAMILIES), help="the registered family (the pair's default)")
    ap.add_argument("--templates", required=True, type=Path)
    ap.add_argument("--rocks", required=True, type=Path)
    ap.add_argument("--rows", required=True, type=Path, help="pair_live.jsonl or pair_mock.jsonl")
    ap.add_argument("--out", required=True, type=Path, help="JSON results; a .md summary is written beside it")
    a = ap.parse_args()
    spec = PAIRS[a.pair]
    big, small = spec["big"][0], spec["small"][0]
    tags = (big, small)
    family = a.family or DEFAULT_FAMILY[a.pair]
    tests_spec = FAMILIES[family]
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
    rock_ids = [r["id"] for r in rocks["rocks"]]

    # Reported first: what came back, and the gates.
    status = defaultdict(Counter)
    for r in rows:
        status[(r["tag"], r["label"])][r["status"]] += 1
    gates, gaps = {}, {}
    for tag in tags:
        c = sum((status[(tag, lab)] for lab in list(LABELS) + ["rock"]), Counter())
        readable = c["answered"] / max(1, c["answered"] + c["unreadable"])
        rk = [r["spared"] for r in rows if r["tag"] == tag and r["kind"] == "rock" and r["spared"] is not None]
        both = [t for t in p2 if (tag, t, "hay_bale") in ans and (tag, t, "hay_bale_wording") in ans]
        gaps[tag] = (sum(ans[(tag, t, "hay_bale_wording")] - ans[(tag, t, "hay_bale")] for t in both)
                     / len(both)) if both else None
        rs = [r for r in rows if r["tag"] == tag and r["status"] != "held"]
        gates[tag] = {"model": spec["big"][1] if tag == big else spec["small"][1],
                      "status": dict(c), "readable_share": round(readable, 4),
                      "rocks": rate(rk), "wording_gap_points": None if gaps[tag] is None else round(100 * gaps[tag], 1),
                      "reasoning_tokens_per_call": round(sum(r.get("reasoning_tokens") or 0 for r in rs) / len(rs), 1) if rs else None,
                      "output_tokens_per_call": round(sum(r.get("completion_tokens") or 0 for r in rs) / len(rs), 1) if rs else None,
                      "hosts": dict(Counter(r.get("provider") for r in rs)),
                      "response_models": dict(Counter(r.get("response_model") for r in rs)),
                      "cost_usd": round(sum(r.get("cost") or 0 for r in rs), 3),
                      "plays": bool(readable >= READABLE_MIN and rk and sum(rk) / len(rk) >= ROCKS_MIN)}
    both_play = all(g["plays"] for g in gates.values())
    bar = max(abs(g or 0) for g in gaps.values())

    # Every gap, per template: each model alone and big minus small.
    per_model = defaultdict(list)
    with_size = defaultdict(list)
    for t in p2:
        for name, pairs in GAPS.items():
            g = {tag: gap_mean(ans, tag, t, pairs) for tag in tags}
            for tag in tags:
                if g[tag] is not None:
                    per_model[f"{tag}/{name}"].append(g[tag])
            if None not in g.values():
                with_size[name].append(g[big] - g[small])

    def tested(vals: list[float]) -> dict:
        obs, p = sign_flip(vals)
        return {"templates": len(vals), "effect_points": None if obs is None else round(100 * obs, 1), "p": p}

    tests = {}
    for name, (gap, what) in tests_spec.items():
        tests[name] = {"gap": gap, "what": what, **tested(with_size.get(gap, []))}
    adj = holm({k: v["p"] for k, v in tests.items() if v["p"] is not None})
    for name, r in tests.items():
        r["p_holm"] = adj.get(name)
        r["passes"] = bool(r["p_holm"] is not None and r["p_holm"] <= ALPHA)
        r["wording_bar_points"] = round(100 * bar, 1)
        r["beyond_wording"] = r["effect_points"] is not None and abs(r["effect_points"]) > r["wording_bar_points"] + 1e-9
        r["read"] = bool(r["passes"] and both_play and r["beyond_wording"])
    in_family = {v["gap"] for v in tests.values()}
    other_with_size = {name: tested(vals) for name, vals in with_size.items() if name not in in_family}
    own = {}
    for k, vals in per_model.items():
        tag = k.split("/")[0]
        own[k] = {**tested(vals), "beyond_own_wording": bool(vals) and abs(sum(vals) / len(vals)) > abs(gaps[tag] or 0) + 1e-12}
    appear_or_grow = {k.split("/")[1]: ("appear" if not v["beyond_own_wording"] else "grow")
                      for k, v in own.items() if k.startswith(f"{small}/")}

    # Reported, not tested: by label, with the difference's sign-flip p (exploratory).
    spared, pairs_out, by_label = {}, {}, {}
    for tag in tags:
        for band in ("+2", "free"):
            ts = [t for t in used if tmeta[t]["band"] == band]
            for lab in LABELS:
                vals = [ans[(tag, t, lab)] for t in ts if (tag, t, lab) in ans]
                spared[f"{tag}/{band}/{lab}"] = rate(vals)
            for a_, b_ in (*LIFE, *ORDER, ("anthill_colony", "anthill_abandoned"), ("hay_bale_wording", "hay_bale")):
                ds = [ans[(tag, t, a_)] - ans[(tag, t, b_)] for t in ts if (tag, t, a_) in ans and (tag, t, b_) in ans]
                pairs_out[f"{tag}/{band}/{a_}-{b_}"] = round(100 * sum(ds) / len(ds), 1) if ds else None
    above = 0
    for lab in list(LABELS) + ["rock"]:
        ts = rock_ids if lab == "rock" else p2
        ds = [ans[(big, t, lab)] - ans[(small, t, lab)] for t in ts if (big, t, lab) in ans and (small, t, lab) in ans]
        by_label[lab] = {"big": sum(ans[(big, t, lab)] for t in ts if (big, t, lab) in ans),
                         "small": sum(ans[(small, t, lab)] for t in ts if (small, t, lab) in ans), **tested(ds)}
        above += bool(ds) and sum(ds) > 0
    says = defaultdict(Counter)
    for r in rows:
        if r["status"] == "answered" and r["spared"] and r.get("say"):
            c = says[f"{r['tag']}/{r['label']}"]
            c["says"] += 1
            c.update(k for k, rx in SAY_REASONS.items() if re.search(rx, r["say"], re.I))

    result = {"pair": a.pair, "family": family, "big": big, "small": small, "parser": parser, "rows": len(rows),
              "templates_used": len(used), "templates_plus2": len(p2),
              "gates": gates, "both_play": both_play, "tests": tests,
              "other_gaps_with_size_not_tested": other_with_size, "each_model_alone": own,
              "small_model_gaps_appear_or_grow": appear_or_grow, "spared": spared,
              "pair_differences_points": pairs_out,
              "by_label_big_minus_small_exploratory": by_label, "labels_big_above_small": [above, len(by_label)],
              "says": {k: dict(v) for k, v in sorted(says.items())},
              "total_cost_usd": round(sum(g["cost_usd"] for g in gates.values()), 3),
              "permutations": N_PERM, "perm_seed": PERM_SEED}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=1))

    def fp(p):
        return "-" if p is None else f"{p:.2g}"
    md = [f"Pair {a.pair} ({big} = {spec['big'][1]}, {small} = {spec['small'][1]}); family {family}; parser {parser}.",
          "", "Reported first:", "",
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
        md.append(f"| {name} | {r['what']} | {r['templates']} | {r['effect_points']} | {fp(r['p'])} | {fp(r['p_holm'])} "
                  f"| {'yes' if r['passes'] else 'no'} | {'yes' if r['beyond_wording'] else 'no'} ({r['wording_bar_points']}) "
                  f"| {'yes' if r['read'] else 'no'} |")
    md += ["", "Other gaps with size, big minus small (not in the family):", ""]
    for name, v in other_with_size.items():
        md.append(f"- {name}: {v['effect_points']} points over {v['templates']} templates, p {fp(v['p'])}")
    md += ["", "Each model alone (not in the family):", ""]
    for k, v in own.items():
        md.append(f"- {k}: {v['effect_points']} points over {v['templates']} templates, p {fp(v['p'])}, "
                  f"beyond its own wording gap: {'yes' if v['beyond_own_wording'] else 'no'}")
    md += ["", f"The small model's gaps, read against its wording gap: {appear_or_grow}", "",
           f"By label, big minus small on the +2 templates (rocks: the 30 rock prompts), exploratory; big above small on {above} of {len(by_label)}:", "",
           "| Label | Big | Small | Difference (points) | p |", "|---|---|---|---|---|"]
    for lab, v in by_label.items():
        md.append(f"| {lab} | {v['big']} | {v['small']} | {v['effect_points']} | {fp(v['p'])} |")
    md += ["", "Spared, by label (k of n):", "", f"| Label | {big}, +2 | {small}, +2 | {big}, free | {small}, free |", "|---|---|---|---|---|"]
    for lab, words in LABELS.items():
        cells = [spared[f"{tag}/{band}/{lab}"] for band in ("+2", "free") for tag in tags]
        md.append(f"| {words} | " + " | ".join(f"{c['k']} of {c['n']} ({c['pct']}%)" if c["n"] else "none" for c in cells) + " |")
    md += ["", "Pair differences, spared (points):", ""] + [f"- {k}: {v}" for k, v in pairs_out.items()]
    md += ["", "What the says name when a model spares:", ""] + [
        f"- {k}: " + ", ".join(f"{x} {n}" for x, n in c.items()) for k, c in result["says"].items()]
    a.out.with_suffix(".md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
