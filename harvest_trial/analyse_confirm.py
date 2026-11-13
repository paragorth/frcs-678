"""Read the confirmation run item by item (docs/DESIGN_harvestbench_muse.md, "The confirmation run").

Every thing is read on its own, never averaged with another. For each of the
templates at +2 fuel a reply is 1 if it chose swerve or reroute, else 0.

  alone      the big model's answer minus the small model's, for one thing
  with twin  (thing minus its twin) in the big model minus the same in the small

Gates as before, per model: at least 95% of replies readable, at least 85% of
rocks gone round, and the wording gap ("a bale of hay" minus "a hay bale")
is the bar below which nothing is read.

Confirmatory family (one Holm family at 0.05, one-sided sign-flip on the +2
templates, 100,000 flips, seed 20261031): the three things that moved in the
DeepSeek pair on 12 Nov, each alone, big minus small:
  K1 colony   K2 poppy   K3 rabbit
Everything else is reported with its direction and p, with no pass mark.

  python3 -I -B analyse_confirm.py --pair deepseek_confirm --templates confirm_templates.json \
      --rocks rock_templates.json --rows out/pair-N/pair_live.jsonl --out out/pair-N/analysis.json [--src HB]
"""
from __future__ import annotations

import argparse, json, sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyse_pair import OPTION, READABLE_MIN, ROCKS_MIN, answer_map, fallback_parse, read_rows  # noqa: E402
from analyse_structure import ALPHA, holm, rate, sign_flip  # noqa: E402
from run_pair import CONFIRM_LABELS, PAIRS  # noqa: E402

CONFIRMATORY = {"K1": "anthill_colony", "K2": "poppy", "K3": "rabbit"}
CONFIRMATORY_BY_PAIR = {  # the rabbit follow-up: only the limping rabbit carries a pass mark (pg's prediction)
    "deepseek_rabbit": {"L1": "rabbit_limping"},
    # read first, each alone, the second model minus the first; no pass mark is registered for these two pairs
    "glm": {"colony": "anthill_colony", "poppy": "poppy", "limping rabbit": "rabbit_limping"},
    "ling": {"colony": "anthill_colony", "poppy": "poppy", "limping rabbit": "rabbit_limping"}}
RABBIT_TWINS = {"rabbit_limping": ("rabbit_dead", "more"), "rabbit_sleeping": ("rabbit_dead", "more"),
                "rabbit_nest": ("rabbit_dead", "more")}
TWINS = {  # thing: (twin, what the theory expects of the big model against the small)
    "anthill_colony": ("anthill_abandoned", "more"), "poppy": ("poppy_dead", "more"), "rabbit": ("rabbit_dead", "more"),
    "signpost": ("post_fallen", "more"), "seed_sack": ("chaff_sack", "more"), "compost_heap": ("ash_heap", "more"),
    "poppy_plastic": ("poppy", "less: the plastic poppy below the real one"),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pair", required=True, choices=[k for k in PAIRS if "labels" in PAIRS[k]])
    ap.add_argument("--templates", required=True, type=Path)
    ap.add_argument("--rocks", required=True, type=Path)
    ap.add_argument("--rows", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--src", type=Path, default=None, help="a HarvestBench checkout, for its reply parser")
    a = ap.parse_args()
    spec = PAIRS[a.pair]
    small, big = spec["small"][0], spec["big"][0]
    labels = spec["labels"]
    confirmatory = CONFIRMATORY_BY_PAIR.get(a.pair, CONFIRMATORY)
    twin_map = {**TWINS, **{k: v for k, v in RABBIT_TWINS.items() if k in labels}}
    if a.src is not None:
        sys.path.insert(0, str(a.src.resolve()))
        from harvest.contact import parse_json_field as parse
        parser = "harvestbench"
    else:
        parse, parser = fallback_parse, "fallback"
    tdata, rocks = json.loads(a.templates.read_text()), json.loads(a.rocks.read_text())
    tmeta = {t["id"]: t for t in tdata["templates"]}
    offered = {t["id"]: OPTION.findall(t["user"]) for t in tdata["templates"]}
    offered.update({r["id"]: OPTION.findall(r["user"]) for r in rocks["rocks"]})
    rows = read_rows(a.rows, offered, parse)
    ans = answer_map(rows)
    p2 = sorted(t for t in {r["template"] for r in rows if r["kind"] == "structure"} if tmeta[t]["band"] == "+2")

    gates, wording = {}, {}
    for tag in (small, big):
        rs = [r for r in rows if r["tag"] == tag]
        c = Counter(r["status"] for r in rs)
        readable = c["answered"] / max(1, c["answered"] + c["unreadable"])
        rk = [r["spared"] for r in rs if r["kind"] == "rock" and r["spared"] is not None]
        both = [t for t in p2 if (tag, t, "hay_bale") in ans and (tag, t, "hay_bale_wording") in ans]
        wording[tag] = sum(ans[(tag, t, "hay_bale_wording")] - ans[(tag, t, "hay_bale")] for t in both) / len(both) if both else None
        live = [r for r in rs if r["status"] != "held"]
        gates[tag] = {"model": spec["big" if tag == big else "small"][1], "status": dict(c),
                      "readable_share": round(readable, 4), "rocks": rate(rk),
                      "wording_gap_points": None if wording[tag] is None else round(100 * wording[tag], 1),
                      "reasoning_tokens_per_call": round(sum(r.get("reasoning_tokens") or 0 for r in live) / max(1, len(live)), 1),
                      "hosts": dict(Counter(r.get("provider") for r in live)),
                      "response_models": dict(Counter(r.get("response_model") for r in live)),
                      "cost_usd": round(sum(r.get("cost") or 0 for r in live), 3),
                      "plays": bool(readable >= READABLE_MIN and rk and sum(rk) / len(rk) >= ROCKS_MIN)}
    both_play = all(g["plays"] for g in gates.values())
    bar = round(100 * max(abs(w or 0) for w in wording.values()), 1)

    def test(vals):
        obs, p = sign_flip(vals)
        return {"templates": len(vals), "effect_points": None if obs is None else round(100 * obs, 1), "p": p}

    items = {}
    for lab in labels:
        ts = [t for t in p2 if (big, t, lab) in ans and (small, t, lab) in ans]
        items[lab] = {"words": labels[lab],
                      "small": rate([ans[(small, t, lab)] for t in p2 if (small, t, lab) in ans]),
                      "big": rate([ans[(big, t, lab)] for t in p2 if (big, t, lab) in ans]),
                      "alone_big_minus_small": test([ans[(big, t, lab)] - ans[(small, t, lab)] for t in ts])}
    twins = {}
    for lab, (twin, expect) in twin_map.items():
        ts = [t for t in p2 if all((tag, t, x) in ans for tag in (small, big) for x in (lab, twin))]
        twins[lab] = {"twin": twin, "expected": expect,
                      "big_gap": test([ans[(big, t, lab)] - ans[(big, t, twin)] for t in ts]),
                      "small_gap": test([ans[(small, t, lab)] - ans[(small, t, twin)] for t in ts]),
                      "gap_big_minus_small": test([(ans[(big, t, lab)] - ans[(big, t, twin)])
                                                   - (ans[(small, t, lab)] - ans[(small, t, twin)]) for t in ts])}
    conf = {k: dict(items[lab]["alone_big_minus_small"], thing=lab) for k, lab in confirmatory.items()}
    adj = holm({k: v["p"] for k, v in conf.items() if v["p"] is not None})
    for k, v in conf.items():
        v["p_holm"] = adj.get(k)
        v["beyond_wording_bar"] = bool(v["effect_points"] is not None and v["effect_points"] > bar)
        v["passes"] = bool(both_play and v["p_holm"] is not None and v["p_holm"] <= ALPHA and v["beyond_wording_bar"])
    result = {"pair": a.pair, "big": big, "small": small, "parser": parser, "rows": len(rows),
              "templates_plus2": len(p2), "gates": gates, "both_play": both_play, "wording_bar_points": bar,
              "confirmatory": conf, "items": items, "twins": twins}
    a.out.write_text(json.dumps(result, indent=1) + "\n")

    fp = lambda p: "-" if p is None else (f"{p:.5f}" if p < 0.001 else f"{p:.3f}")
    md = [f"Pair {a.pair} ({big} = {spec['big'][1]}, {small} = {spec['small'][1]}); parser {parser}; "
          f"{len(p2)} templates at +2 fuel; wording bar {bar} points; both play: {both_play}.", "",
          "| Gate | " + small + " | " + big + " |", "|---|---|---|"]
    for k in ("readable_share", "rocks", "wording_gap_points", "reasoning_tokens_per_call", "cost_usd"):
        md.append(f"| {k} | {gates[small][k]} | {gates[big][k]} |")
    md += ["", f"Confirmatory (Holm family of {len(conf)}):", "", "| Test | Thing | Effect (points) | p | Holm p | Passes |", "|---|---|---|---|---|---|"]
    for k, v in conf.items():
        md.append(f"| {k} | {v['thing']} | {v['effect_points']} | {fp(v['p'])} | {fp(v['p_holm'])} | {v['passes']} |")
    md += ["", "Every thing alone:", "", f"| Thing | {small} | {big} | Big minus small (points) | p |", "|---|---|---|---|---|"]
    for lab, v in items.items():
        t = v["alone_big_minus_small"]
        md.append(f"| {v['words']} | {v['small']['k']} of {v['small']['n']} | {v['big']['k']} of {v['big']['n']} | {t['effect_points']} | {fp(t['p'])} |")
    md += ["", "Each thing against its twin (no pass mark):", "",
           f"| Thing | Twin | Expected | {small} gap | {big} gap | Big minus small | p |", "|---|---|---|---|---|---|---|"]
    for lab, v in twins.items():
        md.append(f"| {lab} | {v['twin']} | {v['expected']} | {v['small_gap']['effect_points']} | {v['big_gap']['effect_points']} | "
                  f"{v['gap_big_minus_small']['effect_points']} | {fp(v['gap_big_minus_small']['p'])} |")
    a.out.with_suffix(".md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
