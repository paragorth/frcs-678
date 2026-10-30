"""Draw the structure test's 150 templates from the four cells' neutral logs.

The rule is the registration's (docs/DESIGN_harvestbench_muse.md, "The
structure test"): from the two neutral cells of job 39672076, the hay-bale
contact asks that offered "continue" and "swerve", with going round at +2 fuel
or free. Identical prompts count once. With seed 20261031: 60 from each cell
at +2 fuel, 2 per map, the maps with fewer topped up from that cell's other
maps; and 30 at +0, 1 per map that has one, the rest from other maps.

Each template keeps the logged user message byte for byte, the cell and map it
came from, and Muse's answer in the game (for the replay check). The neutral
briefing, the system message of every one of these asks, is kept once.

Run on the cluster, where Inspect and HarvestBench are installed:
  python -I -B draw_templates.py --src $SRC --log-dir out/cells-39672076 \
      --out structure_templates.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

SEED = 20261031
HAY = "on your route: a hay bale."
OPTION = re.compile(r'^\s*"(\w+)": ', re.M)
SWERVE_PRICE = re.compile(r'"swerve": go around it \(\+(-?\d+) fuel\)')
EFFORTS = ("minimal", "xhigh")
PER_CELL_PLUS2 = 60
PER_MAP_PLUS2 = 2
FREE = 30
MAPS = range(30)


def hay_asks(log, parse_json_field) -> tuple[list[dict], set[str]]:
    """Every answered hay-bale contact ask in one cell's log, in the order asked."""
    rows, systems = [], set()
    for s in sorted(log.samples or [], key=lambda s: s.metadata["seed"]):
        for e in s.events:
            if getattr(e, "event", None) != "model" or e.error or not e.input:
                continue
            if (e.output.metadata or {}).get("guard_stop"):
                continue
            user = e.input[-1].text
            if HAY not in user:
                continue
            if user.count("a hay bale") != 1:
                raise SystemExit(f"a hay prompt names the bale more than once: {user[:200]!r}")
            offered = OPTION.findall(user)
            price = SWERVE_PRICE.search(user)
            if not ({"continue", "swerve"} <= set(offered)) or not price:
                continue
            band = {2: "+2", 0: "free"}.get(int(price.group(1)))
            if band is None:
                continue
            systems.add(e.input[0].text)
            choice, say = parse_json_field(e.output.completion or "", "choice")
            readable = choice in offered
            rows.append({"band": band, "map": s.metadata["seed"], "user": user,
                         "in_game_choice": choice, "in_game_readable": readable,
                         "in_game_spared": readable and choice in ("swerve", "reroute"),
                         "in_game_say": say})
    return rows, systems


def draw(asks: dict[str, list[dict]]) -> list[dict]:
    rng = random.Random(SEED)
    seen, pools = set(), {}
    for band in ("+2", "free"):
        for cell in EFFORTS:
            by_map = defaultdict(list)
            for r in asks[cell]:
                if r["band"] == band and r["user"] not in seen:
                    seen.add(r["user"])
                    by_map[r["map"]].append(r | {"cell": cell})
            pools[(band, cell)] = by_map
    picked = []
    for cell in EFFORTS:  # +2: 2 per map from each cell, topped up from its other maps
        chosen, rest = [], []
        for m in MAPS:
            pool = pools[("+2", cell)].get(m, [])
            take = rng.sample(pool, min(PER_MAP_PLUS2, len(pool)))
            chosen += take
            rest += [r for r in pool if r not in take]
        chosen += rng.sample(rest, PER_CELL_PLUS2 - len(chosen))
        picked += chosen
    chosen, rest = [], []  # +0: 1 per map from either cell, the rest from other maps
    for m in MAPS:
        pool = pools[("free", "minimal")].get(m, []) + pools[("free", "xhigh")].get(m, [])
        if pool:
            take = rng.choice(pool)
            chosen.append(take)
            rest += [r for r in pool if r is not take]
    chosen += rng.sample(rest, FREE - len(chosen))
    picked += chosen
    count = defaultdict(int)
    for r in picked:
        key = (r["band"], r["cell"], r["map"])
        count[key] += 1
        r["id"] = (f"{'p2' if r['band'] == '+2' else 'p0'}-{r['cell'][:3]}"
                   f"-m{r['map']:02d}-{count[key]}")
    return picked


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--log-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    sys.path.insert(0, str(a.src.resolve()))
    from harvest.contact import parse_json_field
    from inspect_ai.log import read_eval_log

    asks, systems, files = {}, set(), {}
    for p in sorted(a.log_dir.glob("*.eval")):
        log = read_eval_log(str(p), resolve_attachments=True)
        ta = log.eval.task_args
        if ta.get("arm") != "neutral":
            continue
        cell = ta.get("reasoning_effort")
        if cell in asks:
            raise SystemExit(f"two neutral logs for {cell}")
        asks[cell], sy = hay_asks(log, parse_json_field)
        systems |= sy
        files[cell] = p.name
    if set(asks) != set(EFFORTS):
        raise SystemExit(f"need the neutral minimal and xhigh logs, found {sorted(asks)}")
    if len(systems) != 1:
        raise SystemExit(f"{len(systems)} different system messages among the hay asks")
    briefing = systems.pop()
    templates = draw(asks)
    available = {f"{band} {cell}": len({r["user"] for r in asks[cell] if r["band"] == band})
                 for band in ("+2", "free") for cell in EFFORTS}
    out = {"source_job": 39672076, "source_logs": files, "seed": SEED,
           "rule": __doc__.split("\n\n")[1].replace("\n", " "),
           "available_distinct": available,
           "briefing": briefing,
           "briefing_sha256": hashlib.sha256(briefing.encode()).hexdigest(),
           "templates": templates}
    a.out.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    by = defaultdict(int)
    for r in templates:
        by[(r["band"], r["cell"])] += 1
    print(json.dumps({"templates": len(templates), "by_band_and_cell": {f"{b} {c}": n for (b, c), n in by.items()},
                      "maps_+2": len({r["map"] for r in templates if r["band"] == "+2"}),
                      "maps_free": len({r["map"] for r in templates if r["band"] == "free"}),
                      "available_distinct": available,
                      "in_game_spared_+2": {c: sum(r["in_game_spared"] for r in templates
                                                   if r["band"] == "+2" and r["cell"] == c)
                                            for c in EFFORTS},
                      "sha256": hashlib.sha256(a.out.read_bytes()).hexdigest()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
