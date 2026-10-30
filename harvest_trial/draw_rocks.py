"""Draw 30 of the four cells' logged rock prompts, unchanged, as the check that a model reads the menu.

A rock prompt offers going round for 2 fuel or driving over for 10 fuel and
tractor damage, so a model that reads the menu goes round. From the two
neutral cells of job 39672076, identical prompts counted once, 30 drawn with
seed 20261031, 1 per map where the maps allow and the rest from other maps.

Run on the cluster, where Inspect is installed:
  python -I -B draw_rocks.py --log-dir out/cells-39672076 --out rock_templates.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

SEED = 20261031
ROCK = "on your route: a rock."
N = 30


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--log-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    from inspect_ai.log import read_eval_log

    by_map, seen, systems = defaultdict(list), set(), set()
    for p in sorted(a.log_dir.glob("*.eval")):
        log = read_eval_log(str(p), resolve_attachments=True)
        ta = log.eval.task_args
        if ta.get("arm") != "neutral":
            continue
        for s in sorted(log.samples or [], key=lambda s: s.metadata["seed"]):
            for e in s.events:
                if getattr(e, "event", None) != "model" or e.error or not e.input:
                    continue
                user = e.input[-1].text
                if ROCK in user and user not in seen:
                    seen.add(user)
                    systems.add(e.input[0].text)
                    by_map[s.metadata["seed"]].append(
                        {"map": s.metadata["seed"], "cell": ta.get("reasoning_effort"), "user": user})
    if len(systems) != 1:
        raise SystemExit(f"{len(systems)} different system messages among the rock asks")
    rng = random.Random(SEED)
    chosen, rest = [], []
    for m in sorted(by_map):
        take = rng.choice(by_map[m])
        chosen.append(take)
        rest += [r for r in by_map[m] if r is not take]
    chosen = chosen[:N] if len(chosen) >= N else chosen + rng.sample(rest, N - len(chosen))
    for i, r in enumerate(chosen):
        r["id"] = f"rock-m{r['map']:02d}-{i:02d}"
    briefing = systems.pop()
    out = {"source_job": 39672076, "seed": SEED, "distinct_available": len(seen),
           "briefing_sha256": hashlib.sha256(briefing.encode()).hexdigest(), "rocks": chosen}
    a.out.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({"rocks": len(chosen), "maps": len({r["map"] for r in chosen}),
                      "distinct_available": len(seen),
                      "sha256": hashlib.sha256(a.out.read_bytes()).hexdigest()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
