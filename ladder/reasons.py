"""Sample the stated reasons of choices for reading (reported, not tested).

  python reasons.py --results out/*.jsonl --cases I1 C2 --labels q0.5b q32b --n 3
Prints, for each case and size, a few choices for each option with their one-sentence reasons.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True, type=Path)
    ap.add_argument("--cases", nargs="*", default=[])
    ap.add_argument("--labels", nargs="*", default=[])
    ap.add_argument("--kind", default="choice")
    ap.add_argument("--n", type=int, default=2)
    a = ap.parse_args()
    rows = []
    for p in a.results:
        rows += [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r["kind"] == a.kind and (not a.cases or r["case"] in a.cases)
            and (not a.labels or r["label"] in a.labels)]
    rng = random.Random(0)
    keys = sorted({(r["case"], r["label"]) for r in rows})
    for case, label in keys:
        rr = [r for r in rows if r["case"] == case and r["label"] == label]
        for side, want in (("semataxy", True), ("foil", False)):
            pick = [r for r in rr if r["is_s"] is want]
            rng.shuffle(pick)
            for r in pick[: a.n]:
                text = r["text"].split("</think>")[-1].strip().replace("\n", " ")
                print(f"{case} {label} {r['version'][:4]} {side:8s} | {text[:220]}")
        print()


if __name__ == "__main__":
    main()
