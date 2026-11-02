"""Registered analysis of extensions B (makers or the made) and C (a second family, Llama).

  python analyse_ext.py --out out/ext/analysis
B1: across the six maker pairs, the per-pair slope of the maker share on log2(parameters), Qwen2.5 0.5B to
    32B, is positive (one-sided Wilcoxon, zeros dropped).
B2 (secondary): at 32B, the number of pairs where the maker is chosen in more than half of decisions
    (one-sided sign test against 1/2).
C1: across the main test's 24 cases, Llama-3.3-70B chooses the semataxy option more often than
    Llama-3.1-8B (paired one-sided Wilcoxon on per-case shares, zeros dropped).
Each has its own family; no correction across them. Everything else is descriptive.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
SIZES = {"q0.5b": 0.5, "q1.5b": 1.5, "q3b": 3.0, "q7b": 7.0, "q14b": 14.0, "q32b": 32.0}


def load(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def share(rr: list[dict]) -> float:
    got = [r["is_s"] for r in rr if r["is_s"] is not None]
    return sum(got) / len(got) if got else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    res, L = {}, []

    # Extension B on the Qwen2.5 ladder.
    mk = {l: load(HERE / f"out/makers/{l}.jsonl") for l in SIZES}
    labels = [l for l in SIZES if mk[l]]
    pairs = [f"M{i}" for i in range(1, 7)]
    logx = [math.log2(SIZES[l]) for l in labels]
    table = {p: [share([r for r in mk[l] if r["case"] == p and r["kind"] == "choice"]) for l in labels] for p in pairs}
    slopes = {p: float(np.polyfit(logx, table[p], 1)[0]) for p in pairs}
    nz = [v for v in slopes.values() if abs(v) > 1e-12]
    b1 = float(stats.wilcoxon(nz, alternative="greater").pvalue) if nz else 1.0
    top = [table[p][-1] for p in pairs] if labels and labels[-1] == "q32b" else []
    k = sum(x > 0.5 for x in top)
    b2 = float(stats.binomtest(k, len(top), 0.5, alternative="greater").pvalue) if top else None
    res["B"] = {"labels": labels, "maker_share": table, "slopes": slopes, "B1_p": b1, "B1_up": sum(v > 0 for v in nz),
                "B1_nonzero": len(nz), "B2_pairs_maker_majority_at_32B": k, "B2_p": b2}
    L.append("Extension B, the maker chosen (choice prompts, versions and orders pooled):")
    L.append("| Pair | " + " | ".join(f"{SIZES[l]:g}B" for l in labels) + " | Slope | Fact at 32B | People at 32B |")
    L.append("|---|" + "---|" * (len(labels) + 3))
    for p in pairs:
        fct = share([r for r in mk["q32b"] if r["case"] == p and r["kind"] == "fact"]) if mk["q32b"] else float("nan")
        ppl = share([r for r in mk["q32b"] if r["case"] == p and r["kind"] == "people"]) if mk["q32b"] else float("nan")
        L.append(f"| {p} | " + " | ".join(f"{100 * x:.0f}%" for x in table[p]) + f" | {slopes[p]:+.3f} | {100 * fct:.0f}% | {100 * ppl:.0f}% |")
    L.append(f"B1: {res['B']['B1_up']}/{len(nz)} pairs rise, one-sided Wilcoxon p = {b1:.3g}. "
             f"B2: maker chosen by majority in {k}/{len(top)} pairs at 32B, sign test p = {b2 if b2 is None else f'{b2:.3g}'}.")
    for extra in ("q3.6-27b-off", "l8b", "l70b"):
        rr = load(HERE / f"out/makers/{extra}.jsonl")
        if rr:
            L.append(f"{extra}: " + ", ".join(f"{p} {100 * share([r for r in rr if r['case'] == p and r['kind'] == 'choice']):.0f}%" for p in pairs))

    # Extension C: Llama 8B against 70B on the main prompts.
    l8, l70 = load(HERE / "out/l8b.jsonl"), load(HERE / "out/l70b.jsonl")
    if l8 and l70:
        cases = sorted({r["case"] for r in l8}, key=lambda c: ("IORC".index(c[0]), c))
        s8 = {c: share([r for r in l8 if r["case"] == c and r["kind"] == "choice"]) for c in cases}
        s70 = {c: share([r for r in l70 if r["case"] == c and r["kind"] == "choice"]) for c in cases}
        d = [s70[c] - s8[c] for c in cases if abs(s70[c] - s8[c]) > 1e-12]
        c1 = float(stats.wilcoxon(d, alternative="greater").pvalue) if d else 1.0
        res["C"] = {"l8b": s8, "l70b": s70, "C1_p": c1, "C1_up": sum(x > 0 for x in d), "C1_nonzero": len(d)}
        L.append("")
        L.append("Extension C, Llama, the semataxy option chosen on the main 24 cases:")
        L.append("| Model | All | Instinct | Order | Route | Ceiling | Fact | People |")
        L.append("|---|---|---|---|---|---|---|---|")
        for name, rr in (("Llama-3.1-8B", l8), ("Llama-3.3-70B (AWQ)", l70)):
            ch = [r for r in rr if r["kind"] == "choice"]
            L.append(f"| {name} | {100 * share(ch):.0f}% | " + " | ".join(f"{100 * share([r for r in ch if r['cat'] == k]):.0f}%" for k in "IORC")
                     + f" | {100 * share([r for r in rr if r['kind'] == 'fact']):.0f}% | {100 * share([r for r in rr if r['kind'] == 'people']):.0f}% |")
        L.append(f"C1: 70B above 8B in {res['C']['C1_up']}/{len(d)} cases that differ, one-sided Wilcoxon p = {c1:.3g}.")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.with_suffix(".json").write_text(json.dumps(res, indent=1))
    a.out.with_suffix(".md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
