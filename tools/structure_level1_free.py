"""Dimension-free level-1 summary of results/structure/*.csv.

Sigma_struct (octave span) grows with matrix width, so it is not comparable across shapes.
This script reports per model the quantities that are not, or much less, width-dependent:

  alpha        tail exponent of the above-edge spectrum (a slope; Martin & Mahoney's HT-SR band
               is roughly 2-4; lower = heavier tail = "better trained" in their data)
  frac_above   fraction of eigen-directions above the shuffled-entries edge
  mass_above   share of total spectral mass above the edge
  conc         concentration = mass_above / frac_above: mass carried per above-edge direction,
               relative to an even spread (1 = uniform; large = few directions carry the mass)
  span_norm    sigma_struct / log2(min dim): octaves per available doubling of scale

Each is given as median and interquartile range across matrices, not a mean, and the shape
mix is reported so that like is compared with like.

    python tools/structure_level1_free.py [--only slug ...]
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd

D = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "structure")


def load(path: str) -> pd.DataFrame:
    d = pd.read_csv(path)
    d["mind"] = d["shape"].apply(lambda s: min(eval(s)))
    d["conc"] = d["mass_above"] / d["frac_above"].replace(0, np.nan)
    d["span_norm"] = d["sigma_struct"] / np.log2(d["mind"])
    return d


def iqr(x):
    x = x.dropna()
    return f"{x.median():.2f} [{x.quantile(.25):.2f}, {x.quantile(.75):.2f}]" if len(x) else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--out", default=os.path.join(D, "level1_free_summary.csv"))
    a = ap.parse_args()
    paths = sorted(p for p in glob.glob(os.path.join(D, "*.csv")) if "summary" not in p)
    if a.only:
        paths = [p for p in paths if os.path.basename(p)[:-4] in a.only]
    rows = []
    for p in paths:
        slug = os.path.basename(p)[:-4]
        d = load(p)
        if slug.startswith("random_"):
            continue
        rows.append({
            "model": slug, "n": len(d), "min_dim_median": int(d["mind"].median()),
            "alpha": iqr(d["alpha"]), "frac_above": iqr(d["frac_above"]), "mass_above": iqr(d["mass_above"]),
            "conc": iqr(d["conc"]), "span_norm": iqr(d["span_norm"]),
            "_alpha": d["alpha"].median(), "_mass": d["mass_above"].median(), "_conc": d["conc"].median(),
            "_span_norm": d["span_norm"].median(), "_frac": d["frac_above"].median(),
        })
    t = pd.DataFrame(rows)
    t.to_csv(a.out, index=False)
    pd.set_option("display.width", 250)
    print(t[["model", "n", "min_dim_median", "alpha", "frac_above", "mass_above", "conc", "span_norm"]].to_string(index=False))


if __name__ == "__main__":
    main()
