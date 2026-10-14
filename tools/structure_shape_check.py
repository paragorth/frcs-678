"""Does Sigma_struct depend on matrix shape? Matched-pair and shape-adjusted checks on results/structure/*.csv.

    python tools/structure_shape_check.py

1. Qwen2-7B vs Qwen2.5-7B: identical architecture, one generation of training apart. Paired per matrix.
2. Qwen2.5-7B base vs instruct: same weights before and after instruction tuning.
3. Pooled over every trained model: Sigma_struct against the smaller matrix dimension, and a
   shape-adjusted score per model (residual after a linear fit on log2 of the smaller dimension).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

D = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "structure")

LADDER = ["qwen2.5-0.5b", "qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b", "qwen2.5-14b", "qwen2.5-32b", "qwen2.5-72b"]
SAME = ["qwen_qwen1.5-7b-chat", "qwen_qwen2-7b-instruct", "qwen_qwen2.5-7b", "mistralai_mistral-7b-instruct-v0.3",
        "unsloth_llama-3.1-8b-instruct", "unsloth_gemma-2-9b-it"]


def load(f):
    d = pd.read_csv(os.path.join(D, f + ".csv"))
    d["mind"] = d["shape"].apply(lambda s: min(eval(s)))
    return d


def main():
    a, b = load("qwen_qwen2-7b-instruct"), load("qwen2.5-7b")
    m = a.merge(b, on="name", suffixes=("_q2", "_q25"))
    ds = m.sigma_struct_q25 - m.sigma_struct_q2
    print(f"Qwen2-7B -> Qwen2.5-7B, identical shapes, n={len(m)} matrices")
    print(f"  sigma mean {m.sigma_struct_q2.mean():.3f} -> {m.sigma_struct_q25.mean():.3f}; paired diff {ds.mean():+.3f} "
          f"(sd {ds.std():.2f}); up {(ds > 0).mean():.2f}, down {(ds < 0).mean():.2f}, same {(ds == 0).mean():.2f}")
    dm = m.mass_above_q25 - m.mass_above_q2
    print(f"  mass  mean {m.mass_above_q2.mean():.3f} -> {m.mass_above_q25.mean():.3f}; paired diff {dm.mean():+.4f}")

    c = load("qwen_qwen2.5-7b")
    m2 = c.merge(b, on="name", suffixes=("_base", "_inst"))
    print(f"Qwen2.5-7B base -> instruct: sigma {m2.sigma_struct_base.mean():.3f} -> {m2.sigma_struct_inst.mean():.3f}; "
          f"mass {m2.mass_above_base.mean():.3f} -> {m2.mass_above_inst.mean():.3f}")

    all_ = pd.concat([load(f).assign(file=f) for f in LADDER + SAME])
    print("\nPooled over all trained models: Sigma_struct by smaller matrix dimension")
    print(all_.groupby("mind").agg(n=("sigma_struct", "size"), sigma=("sigma_struct", "mean"),
                                   mass=("mass_above", "mean")).round(3).to_string())
    X = np.log2(all_.mind)
    print(f"\ncorr(sigma, log2 min dim) = {np.corrcoef(all_.sigma_struct, X)[0, 1]:.3f}")
    k, b0 = np.polyfit(X, all_.sigma_struct, 1)
    all_["adj"] = all_.sigma_struct - (k * X + b0)
    print(f"slope: {k:+.3f} octaves per doubling of the smaller dimension")
    print("\nShape-adjusted score per model (mean residual; 0 = what its matrix sizes alone predict)")
    t = all_.groupby("file").agg(n=("adj", "size"), sigma=("sigma_struct", "mean"), adjusted=("adj", "mean"),
                                 min_dim_median=("mind", "median")).round(3)
    print(t.loc[[f for f in LADDER + SAME if f in t.index]].to_string())


if __name__ == "__main__":
    main()
