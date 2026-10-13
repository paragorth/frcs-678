"""Sigma_struct for a model of any size, streaming one tensor at a time from safetensors.

Never loads the model; RAM use is one matrix at a time, so 72B runs on a CPU node.

    python tools/structure_stream.py --model Qwen/Qwen2.5-7B-Instruct --out results/structure/qwen2.5-7b.csv
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from dataclasses import asdict

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import semataxy  # noqa: E402,F401
from semataxy.structure import analyze_matrix  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-dim", type=int, default=64)
    ap.add_argument("--max-side", type=int, default=20000, help="skip matrices larger than this (vocab projections)")
    ap.add_argument("--random-init", action="store_true",
                    help="size control: replace every matrix with N(0, initializer_range) of the same shape, "
                         "so the measure is scored on untrained wiring of identical dimensions")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    init_std = 0.02
    if a.random_init:
        import json as _json
        cfg = _json.load(open(os.path.join(snapshot_download(a.model, allow_patterns=["config.json"]), "config.json")))
        init_std = float(cfg.get("initializer_range", 0.02))
    local = snapshot_download(a.model, allow_patterns=["*.safetensors", "*.json"])
    files = sorted(glob.glob(os.path.join(local, "*.safetensors")))
    rows = []
    for f in files:
        # framework="pt", not "np": Qwen2.5 checkpoints are bfloat16, which numpy cannot represent.
        with safe_open(f, framework="pt") as sf:
            for name in sf.keys():
                if "embed" in name or "lm_head" in name:
                    continue
                shape = sf.get_slice(name).get_shape()
                if len(shape) != 2 or min(shape) < a.min_dim or max(shape) > a.max_side:
                    continue
                if a.random_init:
                    W = rng.normal(0.0, init_std, size=tuple(shape)).astype(np.float32)
                else:
                    W = sf.get_tensor(name).float().numpy()
                st = analyze_matrix(name, W, rng)
                rows.append({"model": a.model, "random_init": a.random_init, **asdict(st)})
                print(f"{name}: sigma {st.sigma_struct} mass {st.mass_above:.3f} alpha {st.alpha:.2f}", flush=True)
                del W
        pd.DataFrame(rows).to_csv(a.out, index=False)
    df = pd.DataFrame(rows)
    print(f"\n{a.model}: {len(df)} matrices; sigma mean {df.sigma_struct.mean():.2f} max {df.sigma_struct.max()}; "
          f"mass {df.mass_above.mean():.3f}; alpha median {df.alpha.median():.2f}")


if __name__ == "__main__":
    main()
