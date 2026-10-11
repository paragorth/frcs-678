"""Sigma_struct across a real language model's training run.

EleutherAI's Pythia suite publishes checkpoints throughout pretraining (revision
"step{N}"). This computes Sigma_struct for each, giving structure-vs-training-step
on a real LM without training anything. Prediction: monotone rise, tracking the
loss curve; attention q/o first.

    python tools/pythia_trajectory.py --model EleutherAI/pythia-160m --out results/pythia160m_trajectory.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import semataxy  # noqa: E402,F401
from semataxy.structure import analyze_matrix  # noqa: E402

DEFAULT_STEPS = [0, 128, 512, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 143000]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-160m")
    ap.add_argument("--steps", type=int, nargs="*", default=DEFAULT_STEPS)
    ap.add_argument("--out", default="results/pythia160m_trajectory.csv")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    rows = []
    for step in a.steps:
        try:
            m = AutoModelForCausalLM.from_pretrained(a.model, revision=f"step{step}", dtype=torch.float32)
        except Exception as e:  # noqa: BLE001
            print(f"step {step}: unavailable ({type(e).__name__})", flush=True)
            continue
        for name, p in m.named_parameters():
            if p.ndim == 2 and min(p.shape) >= 64 and "embed" not in name:
                st = analyze_matrix(name, p.detach().cpu().numpy(), rng)
                rows.append({"step": step, **asdict(st)})
        df = pd.DataFrame(rows)
        g = df[df.step == step]
        print(f"step {step:6d}: sigma {g.sigma_struct.mean():.2f} max {g.sigma_struct.max()} "
              f"mass {g.mass_above.mean():.3f} ({len(g)} matrices)", flush=True)
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        df.to_csv(a.out, index=False)
        del m


if __name__ == "__main__":
    main()
