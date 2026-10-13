"""Summarise Experiment C: one row per model from results/structure/*.csv, and one figure.

    python tools/structure_ladder.py --out results/structure/ladder_summary.csv --fig docs/figures/structure_ladder.png

Rows are per-model means over matrices (same aggregation as T1). Parameter counts are
nominal (from the model name). Files for jobs still running are partial: pass --only to
restrict to finished ones.
"""

from __future__ import annotations

import argparse
import glob
import os
import re

import pandas as pd

PARAMS = {  # nominal billions, from the model card
    "qwen2.5-0.5b": 0.5, "qwen2.5-1.5b": 1.5, "qwen2.5-3b": 3, "qwen2.5-7b": 7, "qwen2.5-14b": 14,
    "qwen2.5-32b": 32, "qwen2.5-72b": 72,
    "qwen_qwen1.5-7b-chat": 7, "qwen_qwen2-7b-instruct": 7, "qwen_qwen2.5-7b": 7,
    "mistralai_mistral-7b-instruct-v0.3": 7, "unsloth_llama-3.1-8b-instruct": 8, "unsloth_gemma-2-9b-it": 9,
}


def summarise(paths: list[str]) -> pd.DataFrame:
    rows = []
    for p in sorted(paths):
        slug = os.path.basename(p)[:-4]
        random = slug.startswith("random_")
        key = slug[len("random_"):] if random else slug
        df = pd.read_csv(p)
        rows.append({
            "file": slug, "model": df["model"].iloc[0], "random_init": random,
            "params_b": PARAMS.get(key), "n_matrices": len(df),
            "sigma_mean": round(df["sigma_struct"].mean(), 3), "sigma_max": int(df["sigma_struct"].max()),
            "mass_above": round(df["mass_above"].mean(), 4), "alpha_median": round(df["alpha"].median(), 2),
        })
    return pd.DataFrame(rows)


def plot(summary: pd.DataFrame, fig_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ladder = summary[summary["file"].str.match(r"^qwen2\.5-[0-9.]+b$")].sort_values("params_b")
    ctrl = summary[summary["random_init"]].sort_values("params_b")
    same = summary[(~summary["random_init"]) & (~summary["file"].str.match(r"^qwen2\.5-[0-9.]+b$"))]
    q7 = summary[summary["file"] == "qwen2.5-7b"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, col, label in ((axes[0], "sigma_mean", "Σ_struct (mean octaves above shuffled edge)"),
                           (axes[1], "mass_above", "mass above noise edge (mean share)")):
        ax.plot(ladder["params_b"], ladder[col], "o-", label="Qwen2.5 instruct ladder")
        if len(ctrl):
            ax.plot(ctrl["params_b"], ctrl[col], "s--", color="grey", label="random init, same shapes")
        for _, r in pd.concat([same, q7]).iterrows():
            ax.plot([r["params_b"]], [r[col]], "^", color="C3")
            ax.annotate(r["file"].replace("unsloth_", "").replace("mistralai_", "").replace("qwen_", ""),
                        (r["params_b"], r[col]), fontsize=6, xytext=(3, 3), textcoords="offset points")
        ax.set_xscale("log"); ax.set_xlabel("parameters (billions, nominal)"); ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Experiment C: structural Semataxy across the model ladder (CSD3, Oct 2026)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    fig.savefig(fig_path, dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/structure")
    ap.add_argument("--only", nargs="*", help="file slugs to include (default: all)")
    ap.add_argument("--out", default="results/structure/ladder_summary.csv")
    ap.add_argument("--fig", default="docs/figures/structure_ladder.png")
    a = ap.parse_args()
    paths = [p for p in glob.glob(os.path.join(a.dir, "*.csv")) if not p.endswith("ladder_summary.csv")]
    if a.only:
        paths = [p for p in paths if os.path.basename(p)[:-4] in a.only]
    s = summarise(paths)
    s.to_csv(a.out, index=False)
    print(s.to_string(index=False))
    plot(s, a.fig)
    print("figure:", a.fig)


if __name__ == "__main__":
    main()
