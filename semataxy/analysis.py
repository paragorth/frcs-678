"""Metrics from a run directory. Reads only recorded transitions, never prompts.

Inputs (all written by the runner / upstream env):
  run.json        manifest, resolved config, status, parse-fallback counts
  log_env.json    upstream's per-step log: harvest rows and utterance rows
  llm_calls.jsonl per-call trace (used only for parse-fallback rates per agent)

Outputs are plain dicts so they can be dumped as JSON or stacked into a table.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

COLLAPSE_THRESHOLD = 5  # upstream: terminate when fewer than 5 tons remain


def _load(run_dir: str):
    with open(os.path.join(run_dir, "run.json")) as f:
        run = json.load(f)
    env_path = os.path.join(run_dir, "log_env.json")
    env = pd.read_json(env_path, orient="records") if os.path.exists(env_path) else pd.DataFrame()
    return run, env


def _entry_round(run: dict):
    cfg = run["resolved_config"]["experiment"]["env"]
    for p in cfg.get("perturbations") or []:
        p = p.get("perturbation", p)
        if p.get("type") == "insert_outsider":
            return int(p["round"])
    return None


def _gini(x) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    if x.sum() == 0:
        return 0.0
    n = len(x)
    return float((2 * np.sum((np.arange(1, n + 1)) * x) / (n * x.sum())) - (n + 1) / n)


def parse_fallbacks_by_agent(run_dir: str) -> dict:
    path = os.path.join(run_dir, "llm_calls.jsonl")
    counts, totals = defaultdict(int), defaultdict(int)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            totals[rec["agent"]] += 1
            if rec.get("parser_status") == "default":
                counts[rec["agent"]] += 1
    return {a: {"calls": totals[a], "fallbacks": counts[a]} for a in totals}


def analyze_run(run_dir: str) -> dict:
    run, env = _load(run_dir)
    manifest = run["manifest"]
    personas = run["resolved_config"]["experiment"]["personas"]
    slot_model = {f"persona_{i}": manifest["agents"][i] for i in range(5)}
    entry = _entry_round(run)
    residents = [f"persona_{i}" for i in range(4)] if entry is not None else [f"persona_{i}" for i in range(5)]
    newcomer = "persona_4" if entry is not None else None

    out = {
        "run_id": run["run_id"],
        "manifest": manifest["name"],
        "experiment": manifest["experiment"],
        "seed": manifest["seed"],
        "status": run.get("status"),
        "termination_reason": run.get("termination_reason"),
        "entry_round": entry,
        "slot_models": slot_model,
        "llm_calls": run.get("llm_calls"),
        "parse_fallbacks": run.get("parse_fallbacks"),
        "cache_hits": run.get("cache_hits"),
        "cache_misses": run.get("cache_misses"),
    }
    if env.empty or "action" not in env:
        out["rounds"] = 0
        return out

    h = env[env["action"] == "harvesting"].copy()
    h["round"] = h["round"].astype(int)
    rounds = sorted(h["round"].unique())
    out["rounds"] = len(rounds)

    by_round = h.groupby("round")
    stock_before = by_round["resource_in_pool_before_harvesting"].first()
    stock_after = by_round["resource_in_pool_after_harvesting"].first()
    total_catch = by_round["resource_collected"].sum()
    out["stock_before"] = stock_before.tolist()
    out["stock_after"] = stock_after.tolist()
    out["total_catch_by_round"] = total_catch.tolist()
    collapsed_rounds = [r for r in rounds if stock_after[r] < COLLAPSE_THRESHOLD]
    out["collapse_round"] = collapsed_rounds[0] if collapsed_rounds else None
    out["survival_rounds"] = int(collapsed_rounds[0]) + 1 if collapsed_rounds else len(rounds)
    out["mean_stock_after"] = float(stock_after.mean())
    out["group_yield"] = int(total_catch.sum())

    per_agent = h.groupby("agent_id")["resource_collected"].sum()
    wanted = h.groupby("agent_id")["wanted_resource"].sum()
    out["catch_by_agent"] = per_agent.to_dict()
    out["wanted_by_agent"] = wanted.to_dict()
    out["gini_catch"] = _gini(per_agent.reindex(sorted(per_agent.index)).values)

    # per-round per-agent matrix for convergence measures
    mat = h.pivot_table(index="round", columns="agent_id", values="resource_collected", aggfunc="first")
    out["catch_matrix"] = {str(r): {a: int(v) for a, v in row.items() if not pd.isna(v)}
                           for r, row in mat.iterrows()}

    if entry is not None:
        pre = mat[mat.index < entry]
        post = mat[mat.index >= entry]
        res_cols = [c for c in residents if c in mat.columns]
        out["pre_entry_rounds"] = int(len(pre))
        out["post_entry_rounds"] = int(len(post))
        out["eligible_for_entry"] = bool(len(post) > 0)
        if len(pre):
            out["resident_mean_catch_pre"] = float(pre[res_cols].stack().mean())
        if len(post):
            out["resident_mean_catch_post"] = float(post[res_cols].stack().mean())
            if newcomer in post.columns:
                nc = post[newcomer].dropna()
                out["newcomer_mean_catch"] = float(nc.mean()) if len(nc) else None
                out["newcomer_total_catch"] = int(nc.sum())
                tot_post = post.stack().sum()
                out["resident_share_post"] = float(post[res_cols].stack().sum() / tot_post) if tot_post else None
                # convergence: do residents move toward the newcomer's level after entry?
                gap = (post[res_cols].mean(axis=1) - post[newcomer]).abs()
                out["resident_newcomer_gap_by_round"] = gap.round(2).tolist()
                if len(gap) >= 2:
                    out["resident_newcomer_gap_trend"] = float(gap.iloc[-1] - gap.iloc[0])
            # residents' own change relative to their pre-entry level
            if len(pre):
                out["resident_catch_delta"] = out["resident_mean_catch_post"] - out["resident_mean_catch_pre"]

    limits = env[env["action"] == "conversation_resource_limit"]
    if len(limits):
        out["agreed_limits_by_round"] = limits.set_index("round")["resource_limit"].apply(
            lambda v: None if pd.isna(v) else int(v)).to_dict()

    # parse fallbacks per agent, keyed by persona slot
    name_to_slot = {personas[f"persona_{i}"]["name"]: f"persona_{i}" for i in range(5)}
    pf = parse_fallbacks_by_agent(run_dir)
    out["parse_fallbacks_by_slot"] = {name_to_slot.get(k, k): v for k, v in pf.items()}
    return out


def summarize_runs(run_dirs: list) -> pd.DataFrame:
    rows = []
    for d in run_dirs:
        a = analyze_run(d)
        rows.append({
            "run_id": a["run_id"], "manifest": a["manifest"], "seed": a["seed"],
            "status": a["status"], "termination": a["termination_reason"],
            "rounds": a.get("rounds"), "collapse_round": a.get("collapse_round"),
            "survival_rounds": a.get("survival_rounds"), "mean_stock_after": a.get("mean_stock_after"),
            "group_yield": a.get("group_yield"), "gini_catch": a.get("gini_catch"),
            "entry_round": a.get("entry_round"),
            "resident_pre": a.get("resident_mean_catch_pre"), "resident_post": a.get("resident_mean_catch_post"),
            "newcomer_mean": a.get("newcomer_mean_catch"), "resident_share_post": a.get("resident_share_post"),
            "gap_trend": a.get("resident_newcomer_gap_trend"),
            "parse_fallbacks": a.get("parse_fallbacks"), "llm_calls": a.get("llm_calls"),
            "slot4_model": a["slot_models"]["persona_4"],
        })
    return pd.DataFrame(rows)
