"""The four registered cells: Muse Spark 1.2 at minimal and xhigh effort, under
HarvestBench's neutral and morality briefings, 30 shifts each (map seeds 0-29).

Registered in docs/DESIGN_harvestbench_muse.md before any of their data
existed. Every setting is the trial's (run_trial.py) except: both briefings;
seeds 0-29; up to 8 calls in flight, 4 shifts at a time per cell; a shift that
fails is re-run once and never stops its cell; and the spending stop has a $100
pot (pg's approval, 30 Oct 2026) with a cap per effort that both briefings
share.

Modes as in run_trial.py: mock runs Inspect's mockllm with a scripted driver
(free); fake runs the real provider code against a local stand-in for Meta's
endpoint (free); live calls Meta's API (paid).
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fake_meta_server  # noqa: E402
import muse_guard  # noqa: E402
from run_trial import (HB_COMMIT, META_BASE_URL, MODEL, MODEL_ARGS,  # noqa: E402
                       check_wire, fingerprint)

ARMS = ("neutral", "morality")
EFFORTS = ("minimal", "xhigh")
SEEDS = tuple(range(30))
POT_LIMIT = 100.00   # pg's approval for the four cells, 30 Oct 2026
CAPS = "minimal=20,xhigh=90"
TASK_ARGS = dict(detour_costs=(12,), price_mult=1.0, max_output_tokens=None)
EVAL_ARGS = dict(max_connections=8, max_samples=4, retry_on_error=1,
                 fail_on_error=False, max_retries=8, log_model_api=True,
                 display="none", log_level="warning")
POLICIES = {"muse-like": fake_meta_server.muse_like_reply,
            "coin": fake_meta_server.coin_reply,
            "cautious": fake_meta_server.cautious_reply}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", required=True, type=Path,
                    help="HarvestBench checkout (imported, never edited)")
    ap.add_argument("--mode", required=True, choices=("mock", "fake", "live"))
    ap.add_argument("--log-dir", required=True, type=Path)
    ap.add_argument("--seeds", default="0-29", help="map seeds; live runs 0-29 only")
    ap.add_argument("--caps", default=CAPS, help="dollars per effort, both briefings")
    ap.add_argument("--pot", type=float, default=POT_LIMIT,
                    help=f"dollars for all cells together (at most ${POT_LIMIT:.0f})")
    ap.add_argument("--policy", choices=sorted(POLICIES), default="coin",
                    help="the scripted driver in mock mode")
    a = ap.parse_args()
    lo, _, hi = a.seeds.partition("-")
    seeds = tuple(range(int(lo), int(hi or lo) + 1))
    if a.mode == "live" and seeds != SEEDS:
        raise SystemExit("the registered cells are seeds 0-29")
    a.log_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(a.src.resolve()))

    import inspect_ai
    from inspect_ai import eval as inspect_eval
    from inspect_ai.model import ModelOutput, get_model
    from inspect_ai.model._providers.openai_compatible import OpenAICompatibleAPI

    from harvest.contact_task import BRIEFING_VERSION, harvest_contact

    ledger = server = caps = None
    if a.mode == "mock":
        reply = POLICIES[a.policy]

        def mock(input, tools, tool_choice, config):
            return ModelOutput.from_content("mockllm/model", reply(input[-1].text))
        model, model_args = get_model("mockllm/model", custom_outputs=mock), {}
    else:
        caps = muse_guard.parse_budgets(a.caps)
        if set(EFFORTS) - set(caps):
            raise SystemExit(f"no cap for {sorted(set(EFFORTS) - set(caps))}")
        ledger = muse_guard.Ledger(caps, a.log_dir / f"spend_ledger_{a.mode}.jsonl",
                                   total=a.pot, ceiling=POT_LIMIT)
        muse_guard.install(OpenAICompatibleAPI, ledger)
        if a.mode == "fake":
            server = fake_meta_server.start()
            os.environ["META_BASE_URL"] = f"http://127.0.0.1:{server.server_port}/v1"
            os.environ["META_API_KEY"] = "not-a-key-local-stand-in"
        else:
            os.environ.setdefault("META_BASE_URL", META_BASE_URL)
            if not os.environ.get("META_API_KEY"):
                raise SystemExit("META_API_KEY is not set")
        model, model_args = MODEL, MODEL_ARGS

    task_args = {**TASK_ARGS, "seeds": seeds}
    cells = [(arm, e) for arm in ARMS for e in EFFORTS]
    settings = {
        "mode": a.mode,
        "model": "mockllm/model" if a.mode == "mock" else MODEL,
        "policy": a.policy if a.mode == "mock" else None,
        "base_url": None if a.mode == "mock" else os.environ["META_BASE_URL"],
        "request": None if a.mode == "mock" else (
            "chat completions; reasoning_effort top-level; no temperature, "
            "seed or max_tokens"),
        "model_args": model_args if a.mode != "mock" else {},
        "cells": [f"{arm}/{e}" for arm, e in cells],
        "task_args": {**task_args, "briefing_version": BRIEFING_VERSION,
                      "max_calls": 160, "shuffle_options": True},
        "eval_args": {**EVAL_ARGS, "max_tasks": len(cells)},
        "caps": caps, "pot": None if a.mode == "mock" else a.pot,
        "reserve_per_call": round(muse_guard.RESERVE, 4),
        "prices_per_million": {"input": muse_guard.PRICE_IN * 1e6,
                               "output": muse_guard.PRICE_OUT * 1e6},
        "harvestbench": {"commit": HB_COMMIT, "src": str(a.src),
                         "sha256": fingerprint(a.src)},
        "inspect_ai": inspect_ai.__version__,
        "python": sys.version.split()[0],
        "host": socket.gethostname(),
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "inspect_http_request_timeout": os.environ.get("INSPECT_HTTP_REQUEST_TIMEOUT"),
    }
    (a.log_dir / f"settings_{a.mode}.json").write_text(json.dumps(settings, indent=1))

    tasks = [harvest_contact(arm=arm, reasoning_effort=e, **task_args)
             for arm, e in cells]
    t0 = time.time()
    logs = inspect_eval(tasks, model=model, model_args=model_args,
                        log_dir=str(a.log_dir), max_tasks=len(tasks), **EVAL_ARGS)
    report = {"settings": settings, "wall_s": round(time.time() - t0, 1),
              "logs": [{"cell": f"{lg.eval.task_args.get('arm')}/"
                                f"{lg.eval.task_args.get('reasoning_effort')}",
                        "status": lg.status, "file": Path(lg.location).name,
                        "shifts_scored": sum(1 for s in (lg.samples or []) if s.scores),
                        "shifts_failed": sum(1 for s in (lg.samples or []) if s.error),
                        "error": lg.error.message if lg.error else None}
                       for lg in logs]}
    ok = all(lg.status == "success" for lg in logs)
    if ledger is not None:
        report["spend"] = ledger.summary()
    if server is not None:
        report["wire"] = check_wire(server.requests, list(EFFORTS))
        ok = ok and report["wire"]["ok"]
        server.shutdown()
    (a.log_dir / f"run_{a.mode}.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "settings"}, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
