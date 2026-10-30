"""Trial shifts of HarvestBench's published task on Meta's Muse Spark 1.2.

HarvestBench (Brazilek et al., arXiv 2609.04444; MIT licence; commit 526de705)
is imported from --src and never edited. What this trial sets is all here:

  model   Inspect's OpenAI-compatible provider, openai-api/meta/muse-spark-1.2,
          at META_BASE_URL with META_API_KEY: a plain chat-completions request
          with reasoning_effort as a top-level field and no temperature, seed
          or max_tokens, which is what our SOVSIM client sends Muse
  effort  one shift each at minimal, medium and xhigh, run side by side
  cap     none per reply (max_output_tokens=None, so no max_tokens is sent);
          spending is bounded by muse_guard's dollar stop instead: a cap per
          shift and a pot for the whole trial, at most $5

and, as the published panel calls the task (scripts/run_variance.py): arm
"morality", detour_costs=(12,), price_mult=1.0, here one seed (0); briefing
version 1, max_calls 160 and shuffled options are the task's defaults.

Modes: mock runs Inspect's mockllm (free); fake runs the real provider code
against a local stand-in for Meta's endpoint (free) to check what goes on the
wire and that the stop works; live calls Meta's API (paid).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fake_meta_server  # noqa: E402
import muse_guard  # noqa: E402

MODEL = "openai-api/meta/muse-spark-1.2"
META_BASE_URL = "https://api.meta.ai/v1"
HB_COMMIT = "526de705"
TASK_ARGS = dict(arm="morality", detour_costs=(12,), seeds=(0,),
                 price_mult=1.0, max_output_tokens=None)
# The SDK's own silent retries are off, so every attempt passes the stop;
# Inspect retries instead, and logs each attempt.
MODEL_ARGS = {"max_retries": 0}
EVAL_ARGS = dict(max_connections=3, retry_on_error=0, max_retries=8,
                 log_model_api=True, display="none", log_level="warning")
NOT_TO_SEND = ("temperature", "seed", "max_tokens", "max_completion_tokens",
               "top_p", "n", "stop")


def fingerprint(src: Path) -> str:
    """sha256 over the HarvestBench files the trial imports or reads."""
    h = hashlib.sha256()
    files = sorted([*src.glob("harvest/*.py"),
                    *src.glob("harvest/docs/instructions/*.md"),
                    *(src / "scripts" / f for f in
                      ("panel.py", "validate_cells.py", "build_cache.py"))])
    for p in files:
        h.update(p.relative_to(src).as_posix().encode() + b"\0")
        h.update(p.read_bytes())
    return h.hexdigest()


def check_wire(requests: list[dict], efforts: list[str]) -> dict:
    """What the provider actually sent to the stand-in endpoint."""
    problems = []
    for r in requests:
        b = r["body"]
        if r["path"] != "/v1/chat/completions":
            problems.append(f"path {r['path']}")
        if not r["authorization_sent"]:
            problems.append("no Authorization header")
        if b.get("model") != "muse-spark-1.2":
            problems.append(f"model {b.get('model')!r}")
        if b.get("reasoning_effort") not in efforts:
            problems.append(f"reasoning_effort {b.get('reasoning_effort')!r}")
        problems += [f"{k} sent" for k in NOT_TO_SEND if k in b]
        roles = [m.get("role") for m in b.get("messages", [])]
        if roles != ["system", "user"]:
            problems.append(f"roles {roles}")
    return {"requests": len(requests),
            "fields_sent": sorted({k for r in requests for k in r["body"]}),
            "by_effort": dict(Counter(r["body"].get("reasoning_effort")
                                      for r in requests)),
            "problems": sorted(Counter(problems).items()),
            "ok": not problems}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", required=True, type=Path,
                    help="HarvestBench checkout (imported, never edited)")
    ap.add_argument("--mode", required=True, choices=("mock", "fake", "live"))
    ap.add_argument("--log-dir", required=True, type=Path)
    ap.add_argument("--efforts", default="minimal,medium,xhigh")
    ap.add_argument("--seeds", default="0",
                    help="map seeds, e.g. 0 or 0-29 (more than seed 0: mock only)")
    ap.add_argument("--policy", choices=("cautious", "muse-like", "muse-holds"),
                    default="cautious",
                    help="the scripted driver in mock mode (muse-holds: its "
                         "'stay' keeps the tractor put instead of being overridden)")
    ap.add_argument("--budgets", default="minimal=1.00,medium=2.50,xhigh=3.50",
                    help="cap in dollars per shift, keyed by effort")
    ap.add_argument("--pot", type=float, default=muse_guard.TOTAL_LIMIT,
                    help="dollars for all shifts together (at most $5)")
    ap.add_argument("--stop-after-min", type=float, default=None,
                    help="send no call after this many minutes (job time limit)")
    a = ap.parse_args()
    efforts = a.efforts.split(",")
    lo, _, hi = a.seeds.partition("-")
    seeds = tuple(range(int(lo), int(hi or lo) + 1))
    if seeds != (0,) and a.mode != "mock":
        raise SystemExit("the paid trial is seed 0 only")
    task_args = {**TASK_ARGS, "seeds": seeds}
    a.log_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(a.src.resolve()))

    import inspect_ai
    from inspect_ai import eval as inspect_eval
    from inspect_ai.model import ModelOutput, get_model
    from inspect_ai.model._providers.openai_compatible import OpenAICompatibleAPI

    from harvest.contact_task import BRIEFING_VERSION, harvest_contact

    ledger = server = None
    budgets = None
    if a.mode == "mock":
        reply = {"cautious": fake_meta_server.cautious_reply,
                 "muse-like": fake_meta_server.muse_like_reply,
                 "muse-holds": lambda u: fake_meta_server.muse_like_reply(u, holds=True),
                 }[a.policy]

        def mock(input, tools, tool_choice, config):
            return ModelOutput.from_content("mockllm/model", reply(input[-1].text))
        model, model_args = get_model("mockllm/model", custom_outputs=mock), {}
    else:
        budgets = muse_guard.parse_budgets(a.budgets)
        if set(efforts) - set(budgets):
            raise SystemExit(f"no budget for {sorted(set(efforts) - set(budgets))}")
        ledger = muse_guard.Ledger(
            budgets, a.log_dir / f"spend_ledger_{a.mode}.jsonl", total=a.pot,
            stop_after_s=None if a.stop_after_min is None else 60 * a.stop_after_min)
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

    settings = {
        "mode": a.mode,
        "model": "mockllm/model" if a.mode == "mock" else MODEL,
        "base_url": None if a.mode == "mock" else os.environ["META_BASE_URL"],
        "request": None if a.mode == "mock" else (
            "chat completions; reasoning_effort top-level; no temperature, "
            "seed or max_tokens"),
        "model_args": model_args if a.mode != "mock" else {},
        "efforts": efforts,
        "policy": a.policy if a.mode == "mock" else None,
        "task_args": {**task_args, "briefing_version": BRIEFING_VERSION,
                      "max_calls": 160, "shuffle_options": True},
        "eval_args": {**EVAL_ARGS, "max_tasks": len(efforts)},
        "budgets": budgets,
        "pot": None if a.mode == "mock" else a.pot,
        "reserve_per_call": round(muse_guard.RESERVE, 4),
        "prices_per_million": {"input": muse_guard.PRICE_IN * 1e6,
                               "output": muse_guard.PRICE_OUT * 1e6},
        "stop_after_min": a.stop_after_min,
        "harvestbench": {"commit": HB_COMMIT, "src": str(a.src),
                         "sha256": fingerprint(a.src)},
        "inspect_ai": inspect_ai.__version__,
        "python": sys.version.split()[0],
        "host": socket.gethostname(),
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "inspect_http_request_timeout": os.environ.get("INSPECT_HTTP_REQUEST_TIMEOUT"),
    }
    (a.log_dir / f"settings_{a.mode}.json").write_text(json.dumps(settings, indent=1))

    tasks = [harvest_contact(reasoning_effort=e, **task_args) for e in efforts]
    t0 = time.time()
    logs = inspect_eval(tasks, model=model, model_args=model_args,
                        log_dir=str(a.log_dir), max_tasks=len(tasks), **EVAL_ARGS)
    report = {"settings": settings, "wall_s": round(time.time() - t0, 1),
              "logs": [{"effort": lg.eval.task_args.get("reasoning_effort"),
                        "status": lg.status, "file": Path(lg.location).name,
                        "error": lg.error.message if lg.error else None,
                        "asks_by_seed": {s.metadata["seed"]: s.store.get("model_calls")
                                         for s in (lg.samples or [])}}
                       for lg in logs]}
    ok = all(lg.status == "success" for lg in logs)
    if ledger is not None:
        report["spend"] = ledger.summary()
    if server is not None:
        (a.log_dir / "fake_requests.json").write_text(json.dumps(server.requests))
        report["wire"] = check_wire(server.requests, efforts)
        ok = ok and report["wire"]["ok"]
        server.shutdown()
    (a.log_dir / f"run_{a.mode}.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
