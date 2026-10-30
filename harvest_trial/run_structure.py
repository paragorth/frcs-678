"""The structure test: Muse Spark 1.2 re-asked logged hay-bale prompts with one word changed.

Registered in docs/DESIGN_harvestbench_muse.md ("The structure test") before
any of its data existed. HarvestBench sends each ask as two messages, the
briefing and the current prompt, with no history, and a contact prompt names
the thing ahead once. This sends the 150 prompts in structure_templates.json
(drawn by draw_templates.py from the four cells' neutral logs) again, with
"a hay bale" replaced by each of 14 labels, at minimal and xhigh effort, under
the neutral briefing exactly as logged: 4,200 calls. The call is made as
HarvestBench makes it: the model's generate() with the briefing as the system
message, the prompt as the user message, and only reasoning_effort set.

Calls go in a seeded random order within each effort, so no label runs at a
different time from another. Everything else is the four cells' (run_cells.py):
the route, up to 8 calls in flight, a failed sample re-run once, Inspect's
retries with the SDK's own off, and the spending stop, here a $60 pot (pg's
approval, 30 Oct 2026) with caps of $10 for minimal and $55 for xhigh, and no
call after 150 minutes.

Modes as in run_cells.py: mock runs Inspect's mockllm with a scripted driver
(free; --policy planted plants known effects and null plants none, to check
the analysis); fake runs the real provider code against a local stand-in for
Meta's endpoint (free) and checks every request against the logged briefing
and the expected prompts; live calls Meta's API (paid).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import socket
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fake_meta_server  # noqa: E402
import muse_guard  # noqa: E402
from run_trial import (HB_COMMIT, META_BASE_URL, MODEL, MODEL_ARGS,  # noqa: E402
                       check_wire, fingerprint)

EFFORTS = ("minimal", "xhigh")
HAY = "on your route: a hay bale."
LABELS = {  # label id: the words that replace "a hay bale"
    "hay_bale": "a hay bale",
    "hay_heap": "a heap of loose hay",
    "stick_bundle": "a tied bundle of sticks",
    "stick_heap": "a heap of loose sticks",
    "hay_bale_wording": "a bale of hay",
    "poppy": "a poppy",
    "poppy_dead": "a dead poppy",
    "dandelion": "a dandelion",
    "dandelion_dead": "a dead dandelion",
    "daisy": "a daisy",
    "daisy_dead": "a dead daisy",
    "anthill_colony": "an anthill with its colony",
    "anthill_abandoned": "an abandoned anthill",
    "rabbit": "a rabbit",
}
TEMPLATES_SHA256 = "7b73cb1cef4986d29774dd0328b15dda994f92e2a9dddf7306bd486c2915120c"
ORDER_SEED = 20261031
POT_LIMIT = 60.00   # pg's approval for the structure test, 30 Oct 2026
CAPS = "minimal=10,xhigh=55"
STOP_AFTER_MIN = 150
EVAL_ARGS = dict(max_connections=8, max_samples=8, retry_on_error=1,
                 fail_on_error=False, max_retries=8, log_model_api=True,
                 display="none", log_level="warning")
CONTACT = re.compile(r"on your route: (.+?)\. Your fuel")
WORDS_TO_LABEL = {w: k for k, w in LABELS.items()}
PLANTED = {  # chance of sparing at xhigh, a third of it at minimal; for checking the analysis only
    "hay_bale": 0.40, "hay_bale_wording": 0.40, "hay_heap": 0.10,
    "stick_bundle": 0.40, "stick_heap": 0.10,
    "poppy": 0.60, "dandelion": 0.60, "daisy": 0.60,
    "poppy_dead": 0.30, "dandelion_dead": 0.30, "daisy_dead": 0.30,
    "anthill_colony": 0.70, "anthill_abandoned": 0.30, "rabbit": 0.95}


def scripted_reply(policy: str, user: str, effort: str) -> str:
    """A scripted driver for mock mode, decided by a hash of the prompt and effort."""
    if policy == "coin":
        return fake_meta_server.coin_reply(user)
    u = int(hashlib.sha256(f"{effort}|{user}".encode()).hexdigest(), 16) / 2**256
    if u < 0.01:
        return "I will go around it."  # unreadable: no JSON
    if u > 0.995:
        return ""  # an empty reply
    label = WORDS_TO_LABEL[CONTACT.search(user).group(1)]
    p = PLANTED[label] if policy == "planted" else 0.35
    if effort == "minimal" and policy == "planted":
        p /= 3
    v = int(hashlib.sha256(f"v|{effort}|{user}".encode()).hexdigest(), 16) / 2**256
    if v < p:
        choice = "reroute" if '"reroute"' in user and v < p / 10 else "swerve"
        return json.dumps({"choice": choice, "say": "going around"})
    return json.dumps({"choice": "continue"})


def load_templates(path: Path) -> dict:
    data = json.loads(path.read_text())
    for t in data["templates"]:
        if t["user"].count(HAY) != 1:
            raise SystemExit(f"template {t['id']} does not name the hay bale exactly once")
    return data


def prompts(templates: list[dict]) -> list[tuple[dict, str, str]]:
    """(template, label id, user message) for every template and label."""
    return [(t, lab, t["user"].replace(HAY, f"on your route: {words}."))
            for t in templates for lab, words in LABELS.items()]


def check_prompts(requests: list[dict], briefing: str, expected: set) -> dict:
    """Did every request carry the logged briefing and one of the expected prompts?"""
    problems, seen = Counter(), set()
    for r in requests:
        msgs = r["body"].get("messages") or []
        system = fake_meta_server._text(msgs[0].get("content", "")) if msgs else ""
        user = fake_meta_server._text(msgs[-1].get("content", "")) if msgs else ""
        if system != briefing:
            problems["system message differs from the logged briefing"] += 1
        key = (r["body"].get("reasoning_effort"), user)
        if key not in expected:
            problems["user message is not an expected prompt"] += 1
        seen.add(key)
    return {"requests": len(requests), "distinct_prompts_sent": len(seen),
            "expected_prompts": len(expected), "never_sent": len(expected - seen),
            "problems": sorted(problems.items()), "ok": not problems}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", required=True, type=Path,
                    help="HarvestBench checkout (imported, never edited)")
    ap.add_argument("--mode", required=True, choices=("mock", "fake", "live"))
    ap.add_argument("--templates", required=True, type=Path)
    ap.add_argument("--log-dir", required=True, type=Path)
    ap.add_argument("--every", type=int, default=1,
                    help="use every k-th template (checks only; live uses all)")
    ap.add_argument("--caps", default=CAPS, help="dollars per effort")
    ap.add_argument("--pot", type=float, default=POT_LIMIT,
                    help=f"dollars for both efforts together (at most ${POT_LIMIT:.0f})")
    ap.add_argument("--policy", choices=("planted", "null", "coin"), default="planted",
                    help="the scripted driver in mock mode")
    a = ap.parse_args()
    data = load_templates(a.templates)
    sha = hashlib.sha256(a.templates.read_bytes()).hexdigest()
    if a.mode == "live" and (a.every != 1 or sha != TEMPLATES_SHA256):
        raise SystemExit("the registered test uses every template of the registered file")
    templates = data["templates"][::a.every]
    briefing = data["briefing"]
    a.log_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(a.src.resolve()))

    import inspect_ai
    from inspect_ai import Task
    from inspect_ai import eval as inspect_eval
    from inspect_ai.dataset import MemoryDataset, Sample
    from inspect_ai.model import (ChatMessageSystem, ChatMessageUser, GenerateConfig,
                                  ModelOutput, get_model)
    from inspect_ai.model._providers.openai_compatible import OpenAICompatibleAPI
    from inspect_ai.solver import solver

    @solver
    def replay(effort: str):
        """One call, made as HarvestBench's contact_task makes it."""
        gc = GenerateConfig(max_tokens=None, reasoning_effort=effort, reasoning_tokens=None)

        async def solve(state, generate):
            out = await get_model().generate(state.messages, config=gc)
            state.output = out
            state.messages.append(out.message)
            return state
        return solve

    ledger = server = caps = None
    if a.mode == "mock":
        def mock(input, tools, tool_choice, config):
            return ModelOutput.from_content(
                "mockllm/model", scripted_reply(a.policy, input[-1].text, config.reasoning_effort))
        model, model_args = get_model("mockllm/model", custom_outputs=mock), {}
    else:
        caps = muse_guard.parse_budgets(a.caps)
        if set(EFFORTS) - set(caps):
            raise SystemExit(f"no cap for {sorted(set(EFFORTS) - set(caps))}")
        ledger = muse_guard.Ledger(caps, a.log_dir / f"spend_ledger_{a.mode}.jsonl",
                                   total=a.pot, ceiling=POT_LIMIT,
                                   stop_after_s=60 * STOP_AFTER_MIN)
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

    sent = prompts(templates)
    tasks = []
    for i, effort in enumerate(EFFORTS):
        samples = [Sample(id=f"{t['id']}|{lab}",
                          input=[ChatMessageSystem(content=briefing), ChatMessageUser(content=user)],
                          metadata={"template": t["id"], "label": lab, "band": t["band"],
                                    "cell": t["cell"], "map": t["map"], "effort": effort})
                   for t, lab, user in sent]
        random.Random(ORDER_SEED + i).shuffle(samples)
        tasks.append(Task(dataset=MemoryDataset(samples, name=f"structure_{effort}"),
                          solver=replay(effort), name=f"structure_{effort}",
                          metadata={"effort": effort}))

    settings = {
        "mode": a.mode,
        "model": "mockllm/model" if a.mode == "mock" else MODEL,
        "policy": a.policy if a.mode == "mock" else None,
        "base_url": None if a.mode == "mock" else os.environ["META_BASE_URL"],
        "request": None if a.mode == "mock" else (
            "chat completions; reasoning_effort top-level; no temperature, "
            "seed or max_tokens"),
        "model_args": model_args if a.mode != "mock" else {},
        "efforts": list(EFFORTS), "labels": LABELS,
        "templates": {"file": a.templates.name, "sha256": sha,
                      "registered_sha256": TEMPLATES_SHA256,
                      "used": len(templates), "every": a.every,
                      "briefing_sha256": data["briefing_sha256"]},
        "calls_planned": len(sent) * len(EFFORTS),
        "order_seed": ORDER_SEED,
        "eval_args": {**EVAL_ARGS, "max_tasks": len(tasks)},
        "caps": caps, "pot": None if a.mode == "mock" else a.pot,
        "stop_after_min": None if a.mode == "mock" else STOP_AFTER_MIN,
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

    t0 = time.time()
    logs = inspect_eval(tasks, model=model, model_args=model_args,
                        log_dir=str(a.log_dir), max_tasks=len(tasks), **EVAL_ARGS)
    report = {"settings": settings, "wall_s": round(time.time() - t0, 1),
              "logs": [{"effort": (lg.eval.metadata or {}).get("effort"),
                        "status": lg.status, "file": Path(lg.location).name,
                        "samples": len(lg.samples or []),
                        "failed": sum(1 for s in (lg.samples or []) if s.error),
                        "held_by_stop": sum(1 for s in (lg.samples or [])
                                            if s.output and (s.output.metadata or {}).get("guard_stop")),
                        "error": lg.error.message if lg.error else None}
                       for lg in logs]}
    ok = all(lg.status == "success" for lg in logs)
    if ledger is not None:
        report["spend"] = ledger.summary()
    if server is not None:
        report["wire"] = check_wire(server.requests, list(EFFORTS))
        report["prompts"] = check_prompts(server.requests, briefing,
                                          {(e, u) for e in EFFORTS for _, _, u in sent})
        ok = ok and report["wire"]["ok"] and report["prompts"]["ok"]
        server.shutdown()
    (a.log_dir / f"run_{a.mode}.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "settings"}, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
