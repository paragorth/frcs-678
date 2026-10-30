"""The structure test on Claude Sonnet 5 and Sonnet 5.5, at low and max effort, through Anthropic's batch service.

Registered in docs/DESIGN_harvestbench_muse.md ("Stage 2: the structure test
on Sonnet 5 and Sonnet 5.5") before any of its data existed. The structure
test's 150 templates with "a hay bale" replaced by each of its 14 labels
(run_structure.prompts), plus 30 of the four cells' logged rock prompts
unchanged (rock_templates.json), under the neutral briefing as logged, for
claude-sonnet-5 and claude-sonnet-5-5 at output_config.effort low and max:
8,520 requests.

The call is the one the project's SOVSIM client makes to Sonnet: the briefing
as the system message, the prompt as the user message, output_config.effort,
and thinking left to the model (adaptive, its default); no temperature;
max_tokens 8,192 at low and 32,000 at max. No fallback to another model: a
refusal is recorded as one.

The requests go to the Message Batches API (half the standard price) in a
seeded random order, in chunks of 100, so no model, effort or label runs at a
different time from another. The spending stop works per chunk: a chunk is
sent only if the cost so far, plus the worst case of every chunk still running
and of this one (every reply as long as max_tokens allows, at batch prices,
with a 2,000-token prompt allowance), stays within the pot. If it would not, the
runner waits for running chunks to end; when even with nothing running the next
chunk would not fit, sending stops for good.
A request that errors on the server or expires is sent once more at the end.
Every result is written as one JSON line, with the text of the reply, its
stop reason and its usage.

Modes: dry builds every request and writes them with a summary (no network);
mock writes made-up answers with planted effects, for checking the analysis
(no network); live calls the API (paid).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import socket
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_structure import LABELS, PLANTED, load_templates, prompts  # noqa: E402

MODELS = {"s5": "claude-sonnet-5", "s55": "claude-sonnet-5-5"}
EFFORTS = ("low", "max")
MAX_TOKENS = {"low": 8192, "max": 32000}
PRICE_IN = 1.00 / 1e6    # batch price: half of $2 per million input tokens
PRICE_OUT = 5.00 / 1e6   # batch price: half of $10 per million output tokens, thinking included
PROMPT_ALLOWANCE = 2000  # tokens; these prompts are about 650
ORDER_SEED = 20261031
CHUNK = 100
MAX_RUNNING = 8
POLL_S = 30
STOP_SENDING_H = 8.0
CANCEL_AT_H = 10.0
POT_LIMIT = 150.00  # pg's approval for this run, 30 Oct 2026 ("$150 cap")
TEMPLATES_SHA256 = "7b73cb1cef4986d29774dd0328b15dda994f92e2a9dddf7306bd486c2915120c"
ROCKS_SHA256 = "a9ad3a0e45331802d9640ad17b4d4d79671717b2d6070c14c9f3061977a69086"


def worst_case(req: dict) -> float:
    return PROMPT_ALLOWANCE * PRICE_IN + req["params"]["max_tokens"] * PRICE_OUT


def build_requests(tdata: dict, rocks: dict) -> list[dict]:
    """Every request, with its metadata, in the seeded random order they are sent in."""
    briefing = tdata["briefing"]
    items = [(t["id"], lab, t["band"], t["map"], "structure", user)
             for t, lab, user in prompts(tdata["templates"])]
    items += [(r["id"], "rock", "rock", r["map"], "rock", r["user"]) for r in rocks["rocks"]]
    reqs = []
    for tag, model in MODELS.items():
        for effort in EFFORTS:
            for template, label, band, m, kind, user in items:
                reqs.append({
                    "custom_id": f"{tag}-{effort}-{template}-{label}",
                    "meta": {"model": model, "effort": effort, "template": template, "label": label,
                             "band": band, "map": m, "kind": kind,
                             "user_sha256": hashlib.sha256(user.encode()).hexdigest()[:16]},
                    "params": {"model": model, "max_tokens": MAX_TOKENS[effort], "system": briefing,
                               "messages": [{"role": "user", "content": user}],
                               "output_config": {"effort": effort}}})
    random.Random(ORDER_SEED).shuffle(reqs)
    ids = [r["custom_id"] for r in reqs]
    if len(set(ids)) != len(ids) or any(len(i) > 64 for i in ids):
        raise SystemExit("custom ids must be unique and at most 64 characters")
    return reqs


def mock_text(req: dict) -> str:
    """Planted answers for checking the analysis: run_structure's planted rates, a third at low."""
    m, user = req["meta"], req["params"]["messages"][0]["content"]
    u = int(hashlib.sha256(f"v|{req['custom_id']}".encode()).hexdigest(), 16) / 2**256
    if m["kind"] == "rock":
        p = 0.97
    else:
        p = PLANTED[m["label"]] / (3 if m["effort"] == "low" else 1)
    if u < 0.005:
        return "I will go around it."
    return json.dumps({"choice": "swerve" if u < p else "continue", "say": "going around" if u < p else ""})


class Spend:
    """Cost so far and the worst case of what is still running, at batch prices."""

    def __init__(self, pot: float, path: Path) -> None:
        self.pot, self.spent, self.running, self.path = pot, 0.0, {}, path
        self.stopped, self.t0 = None, time.time()

    def admit(self, worst: float) -> bool:
        """May a chunk with this worst case go now? A refusal is final only when
        nothing is running, since running chunks give back their reserve as they end."""
        if self.stopped:
            return False
        if time.time() - self.t0 > STOP_SENDING_H * 3600:
            self.stop("time")
            return False
        if self.spent + sum(self.running.values()) + worst > self.pot + 1e-9:
            if not self.running:
                self.stop("pot")
            return False
        return True

    def stop(self, why: str) -> None:
        if not self.stopped:
            self.stopped = why
            self.log({"event": "stop", "why": why, "spent": round(self.spent, 4)})

    def log(self, row: dict) -> None:
        row = {"t": round(time.time() - self.t0, 1), **row}
        with self.path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print("[spend] " + json.dumps(row), file=sys.stderr, flush=True)


def result_row(req: dict, res, batch_id: str) -> tuple[dict, float]:
    """One JSON line for one batch result, and its cost at batch prices."""
    row = {"custom_id": req["custom_id"], **req["meta"], "batch": batch_id, "result": res.type}
    cost = 0.0
    if res.type == "succeeded":
        msg = res.message
        u = msg.usage
        tin = ((u.input_tokens or 0) + (getattr(u, "cache_read_input_tokens", 0) or 0)
               + (getattr(u, "cache_creation_input_tokens", 0) or 0))
        tout = u.output_tokens or 0
        cost = tin * PRICE_IN + tout * PRICE_OUT
        row |= {"stop_reason": msg.stop_reason,
                "text": "".join(b.text for b in msg.content if b.type == "text"),
                "thinking_blocks": sum(b.type in ("thinking", "redacted_thinking") for b in msg.content),
                "in": tin, "out": tout, "cost": round(cost, 6),
                "stop_details": (msg.stop_details.model_dump() if getattr(msg, "stop_details", None) else None)}
    elif res.type == "errored":
        row |= {"error": str(res.error)[:300]}
    return row, cost


def run_live(reqs: list[dict], out: Path, pot: float) -> dict:
    import anthropic

    client = anthropic.Anthropic(max_retries=4)
    spend = Spend(pot, out.parent / "spend_ledger.jsonl")
    by_id = {r["custom_id"]: r for r in reqs}
    queue = [reqs[i:i + CHUNK] for i in range(0, len(reqs), CHUNK)]
    retried, running, done, n_chunk = set(), {}, 0, 0
    while queue or running:
        while queue and len(running) < MAX_RUNNING:
            chunk = queue[0]
            worst = sum(worst_case(r) for r in chunk)
            if not spend.admit(worst):
                break
            queue.pop(0)
            n_chunk += 1
            batch = client.messages.batches.create(
                requests=[{"custom_id": r["custom_id"], "params": r["params"]} for r in chunk])
            running[batch.id] = (chunk, n_chunk)
            spend.running[batch.id] = worst
            spend.log({"event": "sent", "chunk": n_chunk, "batch": batch.id, "requests": len(chunk),
                       "worst_case": round(worst, 2), "spent": round(spend.spent, 4)})
        if spend.stopped and not running:
            break
        time.sleep(POLL_S)
        for bid in list(running):
            b = client.messages.batches.retrieve(bid)
            if b.processing_status != "ended":
                if time.time() - spend.t0 > CANCEL_AT_H * 3600:
                    client.messages.batches.cancel(bid)
                continue
            chunk, k = running.pop(bid)
            spend.running.pop(bid, None)
            again = []
            with out.open("a") as f:
                for res in client.messages.batches.results(bid):
                    req = by_id[res.custom_id]
                    row, cost = result_row(req, res.result, bid)
                    spend.spent += cost
                    if res.result.type in ("errored", "expired") and res.custom_id not in retried:
                        retried.add(res.custom_id)
                        again.append(req)
                        row["retry"] = True
                    f.write(json.dumps(row) + "\n")
                    done += 1
            if again:
                queue.append(again)
            spend.log({"event": "ended", "chunk": k, "batch": bid, "requests": len(chunk),
                       "counts": b.request_counts.model_dump(), "spent": round(spend.spent, 4),
                       "results_written": done})
        if spend.stopped and queue and not running:
            break
    return {"spent": round(spend.spent, 4), "pot": pot, "stopped": spend.stopped,
            "chunks_sent": n_chunk, "chunks_not_sent": len(queue), "results_written": done,
            "retried": len(retried)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", required=True, choices=("dry", "mock", "live"))
    ap.add_argument("--templates", required=True, type=Path)
    ap.add_argument("--rocks", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--pot", type=float, default=None)
    a = ap.parse_args()
    tdata, rocks = load_templates(a.templates), json.loads(a.rocks.read_text())
    shas = {"templates": hashlib.sha256(a.templates.read_bytes()).hexdigest(),
            "rocks": hashlib.sha256(a.rocks.read_bytes()).hexdigest()}
    if rocks["briefing_sha256"] != tdata["briefing_sha256"]:
        raise SystemExit("the rock prompts were logged under a different briefing")
    reqs = build_requests(tdata, rocks)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    out = a.out_dir / f"results_{a.mode}.jsonl"
    worst = sum(worst_case(r) for r in reqs)
    settings = {"mode": a.mode, "models": MODELS, "efforts": list(EFFORTS), "max_tokens": MAX_TOKENS,
                "requests": len(reqs), "chunk": CHUNK, "max_running": MAX_RUNNING,
                "prices_per_million_batch": {"input": PRICE_IN * 1e6, "output": PRICE_OUT * 1e6},
                "worst_case_all_usd": round(worst, 2), "pot": a.pot, "pot_limit": POT_LIMIT,
                "order_seed": ORDER_SEED, "files": shas,
                "registered": {"templates": TEMPLATES_SHA256, "rocks": ROCKS_SHA256},
                "host": socket.gethostname(), "slurm_job": os.environ.get("SLURM_JOB_ID"),
                "python": sys.version.split()[0]}
    if a.mode == "live":
        import anthropic
        settings["anthropic_sdk"] = anthropic.__version__
        if POT_LIMIT is None or a.pot is None or not 0 < a.pot <= POT_LIMIT:
            raise SystemExit("a live run needs the registered pot")
        if shas != {"templates": TEMPLATES_SHA256, "rocks": ROCKS_SHA256}:
            raise SystemExit("the live run uses the registered templates and rock prompts")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("ANTHROPIC_API_KEY is not set")
    (a.out_dir / f"settings_{a.mode}.json").write_text(json.dumps(settings, indent=1))
    t0 = time.time()
    if a.mode == "dry":
        with (a.out_dir / "requests_dry.jsonl").open("w") as f:
            for r in reqs:
                f.write(json.dumps(r) + "\n")
        report = {"requests": len(reqs), "worst_case_all_usd": round(worst, 2),
                  "first": {k: v for k, v in reqs[0].items() if k != "params"} | {
                      "params": {k: v for k, v in reqs[0]["params"].items() if k != "system"}}}
    elif a.mode == "mock":
        out.unlink(missing_ok=True)
        with out.open("w") as f:
            for r in reqs:
                f.write(json.dumps({"custom_id": r["custom_id"], **r["meta"], "batch": "mock",
                                    "result": "succeeded", "stop_reason": "end_turn",
                                    "text": mock_text(r), "thinking_blocks": 1, "in": 650,
                                    "out": 300 if r["meta"]["effort"] == "low" else 4500,
                                    "cost": 0.0}) + "\n")
        report = {"requests": len(reqs), "results_written": len(reqs)}
    else:
        report = run_live(reqs, out, a.pot)
    report["wall_s"] = round(time.time() - t0, 1)
    (a.out_dir / f"run_{a.mode}.json").write_text(json.dumps({"settings": settings, **report}, indent=1))
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
