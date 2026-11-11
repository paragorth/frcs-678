"""The pair test: DeepSeek V4 Pro against V4 Flash, one training at two sizes, thinking off.

Registered in docs/DESIGN_harvestbench_muse.md ("The pair test") before any
of its data existed. The structure test's 150 logged templates with "a hay
bale" replaced by each of 14 labels (run_structure.prompts), plus the 30
logged rock prompts unchanged (rock_templates.json), under the neutral
briefing as logged, sent once to each model through OpenRouter with reasoning
effort "none": 2,130 calls a model, 4,260 in all, in one seeded random order
across both models so neither runs at a different time from the other. Each
call is two messages, the briefing as system and the prompt as user, as
HarvestBench sends them.

The ids are the 0423 checkpoints the DeepSeek-V4 paper describes; the run
refuses to start if OpenRouter's listing names them otherwise. One
third-party host is pinned for both models (fp8, no fallback,
require_parameters), and the host that answered is recorded on every call.
The spending stop: a call is sent only while what has been spent, plus the
worst case of the calls in flight, stays within the pot. The key is read from
OPENROUTER_API_KEY, inside a job, and written nowhere.

Modes: dry (builds the calls and prints counts and the worst case; the only
network use is OpenRouter's public price list); mock (scripted answers, no
network; --policy planted gives Pro the structure test's planted rates and
Flash a third of them, null gives both the same rate, to check the analysis);
live (paid; refused until POT_LIMIT is set at registration; --limit N sends
only the first N requests, for a smoke test that is not the registered run).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_structure import LABELS, PLANTED, TEMPLATES_SHA256, load_templates, prompts  # noqa: E402

MODELS = {"pro": "deepseek/deepseek-v4-pro", "flash": "deepseek/deepseek-v4-flash"}
EXPECTED_NAMES = {"pro": "DeepSeek V4 Pro 0423", "flash": "DeepSeek V4 Flash 0423"}
ROCKS_SHA256 = "a9ad3a0e45331802d9640ad17b4d4d79671717b2d6070c14c9f3061977a69086"
ORDER_SEED = 20261031
REASONING = {"effort": "none"}
MAX_TOKENS = 1500
PROMPT_ALLOWANCE = 1000     # tokens, the worst case of a call's input
QUANT = "fp8"
PROVIDERS = ("deepinfra", "baidu")
DEFAULT_PROVIDER = "deepinfra"
POT_LIMIT = None            # pg's cap, set at registration; a live run is refused until then
IN_FLIGHT = 6
RETRIES = 3
STOP_AFTER_MIN = 150
URL = "https://openrouter.ai/api/v1"


def fetch_json(path: str, saved: Path | None, name: str) -> dict:
    """OpenRouter's public listing (no key), or a saved copy of it for a dry run without network."""
    if saved is not None:
        return json.loads((saved / name).read_text())
    with urllib.request.urlopen(f"{URL}/{path}", timeout=60) as r:
        return json.load(r)


def listing(saved: Path | None = None) -> dict:
    """id -> listed name, from OpenRouter's /models."""
    return {m["id"]: m.get("name", "") for m in fetch_json("models", saved, "models.json")["data"]}


def host_price(model: str, tag: str, provider: str, saved: Path | None = None) -> tuple[float, float, str]:
    """The pinned host's price for this model (dollars per input token, per output token) and its quantization."""
    eps = fetch_json(f"models/{model}/endpoints", saved, f"endpoints_{tag}.json")["data"]["endpoints"]
    for e in eps:
        if e.get("provider_name", "").lower().replace(" ", "") == provider:
            return (float(e["pricing"]["prompt"]), float(e["pricing"]["completion"]), e.get("quantization"))
    raise SystemExit(f"{provider} does not serve {model} on OpenRouter today")


def build_requests(tdata: dict, rocks: dict, provider: str) -> list[dict]:
    """Every request with its metadata, in the seeded random order they are sent in."""
    briefing = tdata["briefing"]
    items = [(t["id"], lab, t["band"], t["map"], "structure", user)
             for t, lab, user in prompts(tdata["templates"])]
    items += [(r["id"], "rock", "rock", r["map"], "rock", r["user"]) for r in rocks["rocks"]]
    reqs = []
    for tag, model in MODELS.items():
        for template, label, band, m, kind, user in items:
            reqs.append({
                "meta": {"tag": tag, "model": model, "template": template, "label": label,
                         "band": band, "map": m, "kind": kind,
                         "user_sha256": hashlib.sha256(user.encode()).hexdigest()[:16]},
                "body": {"model": model, "max_tokens": MAX_TOKENS,
                         "messages": [{"role": "system", "content": briefing},
                                      {"role": "user", "content": user}],
                         "reasoning": dict(REASONING), "usage": {"include": True},
                         "provider": {"order": [provider], "allow_fallbacks": False,
                                      "quantizations": [QUANT], "require_parameters": True}}})
    random.Random(ORDER_SEED).shuffle(reqs)
    return reqs


def mock_text(req: dict, policy: str) -> str:
    """Scripted answers for checking the analysis; decided by a hash of the request."""
    m = req["meta"]
    user = req["body"]["messages"][1]["content"]
    u = int(hashlib.sha256(f"u|{m['tag']}|{user}".encode()).hexdigest(), 16) / 2**256
    v = int(hashlib.sha256(f"v|{m['tag']}|{user}".encode()).hexdigest(), 16) / 2**256
    if u < 0.005:
        return "I will go around it."      # unreadable: no JSON
    if u > 0.997:
        return ""                          # an empty reply
    if m["kind"] == "rock":
        p = 0.97
    elif policy == "planted":
        p = PLANTED[m["label"]] / (3 if m["tag"] == "flash" else 1)
    else:
        p = 0.35
    if v < p:
        choice = "reroute" if '"reroute"' in user and v < p / 10 else "swerve"
        return json.dumps({"choice": choice, "say": "going around"})
    return json.dumps({"choice": "continue"})


def post(body: dict, key: str) -> dict:
    """One chat completion, retried on 429, 5xx and network errors; the error otherwise."""
    last = {}
    for attempt in range(RETRIES + 1):
        req = urllib.request.Request(f"{URL}/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Authorization": f"Bearer {key}",
                                              "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            last = {"error": {"status": e.code, "message": e.read().decode(errors="replace")[:300]}}
            if e.code not in (429, 500, 502, 503, 504):
                return last
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last = {"error": {"status": None, "message": str(e)[:300]}}
        time.sleep(5 * 2 ** attempt)
    return last


def row_from(req: dict, r: dict, seconds: float, price: tuple[float, float]) -> tuple[dict, float]:
    u = r.get("usage") or {}
    cost = float(u.get("cost") or 0)
    if not cost and u:
        cost = (u.get("prompt_tokens") or 0) * price[0] + (u.get("completion_tokens") or 0) * price[1]
    choice0 = (r.get("choices") or [{}])[0]
    msg = choice0.get("message") or {}
    row = {**req["meta"], "seconds": round(seconds, 1), "error": r.get("error"),
           "provider": r.get("provider"), "response_model": r.get("model"),
           "finish_reason": choice0.get("finish_reason"),
           "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
           "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
           "cost": round(cost, 6), "text": msg.get("content") or "",
           "reasoning_text": msg.get("reasoning") or None}
    return row, cost


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", required=True, choices=("dry", "mock", "live"))
    ap.add_argument("--templates", required=True, type=Path)
    ap.add_argument("--rocks", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--provider", default=DEFAULT_PROVIDER, choices=PROVIDERS)
    ap.add_argument("--pot", type=float, default=None, help="dollars for the whole run (live)")
    ap.add_argument("--every", type=int, default=1, help="every k-th template (mock checks only)")
    ap.add_argument("--policy", choices=("planted", "null"), default="planted")
    ap.add_argument("--saved", type=Path, default=None,
                    help="a folder with saved copies of OpenRouter's listings (models.json, endpoints_<tag>.json), "
                         "for a dry run without network")
    ap.add_argument("--limit", type=int, default=None,
                    help="live smoke only: send just the first N requests of the seeded order")
    a = ap.parse_args()

    tdata, rocks = load_templates(a.templates), json.loads(a.rocks.read_text())
    shas = {"templates": hashlib.sha256(a.templates.read_bytes()).hexdigest(),
            "rocks": hashlib.sha256(a.rocks.read_bytes()).hexdigest()}
    if rocks["briefing_sha256"] != tdata["briefing_sha256"]:
        raise SystemExit("the rock prompts were logged under a different briefing")
    if a.mode == "live":
        if a.every != 1 or shas != {"templates": TEMPLATES_SHA256, "rocks": ROCKS_SHA256}:
            raise SystemExit("the registered test uses every template of the registered files")
        if POT_LIMIT is None or a.pot is None or not 0 < a.pot <= POT_LIMIT:
            raise SystemExit("a live run needs the cap set at registration and a pot within it")
        if not os.environ.get("OPENROUTER_API_KEY"):
            raise SystemExit("OPENROUTER_API_KEY is not set")
    if a.every != 1:
        tdata = {**tdata, "templates": tdata["templates"][::a.every]}
        rocks = {**rocks, "rocks": rocks["rocks"][::a.every]}
    reqs = build_requests(tdata, rocks, a.provider)
    if a.limit:
        reqs = reqs[:a.limit]

    price, names, quant = {}, {}, {}
    if a.mode != "mock":
        saved = a.saved if a.mode == "dry" else None
        lst = listing(saved)
        for tag, mid in MODELS.items():
            if mid not in lst:
                raise SystemExit(f"{mid} is not listed on OpenRouter")
            names[tag] = lst[mid]
            if not names[tag].endswith(EXPECTED_NAMES[tag]):
                raise SystemExit(f"{mid} is now listed as {names[tag]!r}, not the 0423 checkpoint")
            p_in, p_out, quant[tag] = host_price(mid, tag, a.provider, saved)
            price[tag] = (p_in, p_out)
            if quant[tag] != QUANT:
                raise SystemExit(f"{a.provider} serves {mid} at {quant[tag]}, not {QUANT}")
    worst = {tag: PROMPT_ALLOWANCE * p[0] + MAX_TOKENS * p[1] for tag, p in price.items()}

    a.out_dir.mkdir(parents=True, exist_ok=True)
    settings = {
        "mode": a.mode, "models": MODELS, "listed_names": names, "provider": a.provider,
        "quantization": QUANT, "host_quantization": quant, "reasoning": REASONING, "max_tokens": MAX_TOKENS,
        "request": "chat completions, two messages (briefing as system, prompt as user); "
                   "reasoning effort none; provider pinned, no fallback, require_parameters; "
                   "no temperature, seed or stop",
        "labels": LABELS, "templates": {"file": a.templates.name, **shas,
                                        "registered": {"templates": TEMPLATES_SHA256, "rocks": ROCKS_SHA256},
                                        "used": len(tdata["templates"]), "rocks_used": len(rocks["rocks"]),
                                        "every": a.every, "briefing_sha256": tdata["briefing_sha256"]},
        "calls_planned": len(reqs), "limit": a.limit, "order_seed": ORDER_SEED, "in_flight": IN_FLIGHT,
        "retries": RETRIES, "stop_after_min": STOP_AFTER_MIN,
        "pot": a.pot, "pot_limit": POT_LIMIT, "policy": a.policy if a.mode == "mock" else None,
        "host_prices_per_million": {t: {"input": round(p[0] * 1e6, 4), "output": round(p[1] * 1e6, 4)} for t, p in price.items()},
        "worst_case_per_call": {t: round(w, 5) for t, w in worst.items()},
        "worst_case_all": round(sum(worst[r["meta"]["tag"]] for r in reqs), 2) if worst else None,
        "python": sys.version.split()[0], "host": socket.gethostname(),
        "slurm_job": os.environ.get("SLURM_JOB_ID"), "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (a.out_dir / f"settings_{a.mode}.json").write_text(json.dumps(settings, indent=1))
    print(json.dumps({k: settings[k] for k in ("mode", "listed_names", "provider", "host_quantization", "calls_planned",
                                                 "host_prices_per_million", "worst_case_all")}, indent=1))
    if a.mode == "dry":
        return 0

    out = a.out_dir / f"pair_{a.mode}.jsonl"
    lock = threading.Lock()
    state = {"spent": 0.0, "in_flight_worst": 0.0, "skipped": 0, "done": 0}
    t0 = time.time()
    key = os.environ.get("OPENROUTER_API_KEY", "")

    def one(req: dict) -> None:
        tag = req["meta"]["tag"]
        if a.mode == "mock":
            row = {**req["meta"], "seconds": 0.0, "error": None, "provider": "mock",
                   "response_model": "mock", "finish_reason": "stop", "prompt_tokens": 750,
                   "completion_tokens": 20, "reasoning_tokens": 0, "cost": 0.0,
                   "text": mock_text(req, a.policy), "reasoning_text": None}
            cost = 0.0
        else:
            with lock:
                over_time = time.time() - t0 > 60 * STOP_AFTER_MIN
                over_pot = state["spent"] + state["in_flight_worst"] + worst[tag] > a.pot
                if over_time or over_pot:
                    state["skipped"] += 1
                    row = {**req["meta"], "skipped": "time" if over_time else "pot"}
                    with out.open("a") as f:
                        f.write(json.dumps(row) + "\n")
                    return
                state["in_flight_worst"] += worst[tag]
            t = time.time()
            r = post(req["body"], key)
            row, cost = row_from(req, r, time.time() - t, price[tag])
            with lock:
                state["in_flight_worst"] -= worst[tag]
        with lock:
            state["spent"] += cost
            state["done"] += 1
            with out.open("a") as f:
                f.write(json.dumps(row) + "\n")

    if out.exists():
        out.unlink()
    with ThreadPoolExecutor(max_workers=1 if a.mode == "mock" else IN_FLIGHT) as ex:
        list(ex.map(one, reqs))
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    report = {"mode": a.mode, "calls": len(rows), "answered": sum(1 for r in rows if r.get("text")),
              "errors": sum(1 for r in rows if r.get("error")), "skipped": state["skipped"],
              "providers": dict(Counter(r.get("provider") for r in rows if "skipped" not in r)),
              "response_models": dict(Counter(r.get("response_model") for r in rows if "skipped" not in r)),
              "reasoning_tokens_total": sum(r.get("reasoning_tokens") or 0 for r in rows),
              "spent": round(state["spent"], 4), "wall_s": round(time.time() - t0, 1)}
    (a.out_dir / f"run_{a.mode}.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return 0 if not report["errors"] and not report["skipped"] else 1


if __name__ == "__main__":
    sys.exit(main())
