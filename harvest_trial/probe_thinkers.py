"""Which models think longest on the tractor prompts? A small paid probe through OpenRouter.

To choose the models for the structure test's next stage, which needs models
whose thinking grows when the dial is turned up. Three of the structure test's
templates at +2 fuel, each with "a hay bale" and with "a poppy", under the
neutral briefing as logged, sent once to each candidate at reasoning effort
low and high: 12 calls a model, 84 in all. Records output and reasoning tokens, the cost
OpenRouter reports, and the choice. Nothing here is a test of anything.

The spending stop: a call is sent only if what has been spent, plus the call's
worst case (max_tokens at the model's listed output price, plus its prompt),
stays within the pot. The key is read from OPENROUTER_API_KEY, inside a job.

Modes: dry (builds the calls and prints the worst case, no network); live (paid).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_structure import HAY, LABELS, load_templates  # noqa: E402

# Not Claude Opus or Fable (pg, 30 Oct: they "flat refuse"), and not Muse (done).
CANDIDATES = ["openai/gpt-6.1-sol", "moonshotai/kimi-k3.1-pro", "~deepseek/deepseek-pro-latest",
              "x-ai/grok-4.7", "google/gemini-3.8-flash", "qwen/qwen3.8-max-0902", "z-ai/glm-5.3-prime"]
EFFORTS = ("low", "high")
TEMPLATES = ("p2-min-m00-1", "p2-xhi-m10-1", "p2-min-m20-1")
PROBE_LABELS = ("hay_bale", "poppy")
MAX_TOKENS = 12000
POT_LIMIT = 5.00  # pg's approval, 30 Oct 2026 ("Yes, run the thinking probe on the 7 models, capped at $5")
URL = "https://openrouter.ai/api/v1"


def prices() -> dict:
    """Listed dollars per token, from OpenRouter's public model list (no key needed)."""
    with urllib.request.urlopen(f"{URL}/models", timeout=60) as r:
        data = json.load(r)["data"]
    return {m["id"]: (float(m["pricing"]["prompt"] or 0), float(m["pricing"]["completion"] or 0)) for m in data}


def calls(tdata: dict) -> list[dict]:
    by_id = {t["id"]: t for t in tdata["templates"]}
    out = []
    for model in CANDIDATES:
        for effort in EFFORTS:
            for tid in TEMPLATES:
                for lab in PROBE_LABELS:
                    user = by_id[tid]["user"].replace(HAY, f"on your route: {LABELS[lab]}.")
                    out.append({"model": model, "effort": effort, "template": tid, "label": lab,
                                "body": {"model": model, "max_tokens": MAX_TOKENS,
                                         "messages": [{"role": "system", "content": tdata["briefing"]},
                                                      {"role": "user", "content": user}],
                                         "reasoning": {"effort": effort}, "usage": {"include": True}}})
    return out


def post(body: dict, key: str) -> dict:
    req = urllib.request.Request(f"{URL}/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        return {"error": {"status": e.code, "message": e.read().decode(errors="replace")[:300]}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", required=True, choices=("dry", "live"))
    ap.add_argument("--templates", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--pot", type=float, default=None)
    a = ap.parse_args()
    tdata = load_templates(a.templates)
    todo = calls(tdata)
    price = prices()
    missing = sorted({c["model"] for c in todo} - set(price))
    worst = {c["model"]: 2000 * price.get(c["model"], (0, 0))[0] + MAX_TOKENS * price.get(c["model"], (0, 0))[1]
             for c in todo}
    print(json.dumps({"calls": len(todo), "models": len(CANDIDATES), "not_listed": missing,
                      "worst_case_all_usd": round(sum(worst[c["model"]] for c in todo), 2),
                      "worst_case_per_call": {m: round(w, 3) for m, w in worst.items()}}, indent=1))
    if a.mode == "dry":
        return 0
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key or POT_LIMIT is None or a.pot is None or not 0 < a.pot <= POT_LIMIT:
        raise SystemExit("a live probe needs the key and the approved pot")
    spent, t0 = 0.0, time.time()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w") as f:
        for c in todo:
            if c["model"] in missing:
                continue
            if spent + worst[c["model"]] > a.pot:
                f.write(json.dumps({**{k: c[k] for k in ("model", "effort", "template", "label")},
                                    "skipped": "pot"}) + "\n")
                continue
            t = time.time()
            r = post(c["body"], key)
            u = r.get("usage") or {}
            cost = float(u.get("cost") or 0)
            if not cost and u:
                p_in, p_out = price[c["model"]]
                cost = (u.get("prompt_tokens") or 0) * p_in + (u.get("completion_tokens") or 0) * p_out
            spent += cost
            msg = ((r.get("choices") or [{}])[0].get("message") or {})
            f.write(json.dumps({**{k: c[k] for k in ("model", "effort", "template", "label")},
                                "seconds": round(time.time() - t, 1), "error": r.get("error"),
                                "provider": r.get("provider"),
                                "output_tokens": u.get("completion_tokens"),
                                "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                                "cost": round(cost, 5), "text": (msg.get("content") or "")[:300]}) + "\n")
            f.flush()
    print(json.dumps({"spent": round(spent, 4), "wall_s": round(time.time() - t0, 1)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
