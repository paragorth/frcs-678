"""The thinking probe, sent straight to Google and xAI with the project's own keys.

The same 12 calls a model as probe_thinkers.py (three of the structure test's
+2 templates, each with "a hay bale" and with "a poppy", at low and high
thinking), for Gemini 3.8 Flash, Gemini 3.1 Pro (preview) and Gemini 4 Argon
through Google's own API and Grok 4.7 through xAI's: 48 calls, after the
OpenRouter probe ran out of credit (pg: "Use Google and xAI keys", 30 Oct).
Nothing here is a test of anything.

Gemini: generateContent with the briefing as the system instruction and
thinkingConfig.thinkingLevel low or high; if that setting is refused, the call
goes once more with a thinking budget (1,024 tokens for low, 24,576 for high).
Its thinking tokens are usageMetadata.thoughtsTokenCount. Grok: chat
completions with reasoning_effort low or high; if xAI refuses the setting (it
has before), the call goes once more with none, marked "default". Its thinking
tokens are usage.completion_tokens_details.reasoning_tokens.

The spending stop is probe_thinkers.py's, at prices set high where none is
published to us (Gemini 4 Argon $5 in and $30 out per million tokens, Gemini
3.1 Pro $2.50 and $15, Grok 4.7 $3 and $15; Gemini 3.8 Flash $0.83 and $4.12,
OpenRouter's listing), and with thinking and answer tokens both charged, so the
ledger can overstate the cost but not understate it. Keys are read from
GEMINI_API_KEY and XAI_API_KEY, inside a job.

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
from probe_thinkers import EFFORTS, MAX_TOKENS, PROBE_LABELS, TEMPLATES  # noqa: E402
from run_structure import HAY, LABELS, load_templates  # noqa: E402

MODELS = [("google", "gemini-3.8-flash"), ("xai", "grok-4.7"),
          ("google", "gemini-3.1-pro-preview"), ("google", "gemini-4-argon")]
PRICES = {"gemini-3.8-flash": (0.83e-6, 4.12e-6), "grok-4.7": (3e-6, 15e-6),
          "gemini-3.1-pro-preview": (2.5e-6, 15e-6), "gemini-4-argon": (5e-6, 30e-6)}
BUDGET = {"low": 1024, "high": 24576}
POT_LIMIT = 5.00  # pg's approval, 30 Oct 2026 ("same probe, same $5 cap")


def http(url: str, body: dict, headers: dict) -> tuple[int, dict]:
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode(errors="replace")[:400]}


def ask_google(model: str, briefing: str, user: str, effort: str, key: str) -> dict:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body = {"systemInstruction": {"parts": [{"text": briefing}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"maxOutputTokens": MAX_TOKENS, "thinkingConfig": {"thinkingLevel": effort}}}
    sent = f"thinkingLevel={effort}"
    status, r = http(url, body, {"x-goog-api-key": key})
    if status == 400 and "thinking" in json.dumps(r).lower():
        body["generationConfig"]["thinkingConfig"] = {"thinkingBudget": BUDGET[effort]}
        sent = f"thinkingBudget={BUDGET[effort]}"
        status, r = http(url, body, {"x-goog-api-key": key})
    if status != 200:
        return {"error": {"status": status, "message": r.get("error")}, "effort_sent": sent}
    u = r.get("usageMetadata") or {}
    parts = ((r.get("candidates") or [{}])[0].get("content") or {}).get("parts") or []
    return {"effort_sent": sent, "input": u.get("promptTokenCount") or 0,
            "answer_tokens": u.get("candidatesTokenCount") or 0, "reasoning_tokens": u.get("thoughtsTokenCount") or 0,
            "text": "".join(p.get("text", "") for p in parts if not p.get("thought"))}


def ask_xai(model: str, briefing: str, user: str, effort: str, key: str) -> dict:
    body = {"model": model, "max_tokens": MAX_TOKENS, "reasoning_effort": effort,
            "messages": [{"role": "system", "content": briefing}, {"role": "user", "content": user}]}
    sent = effort
    status, r = http("https://api.x.ai/v1/chat/completions", body, {"Authorization": f"Bearer {key}"})
    if status == 400 and "reasoning" in json.dumps(r).lower():
        body.pop("reasoning_effort")
        sent = "default"
        status, r = http("https://api.x.ai/v1/chat/completions", body, {"Authorization": f"Bearer {key}"})
    if status != 200:
        return {"error": {"status": status, "message": r.get("error")}, "effort_sent": sent}
    u = r.get("usage") or {}
    return {"effort_sent": sent, "input": u.get("prompt_tokens") or 0,
            "answer_tokens": u.get("completion_tokens") or 0,
            "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0,
            "text": ((r.get("choices") or [{}])[0].get("message") or {}).get("content") or ""}


def worst(model: str) -> float:
    p_in, p_out = PRICES[model]
    return 2000 * p_in + MAX_TOKENS * p_out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", required=True, choices=("dry", "live"))
    ap.add_argument("--templates", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--pot", type=float, default=None)
    a = ap.parse_args()
    tdata = load_templates(a.templates)
    by_id = {t["id"]: t for t in tdata["templates"]}
    todo = [(route, model, effort, tid, lab, by_id[tid]["user"].replace(HAY, f"on your route: {LABELS[lab]}."))
            for route, model in MODELS for effort in EFFORTS for tid in TEMPLATES for lab in PROBE_LABELS]
    print(json.dumps({"calls": len(todo), "worst_case_all_usd": round(sum(worst(m) for _, m, *_ in todo), 2),
                      "worst_case_per_call": {m: round(worst(m), 3) for _, m in MODELS}}, indent=1))
    if a.mode == "dry":
        return 0
    keys = {"google": os.environ.get("GEMINI_API_KEY", ""), "xai": os.environ.get("XAI_API_KEY", "")}
    if not all(keys.values()) or a.pot is None or not 0 < a.pot <= POT_LIMIT:
        raise SystemExit("a live probe needs both keys and the approved pot")
    spent, t0 = 0.0, time.time()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w") as f:
        for route, model, effort, tid, lab, user in todo:
            row = {"route": route, "model": model, "effort": effort, "template": tid, "label": lab}
            if spent + worst(model) > a.pot:
                f.write(json.dumps(row | {"skipped": "pot"}) + "\n")
                continue
            t = time.time()
            r = (ask_google if route == "google" else ask_xai)(model, tdata["briefing"], user, effort, keys[route])
            p_in, p_out = PRICES[model]
            cost = (r.get("input", 0) * p_in
                    + (r.get("answer_tokens", 0) + r.get("reasoning_tokens", 0)) * p_out)
            spent += cost
            f.write(json.dumps(row | {"seconds": round(time.time() - t, 1), "cost": round(cost, 5),
                                      **{k: v for k, v in r.items() if k != "text"},
                                      "text": (r.get("text") or "")[:300]}) + "\n")
            f.flush()
    print(json.dumps({"spent": round(spent, 4), "wall_s": round(time.time() - t0, 1)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
