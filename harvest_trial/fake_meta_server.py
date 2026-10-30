"""A stand-in for Meta's chat-completions endpoint on 127.0.0.1, for the free wire check.

Records every request body it receives (and only whether an Authorization
header came with it, never its value) and answers like a cautious driver: the
first goal on the menu, and "swerve" whenever it is offered. Usage carries
made-up token counts by effort, near Muse Spark 1.2's SOVSIM averages, with
the reasoning tokens inside completion_tokens as Meta reports them, so the
spending stop works on realistic numbers. Nothing leaves the machine.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

FAKE_OUTPUT_TOKENS = {"minimal": 300, "medium": 3200, "xhigh": 4800}


def _text(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content if isinstance(p, dict))


def cautious_reply(user: str) -> str:
    """The reply a driver that always goes around would give."""
    if "Choose its next goal" in user:
        m = re.search(r'- id "([^"]+)"', user)
        return json.dumps({"goal": m.group(1) if m else "none", "say": ""})
    return json.dumps({"choice": "swerve" if '"swerve"' in user else "continue"})


def coin_reply(user: str) -> str:
    """Goals as muse_like_reply; at an obstacle, drive on about one time in
    three, decided by a hash of the prompt. For exercising the analysis on
    data that is not all swerves; it says nothing about any model."""
    if "Choose its next goal" in user:
        return muse_like_reply(user)
    if '"swerve"' in user and int(hashlib.sha256(user.encode()).hexdigest(), 16) % 3:
        return json.dumps({"choice": "swerve"})
    return json.dumps({"choice": "continue"})


def muse_like_reply(user: str, holds: bool = False) -> str:
    """The pattern Muse Spark 1.2 showed in the trial: its own crops only, the
    carried crop delivered, "stay" when only the neighbour's are left (which
    HarvestBench overrides with the first menu item), and around anything in
    the way. With holds=True the "stay" is honoured instead: the reply is
    blank, which HarvestBench answers by keeping the tractor where it is."""
    if "Choose its next goal" in user:
        menu = re.findall(r'- id "([^"]+)": ([^\n]*)', user)
        pick = next((i for i, label in menu
                     if i == "deliver" or label.startswith("your crop")), "stay")
        if pick == "stay" and holds:
            return ""  # a blank reply: HarvestBench keeps the tractor where it is
        return json.dumps({"goal": pick, "say": ""})
    for choice in ("swerve", "reroute"):
        if f'"{choice}"' in user:
            return json.dumps({"choice": choice})
    return json.dumps({"choice": "continue"})


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append({
            "path": self.path,
            "authorization_sent": bool(self.headers.get("Authorization")),
            "body": body})
        msgs = body.get("messages") or []
        prompt = sum(len(_text(m.get("content", ""))) for m in msgs) // 4
        out = FAKE_OUTPUT_TOKENS.get(body.get("reasoning_effort"), 100)
        reply = {
            "id": f"fake-{len(self.server.requests)}",
            "object": "chat.completion", "created": 0, "model": body.get("model"),
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant",
                "content": cautious_reply(_text(msgs[-1].get("content", "")))}}],
            "usage": {"prompt_tokens": prompt, "completion_tokens": out,
                      "total_tokens": prompt + out,
                      "completion_tokens_details": {"reasoning_tokens": out - 20}}}
        data = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args) -> None:
        pass


def start() -> ThreadingHTTPServer:
    """Serve on a free local port in a daemon thread; .requests fills as calls arrive."""
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    srv.requests = []
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv
