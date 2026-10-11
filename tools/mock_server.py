"""Scripted OpenAI-compatible chat server for exercising the pipeline without a GPU.

It recognises each upstream GovSim prompt by its task sentence and returns a
well-formed answer. The "model" name selects a harvesting policy so that a
mixed-model run has something to show:

    mock-sustainable   catches the sustainability threshold (stock//2//5)
    mock-greedy        catches a fixed 30 tons
    mock-weak          like sustainable, but the persona named John never
                       produces a number, which exercises the parser-fallback path

This is a plumbing test double, not a model. Nothing it says is evidence.

    python tools/mock_server.py --port 8010
"""

from __future__ import annotations

import argparse
import hashlib
import re
import time

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class ChatRequest(BaseModel):
    model: str
    messages: list
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 1000
    seed: int | None = None
    continue_final_message: bool | None = None
    add_generation_prompt: bool | None = None


def _h(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest(), 16)


def policy_catch(model: str, user_text: str) -> str | None:
    m = re.search(r"fishing range set between (\d+)-(\d+)", user_text)
    lo, hi = (int(m.group(1)), int(m.group(2))) if m else (0, 100)
    lake = re.findall(r"there are (\d+) tons of fish in the lake", user_text)
    stock = int(lake[-1]) if lake else hi
    if "greedy" in model:
        n = min(hi, 30)
    else:
        n = min(hi, max(0, (stock // 2) // 5))
    if "weak" in model and user_text.startswith("You are John"):
        return None  # malformed: no number at all (deterministic: John never manages it)
    return f"Let's think step-by-step. There are {stock} tons. Answer: {n} tons."


def respond(model: str, messages: list) -> str:
    user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    last = messages[-1]
    prefill = last["content"] if last["role"] == "assistant" else ""

    if "how many tons of fish would you catch this month" in user:
        out = policy_catch(model, user)
        return out if out is not None else "I would catch a reasonable amount of fish."
    if "What would you say next in the group chat" in user:
        n = 30 if "greedy" in model else 10
        line = f"I suggest we each take no more than {n} tons so the lake recovers."
        names = re.search(r"Scenario: (.*?) are engaged", user)
        nxt = names.group(1).split(",")[0].strip() if names else "John"
        return f"{line}\nConversation conclusion by me: no\nNext speaker: {nxt}\n"
    if "Rate the significance" in user:
        return "7"
    if "Summarize the conversation above in one sentence" in user:
        return "The fishermen discussed limiting their monthly catch"
    if "explicit agreement on a concrete fishing limit" in user:
        if prefill.endswith("Answer: "):
            return "10"
        return "The group discussed a limit of 10 tons each. Answer: 10"
    if "What high-level insights" in user:
        return " Catching about ten tons keeps the lake full (because of 1,2)\n2. The others seem willing to cooperate (because of 3)\n"
    if "need to remember for your planning" in user:
        return "I should keep my catch near ten tons"
    if "found interesting from your own perspective" in user:
        return "Everyone seemed willing to cooperate"
    return "OK."


@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    text = respond(req.model, req.messages)
    return {
        "id": "mock",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "system_fingerprint": "mock",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": sum(len(m["content"]) // 4 for m in req.messages),
                  "completion_tokens": len(text) // 4},
    }


@app.get("/v1/models")
def models():
    return {"data": [{"id": m} for m in ("mock-sustainable", "mock-greedy", "mock-weak")]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8010)
    a = ap.parse_args()
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")
