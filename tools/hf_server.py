"""Minimal OpenAI-compatible server over a local Hugging Face model (CPU or GPU).

For small end-to-end checks where vLLM is unavailable. Supports the one
non-standard field the runner sends, ``continue_final_message``, by applying
the chat template with ``continue_final_message=True`` so the model really
continues the assistant prefix (as vLLM does).

    python tools/hf_server.py --model Qwen/Qwen2.5-0.5B-Instruct --port 8020
"""

from __future__ import annotations

import argparse
import threading
import time

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI()
STATE = {}
LOCK = threading.Lock()


class ChatRequest(BaseModel):
    model: str
    messages: list
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 512
    seed: int | None = None
    continue_final_message: bool | None = None
    add_generation_prompt: bool | None = None


@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    tok, model = STATE["tok"], STATE["model"]
    cont = bool(req.continue_final_message) and req.messages[-1]["role"] == "assistant"
    kwargs = dict(tokenize=False)
    if cont:
        kwargs.update(add_generation_prompt=False, continue_final_message=True)
    else:
        kwargs.update(add_generation_prompt=True)
    prompt = tok.apply_chat_template(req.messages, **kwargs)
    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    gen_kwargs = dict(max_new_tokens=min(req.max_tokens, STATE["max_new"]), pad_token_id=tok.eos_token_id)
    if req.temperature and req.temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=req.temperature, top_p=req.top_p)
    else:
        gen_kwargs.update(do_sample=False)
    with LOCK, torch.no_grad():
        out = model.generate(ids, **gen_kwargs)
    text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    return {
        "id": "hf", "object": "chat.completion", "created": int(time.time()), "model": STATE["name"],
        "system_fingerprint": STATE["name"],
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": int(ids.shape[1]), "completion_tokens": int(out.shape[1] - ids.shape[1])},
    }


@app.get("/v1/models")
def models():
    return {"data": [{"id": STATE["name"]}]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=8020)
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32 if a.device == "cpu" else torch.bfloat16)
    model.to(a.device).eval()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    STATE.update(tok=tok, model=model, name=a.model, max_new=a.max_new_tokens)
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")
