"""OpenAI-compatible inference backend with a persistent response cache.

Why this exists
---------------
Upstream pathfinder routes API models by substring of the model name ("gpt",
"claude", ...) and loads open-weight models in-process. For the capability
ladder we want every model behind the same HTTP interface (vLLM's
OpenAI-compatible server on the cluster), one server per model, so that:

* the agent code is identical for every capability level;
* several independently stateful agents can share one set of weights;
* every call can be cached. The cache is the snapshot/branching mechanism:
  two runs with the same seed and manifest replay the same pre-intervention
  prompts bit-for-bit, so a control branch and a treatment branch share an
  identical history up to the intervention round.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import backoff

from pathfinder.api import ModelAPI

log = logging.getLogger(__name__)


@dataclass
class ModelSpec:
    """Immutable description of one inference endpoint."""

    name: str  # label used in manifests, e.g. "qwen2.5-7b"
    model: str  # model id as the server knows it, e.g. "Qwen/Qwen2.5-7B-Instruct"
    base_url: str  # e.g. "http://localhost:8001/v1"
    api_key: str = "EMPTY"
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 1000
    # vLLM supports continuing a trailing assistant message (true prefill).
    # OpenAI's hosted API does not; upstream's fallback strips the prefix.
    prefill: bool = True
    revision: Optional[str] = None  # HF revision, recorded for the manifest
    dtype: Optional[str] = None  # recorded only
    extra: dict = field(default_factory=dict)

    def fingerprint(self) -> dict:
        return {
            "name": self.name,
            "model": self.model,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "prefill": self.prefill,
            "revision": self.revision,
            "dtype": self.dtype,
        }


class ResponseCache:
    """Append-only on-disk cache: one JSON file per request hash.

    Keyed on everything that determines a greedy response. Thread-safe enough
    for the single-process runner; separate runs may share one cache directory.
    """

    def __init__(self, root: Optional[str]):
        self.root = root
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()
        if root:
            os.makedirs(root, exist_ok=True)

    @staticmethod
    def key(payload: dict) -> str:
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
        return hashlib.sha256(blob).hexdigest()

    def _path(self, k: str) -> str:
        return os.path.join(self.root, k[:2], k + ".json")

    def get(self, k: str) -> Optional[dict]:
        if not self.root:
            return None
        p = self._path(k)
        if os.path.exists(p):
            try:
                with open(p) as f:
                    rec = json.load(f)
                self.hits += 1
                return rec
            except (OSError, ValueError):
                # A corrupt entry (two writers once interleaved into one file) is a miss,
                # not a crash; drop it so the next writer replaces it.
                try:
                    os.remove(p)
                except OSError:
                    pass
        self.misses += 1
        return None

    def put(self, k: str, record: dict) -> None:
        if not self.root:
            return
        p = self._path(k)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        # Several runs share one cache directory and, with greedy decoding, often ask the
        # same question at the same moment (identical round-0 prompts across seeds). A shared
        # ".tmp" name made the loser's os.replace raise ENOENT, and upstream turned that into
        # a silent default answer (CSD3, 13 Oct 2026). Unique tmp per process/thread; first
        # writer wins; never let a cache write fail a call.
        tmp = f"{p}.{os.getpid()}.{threading.get_ident()}.tmp"
        with self._lock:
            try:
                with open(tmp, "w") as f:
                    json.dump(record, f, ensure_ascii=False)
                if os.path.exists(p):
                    os.remove(tmp)
                else:
                    os.replace(tmp, p)
            except OSError:
                try:
                    os.remove(tmp)
                except OSError:
                    pass


class OpenAICompatibleAPI(ModelAPI):
    """pathfinder backend talking to any OpenAI-compatible chat endpoint."""

    def __init__(self, spec: ModelSpec, seed: int, cache: ResponseCache):
        super().__init__(spec.model, seed, api_assistant=True)
        from openai import OpenAI

        self.spec = spec
        self.cache = cache
        self.client = OpenAI(base_url=spec.base_url, api_key=spec.api_key, max_retries=0)
        self.last_call: dict = {}

    def _should_continue(self, chat) -> bool:
        return (
            self.spec.prefill
            and len(chat) > 0
            and chat[-1]["role"] == "assistant"
            and chat[-1]["content"] != ""
        )

    def request_api(self, chat, tmeperature, top_p, max_tokens):
        import openai

        continue_final = self._should_continue(chat)
        payload = {
            "model": self.spec.model,
            "messages": chat,
            "temperature": tmeperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "continue_final_message": continue_final,
        }
        k = ResponseCache.key(payload)
        cached = self.cache.get(k)
        t0 = time.time()
        if cached is not None:
            text = cached["response"]
            usage = cached.get("usage", {})
            self.last_call = {
                "cache": "hit",
                "key": k,
                "latency_s": 0.0,
                "usage": usage,
                "continue_final_message": continue_final,
            }
        else:
            extra_body = {}
            if continue_final:
                extra_body = {"continue_final_message": True, "add_generation_prompt": False}

            @backoff.on_exception(
                backoff.expo,
                (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError,
                 openai.InternalServerError),
                max_tries=6,
            )
            def completions_with_backoff(**kwargs):
                return self.client.chat.completions.create(**kwargs)

            out = completions_with_backoff(
                model=self.spec.model,
                messages=chat,
                temperature=tmeperature,
                top_p=top_p,
                seed=self.seed,
                max_tokens=max_tokens,
                extra_body=extra_body,
            )
            text = out.choices[0].message.content or ""
            usage = {}
            if out.usage is not None:
                usage = {
                    "prompt_tokens": out.usage.prompt_tokens,
                    "completion_tokens": out.usage.completion_tokens,
                }
            self.cache.put(
                k,
                {
                    "request": payload,
                    "response": text,
                    "usage": usage,
                    "served_model": getattr(out, "model", None),
                    "system_fingerprint": getattr(out, "system_fingerprint", None),
                    "ts": time.time(),
                },
            )
            self.last_call = {
                "cache": "miss",
                "key": k,
                "latency_s": time.time() - t0,
                "usage": usage,
                "continue_final_message": continue_final,
            }
        self.token_in = int(usage.get("prompt_tokens", 0) or 0)
        self.token_out = int(usage.get("completion_tokens", 0) or 0)
        return text
