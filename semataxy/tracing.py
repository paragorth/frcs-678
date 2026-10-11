"""Append-only trace logger and a model wrapper that records parse failures.

Replaces upstream's WandbLogger (which hard-wires wandb.init) with a logger of
the same interface that writes JSONL files into the run directory:

* llm_calls.jsonl  one record per gen/find/select, with agent, phase, prompt,
                   response, token usage, latency, cache hit/miss and whether
                   the parser fell back to the default value;
* game.jsonl       one record per environment step (resource in pool, catches).
"""

from __future__ import annotations

import json
import os
import time
import warnings
from datetime import datetime

from simulation.utils import ModelWandbWrapper


class _Span:
    def __init__(self, name: str):
        self.name = name
        self.children = []


class TraceLogger:
    """Duck-types upstream WandbLogger; see simulation/utils/logger.py."""

    def __init__(self, run_dir: str, run_name: str):
        os.makedirs(run_dir, exist_ok=True)
        self.run_dir = run_dir
        self.run_name = run_name
        self.run_id = run_name
        self._calls = open(os.path.join(run_dir, "llm_calls.jsonl"), "a")
        self._game = open(os.path.join(run_dir, "game.jsonl"), "a")
        self.current_agent_name = None
        self.current_phase_name = None
        self.current_query_name = None
        self.token_usage = 0
        self.token_usage_in = 0
        self.token_usage_out = 0
        self.n_calls = 0
        self.n_parse_fallbacks = 0
        self.is_finish_pending = False
        self.step = 0
        self.last_record = None
        # Set by the runner so each call is stamped with where in the game it happened.
        self.game_context = {}

    # -- interface used by ModelWandbWrapper ---------------------------------
    def get_agent_chain(self, agent_name, phase_name):
        self.current_agent_name = agent_name
        self.current_phase_name = phase_name
        return _Span(agent_name)

    def start_chain(self, chain_name):
        self.is_finish_pending = True
        self.current_query_name = chain_name
        return _Span(chain_name)

    def log_trace_llm(
        self, chain, name, default_value, start_time_ms, end_time_ms, system_message,
        prompt, status, status_message, response_text, temperature, top_p,
        token_usage_in, token_usage_out, model_name, **extra,
    ):
        rec = {
            "i": self.n_calls,
            "ts": datetime.utcnow().isoformat(),
            "agent": self.current_agent_name,
            "phase": self.current_phase_name,
            "query": self.current_query_name,
            "var": name,
            "model": model_name,
            "prompt": prompt,
            "response": response_text,
            "default_value": default_value,
            "parser_status": "ok",
            "temperature": temperature,
            "top_p": top_p,
            "token_in": token_usage_in,
            "token_out": token_usage_out,
            "latency_s": (end_time_ms - start_time_ms) / 1000.0,
            **self.game_context,
            **extra,
        }
        self.n_calls += 1
        self.token_usage_in += token_usage_in
        self.token_usage_out += token_usage_out
        self.token_usage += token_usage_in + token_usage_out
        # Buffered: the wrapper annotates it (parser status, cache) and then flushes.
        self.flush_last()
        self.last_record = rec

    def mark_last_as_fallback(self, message: str):
        """Called by the wrapper when gen/find/select returned its default."""
        if self.last_record is None:
            return
        self.last_record["parser_status"] = "default"
        self.last_record["parser_message"] = message[:500]
        self.n_parse_fallbacks += 1

    def annotate_last(self, **fields):
        if self.last_record is not None:
            self.last_record.update(fields)

    def flush_last(self):
        if self.last_record is not None:
            self._calls.write(json.dumps(self.last_record, ensure_ascii=False) + "\n")
            self._calls.flush()
            self.last_record = None

    def end_chain(self, agent_name, chain_span, html_render):
        self.is_finish_pending = False

    def save(self, base_path, agent_name_to_id):
        pass

    def log_game(self, kwargs, last_log=False):
        rec = {"step": self.step, "ts": time.time(), **self.game_context, **kwargs}
        if last_log:
            rec["last"] = True
        self._game.write(json.dumps(rec) + "\n")
        self._game.flush()
        self.step += 1

    def close(self):
        self.flush_last()
        self._calls.close()
        self._game.close()


class TracedModelWrapper(ModelWandbWrapper):
    """Upstream wrapper plus: parse-fallback detection and cache/latency stamps.

    Upstream catches every exception inside gen/find/select, warns, and returns
    the default value. For the analysis we need to know when that happened
    (an invalid action is not a policy choice), so we catch the warning.
    """

    def __init__(self, base_lm, logger: TraceLogger, temperature, top_p, seed):
        super().__init__(
            base_lm, render=False, wanbd_logger=logger, temperature=temperature,
            top_p=top_p, seed=seed, is_api=True,
        )
        self.trace_logger = logger

    def _traced(self, fn, previous_lm, *args, **kwargs):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            lm = fn(previous_lm, *args, **kwargs)
        fallback = [str(w.message) for w in caught if "Returning default value" in str(w.message)]
        if fallback:
            self.trace_logger.mark_last_as_fallback(fallback[0])
        last_call = getattr(lm, "last_call", None)
        if last_call:
            self.trace_logger.annotate_last(cache=last_call.get("cache"), cache_key=last_call.get("key"))
        self.trace_logger.flush_last()
        return lm

    def gen(self, previous_lm, *args, **kwargs):
        return self._traced(super().gen, previous_lm, *args, **kwargs)

    def find(self, previous_lm, *args, **kwargs):
        return self._traced(super().find, previous_lm, *args, **kwargs)

    def select(self, previous_lm, *args, **kwargs):
        return self._traced(super().select, previous_lm, *args, **kwargs)
