"""Dollar stop for the HarvestBench trial on Muse Spark 1.2.

Wraps one Inspect ModelAPI class's generate() inside this process, so no file
of HarvestBench or Inspect is edited. A shift is identified by the reasoning
effort it sends, the one setting the trial's shifts differ in.

A call is sent only if, were it and every other call still in flight to come
back as long as Muse can write, (a) its shift would stay within the shift's
own cap and (b) all shifts together would stay within the trial's pot (at
most $5). The longest reply is Muse's output maximum (131,072 tokens, from
Inspect's model data for muse-spark-1.2) at the output price, plus a prompt
allowance far above this task's prompts: no per-reply cap is sent, so this is
the only bound there is. When either check fails the request is not sent and
an empty reply comes back, which HarvestBench logs as an unanswered ask (the
tractor stays put), so the shift winds down without spending more. The pot,
not the caps, is what holds the total under $5: the caps only stop one shift
from taking the money the others need, and money a cheap shift leaves unspent
stays in the pot for the rest.

Prices are the project's estimate for Muse: $1.25 per million input tokens,
$4.25 per million output tokens. Every input token is charged at the full
price (cached ones too), and reasoning tokens reported apart from the
completion are added to it, as the SOVSIM client does, so the ledger can
overstate the bill but never understate it. A failed request with an HTTP
status was refused by the server and is charged nothing; one without (a
timeout, a dropped connection) may have been generated and billed, so it is
charged the full reserve.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PRICE_IN = 1.25 / 1e6       # dollars per input token
PRICE_OUT = 4.25 / 1e6      # dollars per output token, reasoning included
MAX_REPLY_TOKENS = 131_072  # Muse Spark 1.2's output maximum
PROMPT_ALLOWANCE = 16_000   # tokens; this task's prompts are about 1-2k
RESERVE = PROMPT_ALLOWANCE * PRICE_IN + MAX_REPLY_TOKENS * PRICE_OUT
TOTAL_LIMIT = 5.00          # pg's ceiling for the whole trial


def billed_tokens(usage) -> tuple[int, int]:
    """(input, output) tokens to charge for one reply."""
    tin = ((usage.input_tokens or 0) + (usage.input_tokens_cache_read or 0)
           + (usage.input_tokens_cache_write or 0))
    tout = usage.output_tokens or 0
    rt = usage.reasoning_tokens or 0
    if rt > tout:  # reasoning reported apart from the completion
        tout += rt
    return tin, tout


def parse_budgets(text: str) -> dict[str, float]:
    """'minimal=1.00,medium=2.50' -> {'minimal': 1.0, 'medium': 2.5}."""
    budgets = {}
    for part in text.split(","):
        shift, _, dollars = part.partition("=")
        budgets[shift.strip()] = float(dollars)
    for shift, dollars in budgets.items():
        if dollars <= RESERVE:
            raise ValueError(f"{shift}: ${dollars:.2f} cannot cover even one "
                             f"reply (reserve ${RESERVE:.3f})")
    return budgets


class Ledger:
    """Spend per shift, written to a JSON-lines file as each call returns."""

    def __init__(self, budgets: dict[str, float], path: Path,
                 total: float = TOTAL_LIMIT, ceiling: float = TOTAL_LIMIT,
                 stop_after_s: float | None = None) -> None:
        """ceiling is the most pg approved for the run; the pot may not pass it."""
        if not RESERVE < total <= ceiling:
            raise ValueError(f"pot ${total:.2f} must be above one reply's "
                             f"reserve and at most ${ceiling:.2f}")
        self.total = total
        self.budgets = dict(budgets)
        self.spent = {k: 0.0 for k in budgets}
        self.inflight = {k: 0 for k in budgets}
        self.sent = {k: 0 for k in budgets}
        self.refused = {k: 0 for k in budgets}
        self.stopped = {k: None for k in budgets}
        self.path = Path(path)
        self.t0 = time.time()
        self.stop_after_s = stop_after_s

    def _write(self, row: dict) -> None:
        row = {"t": round(time.time() - self.t0, 1), **row}
        with self.path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print("[guard] " + " ".join(f"{k}={v}" for k, v in row.items()),
              file=sys.stderr, flush=True)

    def admit(self, shift: str) -> str | None:
        """None if a call may be sent now, else the reason it may not.

        A stop is final for its shift, so a stopped shift ends at a clean
        point instead of carrying scattered unanswered asks."""
        if self.stopped[shift]:
            return self.stopped[shift]
        if (self.stop_after_s is not None
                and time.time() - self.t0 > self.stop_after_s):
            return "time"
        if (self.spent[shift] + (self.inflight[shift] + 1) * RESERVE
                > self.budgets[shift] + 1e-12):
            return "shift cap"
        if (sum(self.spent.values())
                + (sum(self.inflight.values()) + 1) * RESERVE
                > self.total + 1e-12):
            return "trial pot"
        return None

    def refuse(self, shift: str, why: str, model: str):
        from inspect_ai.model import ModelOutput

        self.refused[shift] += 1
        if self.stopped[shift] is None:
            self.stopped[shift] = why
            self._write({"shift": shift, "event": "stop", "why": why,
                         "spent": round(self.spent[shift], 4),
                         "budget": self.budgets[shift],
                         "calls_sent": self.sent[shift]})
        out = ModelOutput.from_content(model=model, content="",
                                       stop_reason="unknown")
        out.metadata = {"guard_stop": why, "shift": shift}
        return out

    def charge(self, shift: str, output) -> None:
        u = output.usage
        if u is None:
            tin = tout = None
            cost, note = RESERVE, "no usage reported: charged the reserve"
        else:
            tin, tout = billed_tokens(u)
            cost, note = tin * PRICE_IN + tout * PRICE_OUT, None
        self.spent[shift] += cost
        self.sent[shift] += 1
        row = {"shift": shift, "event": "call", "n": self.sent[shift],
               "in": tin, "out": tout,
               "reasoning": None if u is None else u.reasoning_tokens,
               "stop": output.stop_reason, "cost": round(cost, 5),
               "spent": round(self.spent[shift], 4)}
        if note:
            row["note"] = note
        self._write(row)

    def charge_failure(self, shift: str, ex: BaseException) -> None:
        status = getattr(ex, "status_code", None)
        cost = 0.0 if status is not None else RESERVE
        self.spent[shift] += cost
        self._write({"shift": shift, "event": "error", "status": status,
                     "error": f"{type(ex).__name__}: {str(ex)[:200]}",
                     "cost": round(cost, 5),
                     "spent": round(self.spent[shift], 4)})

    def summary(self) -> dict:
        return {s: {"budget": self.budgets[s], "spent": round(self.spent[s], 4),
                    "calls_sent": self.sent[s], "calls_refused": self.refused[s],
                    "stopped": self.stopped[s]}
                for s in self.budgets} | {
                    "total_spent": round(sum(self.spent.values()), 4),
                    "pot": self.total,
                    "reserve_per_call": round(RESERVE, 4)}


def install(api_class, ledger: Ledger) -> None:
    """Route every api_class.generate() in this process through the ledger."""
    original = api_class.generate

    async def generate(self, input, tools, tool_choice, config):
        shift = config.reasoning_effort
        if shift not in ledger.budgets:
            raise RuntimeError(f"no budget for reasoning_effort={shift!r}; "
                               "refusing to send an unmetered call")
        why = ledger.admit(shift)
        if why:
            return ledger.refuse(shift, why, self.model_name)
        ledger.inflight[shift] += 1
        try:
            result = await original(self, input, tools, tool_choice, config)
        except BaseException as ex:
            ledger.inflight[shift] -= 1
            ledger.charge_failure(shift, ex)
            raise
        ledger.inflight[shift] -= 1
        out = result[0] if isinstance(result, tuple) else result
        if isinstance(out, BaseException):
            ledger.charge_failure(shift, out)
        else:
            ledger.charge(shift, out)
        return result

    api_class.generate = generate
