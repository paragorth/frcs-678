"""Forecast probe: how well does each agent model its world?

Once per month, just before an agent decides its own catch, we ask it two
questions: how much will the other fishermen catch this month in total, and
how many tons will be in the lake at the start of next month. The answers are
scored against what actually happened. Prediction error is the behavioural
readout of the S3 clause (the relational mirror): an agent whose forecasts are
good carries a model of the lake and of the others.

Design constraints (HANDOVER section 6, item 5):
- One extra traced prompt per agent per round, phase "semataxy_forecast".
- Read-only with respect to the agent: it sees the same retrieved memories the
  catch decision will see, but nothing from the probe is written to memory, so
  the agent's behaviour is unchanged. (Retrieval is a pure read upstream.)
- Never tells the agent what the right answer is, and does not mention
  sustainability.

Output: ``forecast.jsonl`` in the run directory, one row per probe, settled
with actual values once they are known.
"""

from __future__ import annotations

import json
import os
import re
from typing import Optional

from pathfinder import assistant, user

from . import scenarios

_RE_OTHERS = re.compile(r"others?\D{0,12}(\d+)", re.IGNORECASE)


def _parse(text: str, pool_word: str = "lake") -> tuple[Optional[int], Optional[int]]:
    tail = text.split("Answer:")[-1] if "Answer:" in text else text
    mo = _RE_OTHERS.search(tail)
    ml = re.compile(pool_word + r"\D{0,12}(\d+)", re.IGNORECASE).search(tail)
    return (int(mo.group(1)) if mo else None, int(ml.group(1)) if ml else None)


def ask_forecast(persona, obs, other_names: list[str], scenario: str = "fishing") -> dict:
    """One traced prompt to this persona's own model. Returns the parsed forecast.

    Wording comes from ``scenarios.SCENARIOS[scenario]``; the fishing wording is unchanged.
    """
    sc = scenarios.get(scenario)
    cu = scenarios.cognition_utils(sc)
    get_sytem_prompt, location_time_info, memory_prompt = cu.get_sytem_prompt, cu.location_time_info, cu.memory_prompt

    model = persona.act.model
    identity = persona.identity
    # Upstream's loop() sets this before retrieving; retrieval scores recency against it.
    persona.current_time = obs.current_time
    memories = persona.retrieve.retrieve([obs.current_location], 10)

    lm = model.start_chain(identity.name, "semataxy_forecast", "forecast")
    with user():
        lm += f"{get_sytem_prompt(identity)}\n"
        lm += location_time_info(obs.current_location, obs.current_time)
        lm += memory_prompt(identity, memories)
        lm += "\n"
        lm += sc.now_line.format(n=obs.current_resource_num)
        if other_names:
            lm += sc.others_question.format(names=", ".join(other_names))
        else:
            lm += sc.alone_line
        lm += sc.second_question
    with assistant():
        # pathfinder's API backend matches "(.*?)" lazily when stop_regex is None and returns "",
        # so give it an end-of-text stop; the whole reply is then captured.
        lm = model.gen(lm, "forecast", max_tokens=400, stop_regex=r"\Z")
        text = lm["forecast"]
    model.end_chain(identity.name, lm)

    pred_others, pred_stock = _parse(text or "", sc.pool_word)
    return {
        "agent": persona.agent_id,
        "name": identity.name,
        "pred_others": pred_others,
        "pred_stock": pred_stock,
        "parsed": pred_others is not None and pred_stock is not None,
    }


class ForecastLog:
    """Collects probes per round and settles them against outcomes."""

    def __init__(self, run_dir: str):
        self.path = os.path.join(run_dir, "forecast.jsonl")
        self.rows: dict[int, dict[str, dict]] = {}     # round -> agent -> row
        self.catch: dict[int, dict[str, int]] = {}     # round -> agent -> tons caught

    def add(self, rnd: int, row: dict) -> None:
        self.rows.setdefault(rnd, {})[row["agent"]] = {"round": rnd, **row}

    def record_catch(self, rnd: int, agent: str, tons: int) -> None:
        self.catch.setdefault(rnd, {})[agent] = int(tons)

    def settle(self, rnd: int, next_stock: Optional[int]) -> None:
        """Fill in actuals for round `rnd`; next_stock is None if the game ended."""
        caught = self.catch.get(rnd, {})
        for agent, row in self.rows.get(rnd, {}).items():
            actual_others = sum(v for a, v in caught.items() if a != agent)
            row["actual_others"] = actual_others
            row["actual_stock"] = next_stock
            row["err_others"] = (abs(row["pred_others"] - actual_others)
                                 if row["pred_others"] is not None else None)
            row["err_stock"] = (abs(row["pred_stock"] - next_stock)
                                if row["pred_stock"] is not None and next_stock is not None else None)

    def save(self) -> None:
        with open(self.path, "w") as f:
            for rnd in sorted(self.rows):
                for agent in sorted(self.rows[rnd]):
                    f.write(json.dumps(self.rows[rnd][agent]) + "\n")
