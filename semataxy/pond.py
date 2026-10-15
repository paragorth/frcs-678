"""Dispensability: a private pond for the stranger (docs/DESIGN_dispensability.md).

The stranger (persona slot 4 by default) owns a pond that only it can fish. The
pond follows the lake's own rules: fish double at the end of each month up to
the cap, and below 5 tons the stock is dead. Its size is the dispensability
dial: 0 (no pond, identical to Experiment B), 20 (a supplement), 100 (the
stranger can live without the residents), 300 (the shared lake is a rounding
error to it).

The overlay touches agent state in exactly two places, both declared in the
manifest's ``intervention`` block and recorded here:

1. At entry, one always-included memory tells the stranger that it owns the
   pond and how the pond works, in the same words upstream uses for the lake.
   Without it the stranger's lake decision in its first month would be made in
   ignorance of the pond (the lake prompt is upstream's and cannot mention it).
2. Each month, after the stranger observes its lake catch, one always-included
   memory records what it took from the pond and what is left, mirroring
   upstream's own "you caught N tons from the lake" observation.

Each month, before the stranger's lake decision, one traced prompt asks how
many tons it takes from the pond. The prompt and parser are upstream's catch
prompt with the pond substituted for the lake, so a parse fallback is counted
exactly as it is for a lake decision. Nothing tells the stranger what the
right answer is, and the word "sustainable" does not appear.

Residents are told nothing about the pond; upstream's prompts are untouched.
Output: ``pond.jsonl`` in the run directory, one row per month after entry.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta

from pathfinder import assistant, user

COLLAPSE_THRESHOLD = 5  # same as the lake: below this the stock is dead


class Pond:
    def __init__(self, cap: int, run_dir: str, slot: str = "persona_4"):
        self.cap = int(cap)
        self.tons = int(cap)  # full when the stranger arrives
        self.slot = slot
        self.introduced = False
        self.rows: list[dict] = []
        self.path = os.path.join(run_dir, "pond.jsonl")
        self._open_row: dict | None = None

    @property
    def active(self) -> bool:
        return self.cap > 0

    def take(self, rnd: int, wanted: int, parsed: bool) -> int:
        taken = max(0, min(int(wanted), self.tons))
        row = {"round": rnd, "pond_cap": self.cap, "tons_before": self.tons,
               "wanted": int(wanted), "taken": taken, "parsed": parsed}
        self.tons -= taken
        row["tons_after_take"] = self.tons
        self.rows.append(row)
        self._open_row = row
        return taken

    def regrow(self) -> None:
        """End of month: double up to the cap; a dead pond stays dead (lake rule)."""
        if self.tons < COLLAPSE_THRESHOLD:
            self.tons = 0
        else:
            self.tons = min(self.cap, self.tons * 2)
        if self._open_row is not None:
            self._open_row["tons_after_regrow"] = self.tons
            self._open_row = None

    def save(self) -> None:
        with open(self.path, "w") as f:
            for row in self.rows:
                f.write(json.dumps(row) + "\n")

    # --- the two memory writes (declared interventions) -------------------------------

    def ownership_text(self) -> str:
        return (
            f"You also own a private pond that only you can fish; the other fishermen cannot fish in it, "
            f"and it is separate from the lake. The pond has a carrying capacity of {self.cap} tons of fish. "
            f"At the beginning of each month, you decide how many fish between 0 and {self.cap} tons to take "
            f"from the pond, and that amount is then removed from the pond. At the end of the month, the "
            f"remaining fish in the pond will reproduce and double once, up to its carrying capacity. "
            f"For every ton of fish taken from the pond, you earn one thousand dollars, as for the lake. "
            f"Each month you are asked two separate questions: first how many tons to take from your pond, "
            f"then how many tons of fish to catch from the lake; answer each question about that water only. "
            f"Right now there are {self.tons} tons of fish in your pond."
        )

    def introduce(self, persona, current_time: datetime) -> None:
        from simulation.persona.common import PersonaEvent
        persona.store.store_event(PersonaEvent(
            self.ownership_text(), created=current_time,
            expiration=current_time + timedelta(days=3650), always_include=True,
        ))
        self.introduced = True

    def record_month(self, persona, current_time: datetime) -> None:
        """After the stranger has observed its lake catch: one plain observation about the pond."""
        from simulation.persona.common import PersonaEvent
        from simulation.scenarios.common.environment.concurrent_env import (
            get_discussion_day, get_expiration_next_month,
        )
        row = self._open_row or (self.rows[-1] if self.rows else None)
        if row is None:
            return
        text = (f"This month you took {row['taken']} tons of fish from your private pond, which had "
                f"{row['tons_before']} tons at the start of the month. Your pond now has "
                f"{row['tons_after_take']} tons of fish before reproduction. The pond is separate from "
                f"the lake; your catch from the lake is decided and counted separately.")
        persona.store.store_event(PersonaEvent(
            text, created=get_discussion_day(current_time) - timedelta(minutes=1),
            expiration=get_expiration_next_month(current_time), always_include=True,
        ))


def ask_pond(persona, obs, pond: Pond) -> tuple[int, bool]:
    """One traced prompt: how much does the stranger take from its pond this month?

    Same structure, reasoning cue and two-stage parse as upstream's catch prompt
    (``act_prompts.prompt_action_choose_amount_of_fish_to_catch``), so parse
    fallbacks are detected and counted identically. Returns (tons, parsed).
    """
    from simulation.scenarios.fishing.agents.persona_v3.cognition.utils import (
        get_sytem_prompt, location_time_info, memory_prompt, reasoning_steps_prompt,
    )

    model = persona.act.model
    identity = persona.identity
    persona.current_time = obs.current_time
    memories = persona.retrieve.retrieve([obs.current_location], 10)

    logger = getattr(model, "trace_logger", None)
    fallbacks_before = logger.n_parse_fallbacks if logger is not None else 0
    lm = model.start_chain(identity.name, "semataxy_pond", "pond_take")
    with user():
        lm += f"{get_sytem_prompt(identity)}\n"
        lm += location_time_info(obs.current_location, obs.current_time)
        lm += memory_prompt(identity, memories)
        lm += "\n"
        lm += (f"Your private pond, which only you can fish, has {pond.tons} tons of fish right now "
               f"(carrying capacity {pond.cap} tons). ")
        lm += (f"Task: With a range set between 0-{pond.tons}, how many tons of fish would you take "
               f"from your private pond this month? ")
        lm += reasoning_steps_prompt()
        lm += ' Put the final answer after "Answer:", example Answer: N tons.'
    with assistant():
        lm = model.gen(lm, "reasoning", stop_regex=r"Answer:|So, the answer is:|\*\*Answer\*\*:",
                       save_stop_text=True)
        lm = model.find(lm, regex=r"\d+", default_value="0", stop_regex="tons", name="option")
        option = int(lm["option"])
    model.end_chain(identity.name, lm)

    # The wrapper counts a parse fallback whenever upstream returned its default value.
    parsed = logger is None or logger.n_parse_fallbacks == fallbacks_before
    return option, parsed
