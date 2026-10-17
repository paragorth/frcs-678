"""Runtime patches to upstream GovSim, applied without editing the submodule.

Each patch is small, named, and justified. Baseline behaviour is unchanged
except where noted; the point is that two runs with the same seed and manifest
produce the same prompts, which the branching design depends on.
"""

from __future__ import annotations

import re
from datetime import datetime

from simulation.persona.cognition.retrieve import RetrieveComponent

_APPLIED = set()

# name -> the model wrapper that persona speaks with (filled by the runner once personas exist)
SPEAKERS: dict = {}


def _own_voice(orig):
    """Upstream's group chat is driven by one persona's converse component, which generates every
    utterance with *its own* model, whoever is speaking. In a mixed-model game that means a 14B
    resident writes the 32B stranger's lines. This wrapper routes each utterance to the speaker's own
    model. Found 17 Oct 2026; every stranger cell before that date had a mute stranger in the chat.
    """
    def prompt(model, current_persona, *args, **kwargs):
        return orig(SPEAKERS.get(current_persona.name, model), current_persona, *args, **kwargs)
    prompt.__wrapped__ = orig
    return prompt


def _deterministic_retrieve(self, focal_points, top_k):
    """Upstream builds a ``set`` of node objects and then sorts by ``created``
    only. Nodes with equal timestamps therefore come out in object-hash order,
    which differs between processes and changes the prompt text. Sorting by
    (created, node.id) is the same ranking with ties broken deterministically.
    """
    res = self._retrieve_dict(focal_points, top_k)
    nodes = {}
    for group in res.values():
        for node in group:
            nodes[node.id] = node
    ordered = sorted(nodes.values(), key=lambda n: (n.created, n.id))
    return [(n.created, n.description) for n in ordered]


# names of the agents currently in the game (the runner refreshes it every step); empty = no patch
PRESENT: set = set()
_NUM = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven"}
_ROSTER = re.compile(r"\b(four|three|five) other ([a-z]+(?: [a-z]+)?) \(([^)]*)\)")


def set_present(names):
    PRESENT.clear(); PRESENT.update(names)


# Track A (17 Oct 2026, pg): the upstream rules example teaches the wrong reproduction rule ("60 tons left ...
# 100 tons after reproduction" reads as "whatever is left refills to 100"). With SEMATAXY_RULES_EXPLICIT=1 the
# example is replaced by one that shows doubling without the cap.
import os as _os
EXPLICIT_RULES = _os.environ.get("SEMATAXY_RULES_EXPLICIT", "0") == "1"
_RE_EXAMPLE = re.compile(r" For example, if there are 90 tons of fish at the beginning of the month and the five fishermen catch a total of 30 fish,"
                         r" there will be 60 tons of fish left at the end of the month before reproduction, and 100 tons after reproduction\.")
EXPLICIT_EXAMPLE = (" For example, if there are 90 tons of fish at the beginning of the month and the fishermen catch a total of 50 tons,"
                    " 40 tons are left at the end of the month, and after reproduction there are 80 tons, not 100: the remaining fish double once."
                    " Only if at least 50 tons are left does the lake return to its capacity of 100 tons. If 10 tons are left, there will be 20.")


_RE_EXAMPLE_V1 = re.compile(r" So if there are 60 tons of fish remaining, there will be 100 tons after reproduction, or if there are 20 tons of fish remaining,"
                            r" there will be 40 tons of fish after reproduction\.")
EXPLICIT_EXAMPLE_V1 = (" So if there are 40 tons of fish remaining, there will be 80 tons after reproduction, not 100: the remaining fish double once,"
                       " and only if at least 50 tons remain does the lake return to 100. If 20 tons remain there will be 40; if 10 remain, 20.")


def _present_roster(orig):
    """Upstream hard-codes the five names into every system prompt, so the residents are told about the
    newcomer from month 0 ("along with four other fishermen (Kate, Jack, Emma, Luke)") and plan for five.
    Found by a cold read on 17 Oct 2026. With PRESENT set, the roster names only the agents in the game."""
    def prompt(persona, *args, **kwargs):
        text = orig(persona, *args, **kwargs)
        if EXPLICIT_RULES:
            text = _RE_EXAMPLE.sub(EXPLICIT_EXAMPLE, text)          # v3 wording (the one the runs use)
            text = _RE_EXAMPLE_V1.sub(EXPLICIT_EXAMPLE_V1, text)    # v1 wording
        if not PRESENT:
            return text
        others = [n for n in _ROSTER_ORDER if n in PRESENT and n != persona.name]
        m = _ROSTER.search(text)
        if not m or not others:
            return text
        return text[: m.start()] + f"{_NUM[len(others)]} other {m.group(2)} ({', '.join(others)})" + text[m.end():]
    prompt.__wrapped__ = orig
    return prompt


_ROSTER_ORDER = ["John", "Kate", "Jack", "Emma", "Luke"]


def apply_all():
    if "retrieve" not in _APPLIED:
        RetrieveComponent.retrieve = _deterministic_retrieve
        _APPLIED.add("retrieve")
    if "own_voice" not in _APPLIED:
        import importlib
        for world in ("fishing", "sheep", "pollution"):
            mod = importlib.import_module(f"simulation.scenarios.{world}.agents.persona_v3.cognition.converse")
            mod.prompt_converse_utterance_in_group = _own_voice(mod.prompt_converse_utterance_in_group)
        _APPLIED.add("own_voice")
    if "present_roster" not in _APPLIED:
        import importlib
        for world in ("fishing", "sheep", "pollution"):
            base = f"simulation.scenarios.{world}.agents.persona_v3.cognition"
            utils = importlib.import_module(base + ".utils")
            patched = _present_roster(utils.get_sytem_prompt)
            utils.get_sytem_prompt = patched
            for modname in ("converse_prompts", "reflect_prompts", "store_prompts", "act_prompts", "plan_prompts", "perceive_prompts", "retrieve_prompts"):
                try:
                    mod = importlib.import_module(f"{base}.{modname}")
                except ModuleNotFoundError:
                    continue
                if getattr(mod, "get_sytem_prompt", None) is not None:
                    mod.get_sytem_prompt = patched
        _APPLIED.add("present_roster")
    return sorted(_APPLIED)


def register_speakers(personas: dict):
    """Called by the runner after personas are built (and again after a successor swap)."""
    SPEAKERS.clear()
    for p in personas.values():
        SPEAKERS[p.identity.name] = p.converse.model
