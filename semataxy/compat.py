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


def _present_roster(orig):
    """Upstream hard-codes the five names into every system prompt, so the residents are told about the
    newcomer from month 0 ("along with four other fishermen (Kate, Jack, Emma, Luke)") and plan for five.
    Found by a cold read on 17 Oct 2026. With PRESENT set, the roster names only the agents in the game."""
    def prompt(persona, *args, **kwargs):
        text = orig(persona, *args, **kwargs)
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
