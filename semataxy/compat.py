"""Runtime patches to upstream GovSim, applied without editing the submodule.

Each patch is small, named, and justified. Baseline behaviour is unchanged
except where noted; the point is that two runs with the same seed and manifest
produce the same prompts, which the branching design depends on.
"""

from __future__ import annotations

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
    return sorted(_APPLIED)


def register_speakers(personas: dict):
    """Called by the runner after personas are built (and again after a successor swap)."""
    SPEAKERS.clear()
    for p in personas.values():
        SPEAKERS[p.identity.name] = p.converse.model
