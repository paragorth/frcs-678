"""Runtime patches to upstream GovSim, applied without editing the submodule.

Each patch is small, named, and justified. Baseline behaviour is unchanged
except where noted; the point is that two runs with the same seed and manifest
produce the same prompts, which the branching design depends on.
"""

from __future__ import annotations

from datetime import datetime

from simulation.persona.cognition.retrieve import RetrieveComponent

_APPLIED = set()


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
    return sorted(_APPLIED)
