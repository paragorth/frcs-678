"""Successor (inheritance): the residents leave and naive newcomers arrive.

Handover section 17.2. A stranger joins four residents at the entry month and
plays with them; at ``successor_round`` the four residents are replaced by four
fresh agents of the same model with empty memories. The stranger is then the
only agent who knows how the group has been fishing. Readouts (analysis.py):
do the newcomers converge to the pre-swap level, does the stranger's own catch
rise once the elders are gone, does the lake survive.

This mirrors upstream's ``insert_outsider`` as closely as the design allows:
the new agents are initialised through the environment's own ``_init_agent``
(so, like upstream's outsider, their clock starts at January 2024), nothing is
written to anyone's memory, and no agent is told that a replacement happened.
The newcomers reuse the departed residents' names, because the five names are
hard-coded in upstream's prompts (docs/INVENTORY.md quirk 1); so the stranger's
memories of "John" now refer to a different agent with the same name. That is
a documented limitation, not a design choice.

Upstream is not edited: the swap happens in the runner loop between two
environment steps, exactly where upstream's own perturbation fires.
"""

from __future__ import annotations

import os


def replace_residents(personas: dict, env, agent_cfg, wrappers: list, framework_wrapper,
                      embedding_model, storage: str, identities: dict, slots: list[str],
                      generation: int) -> dict:
    """Replace ``slots`` with fresh personas; returns the updated personas dict."""
    from simulation.scenarios.fishing.agents.persona_v3 import FishingPersona

    departed = {s: personas[s] for s in slots}
    for p in departed.values():
        p.memory.save()  # keep what the elders knew

    for s in slots:
        i = int(s.split("_")[-1])
        fresh = FishingPersona(agent_cfg, wrappers[i], framework_wrapper, embedding_model,
                               os.path.join(storage, f"{s}_gen{generation}"))
        fresh.init_persona(s, identities[s], social_graph=None)
        personas[s] = fresh
        env._init_agent(s)  # counters, location and clock as for upstream's outsider

    # Everyone references everyone (including themselves), as upstream does at start-up.
    for p in personas.values():
        for q in personas.values():
            p.add_reference_to_other_persona(q)
    return personas
