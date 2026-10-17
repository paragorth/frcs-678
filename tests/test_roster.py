"""The system prompt names only the agents present (semataxy/compat.py present_roster)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "third_party", "GovSim"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from types import SimpleNamespace
from semataxy import compat


def test_roster_follows_presence():
    compat.apply_all()
    from simulation.scenarios.fishing.agents.persona_v3.cognition import utils, converse_prompts, store_prompts
    john = SimpleNamespace(name="John", goals="")
    luke = SimpleNamespace(name="Luke", goals="")
    compat.set_present([])
    assert "four other fishermen (Kate, Jack, Emma, Luke)" in utils.get_sytem_prompt(john)
    compat.set_present(["John", "Kate", "Jack", "Emma"])
    t = utils.get_sytem_prompt(john)
    assert "three other fishermen (Kate, Jack, Emma)" in t and "Luke" not in t
    assert converse_prompts.get_sytem_prompt is utils.get_sytem_prompt and store_prompts.get_sytem_prompt is utils.get_sytem_prompt
    compat.set_present(["John", "Kate", "Jack", "Emma", "Luke"])
    assert "four other fishermen (Kate, Jack, Emma, Luke)" in utils.get_sytem_prompt(john)
    assert "four other fishermen (John, Kate, Jack, Emma)" in utils.get_sytem_prompt(luke)
    compat.set_present([])


def test_explicit_rules_example():
    import importlib
    compat.EXPLICIT_RULES = True
    compat.apply_all()
    from simulation.scenarios.fishing.agents.persona_v3.cognition import utils
    for ver in ("v1", "v3"):
        utils.SYS_VERSION = ver
        t = utils.get_sytem_prompt(SimpleNamespace(name="John", goals=""))
        assert "100 tons after reproduction" not in t and "not 100" in t and "five fishermen" not in t, (ver, t[-400:])
    compat.EXPLICIT_RULES = False
    assert "100 tons after reproduction" in utils.get_sytem_prompt(SimpleNamespace(name="John", goals=""))
