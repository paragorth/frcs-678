"""End-to-end tests against the scripted mock server.

These prove the plumbing (composition, dispatch, tracing, caching, branch
identity, analysis), not anything about models.
"""

import json
import os
import socket
import subprocess
import sys
import time

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import semataxy  # noqa: E402  (sets sys.path for upstream)
from semataxy.analysis import analyze_run  # noqa: E402
from semataxy.backend import ResponseCache  # noqa: E402
from semataxy.manifest import Manifest  # noqa: E402
from semataxy.runner import run  # noqa: E402


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


@pytest.fixture(scope="session")
def mock_url():
    port = _free_port()
    proc = subprocess.Popen([sys.executable, os.path.join(REPO, "tools", "mock_server.py"), "--port", str(port)])
    url = f"http://127.0.0.1:{port}/v1"
    for _ in range(50):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
            break
        except OSError:
            time.sleep(0.2)
    os.environ["MOCK_URL"] = url
    yield url
    proc.terminate()


@pytest.fixture(scope="session")
def embedding():
    from simulation.persona import EmbeddingModel
    return EmbeddingModel(device="cpu")


def _calls(run_dir):
    with open(os.path.join(run_dir, "llm_calls.jsonl")) as f:
        return [json.loads(line) for line in f]


def test_cache_roundtrip(tmp_path):
    c = ResponseCache(str(tmp_path))
    k = ResponseCache.key({"a": 1, "b": [1, 2]})
    assert c.get(k) is None
    c.put(k, {"response": "x"})
    assert c.get(k)["response"] == "x"
    assert ResponseCache.key({"b": [1, 2], "a": 1}) == k


def test_smoke_run_and_analysis(mock_url, tmp_path):
    m = Manifest.load(os.path.join(REPO, "manifests", "smoke_mock.yaml"))
    m.overrides = ["experiment.env.max_num_rounds=1"]
    out = run(m, str(tmp_path), run_id="smoke", seed=3)
    assert os.path.exists(os.path.join(out, "log_env.json"))
    rec = json.load(open(os.path.join(out, "run.json")))
    assert rec["status"] == "completed"
    assert rec["llm_calls"] > 0
    assert rec["parse_fallbacks"] >= 1, "mock-weak should trigger the default-value path"
    a = analyze_run(out)
    assert a["rounds"] == 1
    assert a["catch_by_agent"]["persona_4"] == 30  # greedy policy
    assert a["parse_fallbacks_by_slot"]  # populated per persona slot
    # forecast probe: one traced, read-only prompt per agent per round, settled against outcomes
    with open(os.path.join(out, "forecast.jsonl")) as f:
        fc = [json.loads(line) for line in f]
    assert len(fc) == 5 and all(r["parsed"] for r in fc)
    assert all(r["actual_others"] is not None and r["err_others"] is not None for r in fc)
    assert any(c["phase"] == "semataxy_forecast" for c in _calls(out))
    assert a["forecast_probes"] == 5 and a["forecast_err_others_mean"] is not None


def test_branch_identity(mock_url, tmp_path):
    """Control and treatment share an identical pre-entry history."""
    base = Manifest.load(os.path.join(REPO, "manifests", "smoke_mock.yaml"))
    base.experiment = "fish_perturbation_outsider"
    base.cache_dir = str(tmp_path / "cache")
    entry = 2
    common = [
        f"experiment.env.perturbations.0.perturbation.round={entry}",
        "experiment.env.max_num_rounds=3",
        "experiment/persona@experiment.personas.persona_4=baseline_fisherman",
    ]
    base.agents = ["sustainable"] * 5
    base.overrides = list(common)
    ctrl = run(base, str(tmp_path), run_id="ctrl", seed=5)

    treat = Manifest.load(os.path.join(REPO, "manifests", "smoke_mock.yaml"))
    treat.experiment = base.experiment
    treat.cache_dir = base.cache_dir
    treat.agents = ["sustainable"] * 4 + ["greedy"]
    treat.overrides = list(common)
    tr = run(treat, str(tmp_path), run_id="treat", seed=5)

    c, t = _calls(ctrl), _calls(tr)
    pre_c = [(r["prompt"], r["response"]) for r in c if r["round"] < entry]
    pre_t = [(r["prompt"], r["response"]) for r in t if r["round"] < entry]
    assert pre_c and pre_c == pre_t, "pre-entry prompts/responses must be identical across branches"
    assert all(r.get("cache") == "hit" for r in t if r["round"] < entry), "treatment replays pre-entry calls"
    ac, at = analyze_run(ctrl), analyze_run(tr)
    assert ac["entry_round"] == entry and at["entry_round"] == entry
    assert at["newcomer_mean_catch"] > ac["newcomer_mean_catch"]
    # 4 residents fish before entry, 5 agents after
    assert len(ac["catch_matrix"][str(entry - 1)]) == 4
    assert len(ac["catch_matrix"][str(entry)]) == 5


def test_pond_overlay(mock_url, tmp_path):
    """Dispensability: the stranger gets a private pond; residents and pre-entry history untouched."""
    base = Manifest.load(os.path.join(REPO, "manifests", "smoke_mock.yaml"))
    base.experiment = "fish_perturbation_outsider"
    base.cache_dir = str(tmp_path / "cache")
    entry = 1
    common = [
        f"experiment.env.perturbations.0.perturbation.round={entry}",
        "experiment.env.max_num_rounds=3",
        "experiment/persona@experiment.personas.persona_4=baseline_fisherman",
    ]
    base.agents = ["sustainable"] * 5
    base.overrides = list(common)
    ctrl = run(base, str(tmp_path), run_id="nopond", seed=7)

    treat = Manifest.load(os.path.join(REPO, "manifests", "smoke_mock.yaml"))
    treat.experiment = base.experiment
    treat.cache_dir = base.cache_dir
    treat.agents = ["sustainable"] * 5
    treat.overrides = list(common)
    treat.pond_cap = 100
    tr = run(treat, str(tmp_path), run_id="pond100", seed=7)

    assert not os.path.exists(os.path.join(ctrl, "pond.jsonl"))
    with open(os.path.join(tr, "pond.jsonl")) as f:
        pond = [json.loads(line) for line in f]
    # one row per month after entry (rounds 1 and 2), all parsed, mock takes half the pond
    assert [r["round"] for r in pond] == [1, 2]
    assert pond[0]["tons_before"] == 100 and pond[0]["taken"] == 50 and pond[0]["parsed"]
    assert pond[0]["tons_after_regrow"] == 100  # 50 left, doubles back to the cap
    # pre-entry history is byte-identical to the no-pond control (branch identity holds)
    c, t = _calls(ctrl), _calls(tr)
    pre_c = [(r["prompt"], r["response"]) for r in c if r["round"] < entry]
    pre_t = [(r["prompt"], r["response"]) for r in t if r["round"] < entry]
    assert pre_c and pre_c == pre_t
    # residents' prompts never mention the pond; the stranger's do, via the always-included memory
    res_names = {"John", "Kate", "Jack", "Emma"}
    assert not any("pond" in r["prompt"] for r in t if r["agent"] in res_names)
    assert any(r["phase"] == "semataxy_pond" for r in t)
    acts = [r for r in t if r["phase"] == "fishing_cognition_act" and r["agent"] == "Luke"]
    assert acts and all("private pond" in r["prompt"] for r in acts)
    # analysis picks it up; run.json records the dial
    a = analyze_run(tr)
    assert a["pond_cap"] == 100 and a["pond_months"] == 2 and a["pond_take_mean"] == 50
    rec = json.load(open(os.path.join(tr, "run.json")))
    assert rec["manifest"]["pond_cap"] == 100


def test_successor_overlay(mock_url, tmp_path):
    """Inheritance: residents replaced by naive agents at successor_round; stranger persists."""
    m = Manifest.load(os.path.join(REPO, "manifests", "smoke_mock.yaml"))
    m.experiment = "fish_perturbation_outsider"
    m.cache_dir = str(tmp_path / "cache")
    m.agents = ["sustainable"] * 4 + ["greedy"]
    m.overrides = [
        "experiment.env.perturbations.0.perturbation.round=1",
        "experiment.env.max_num_rounds=4",
        "experiment/persona@experiment.personas.persona_4=baseline_fisherman",
    ]
    m.successor_round = 2
    out = run(m, str(tmp_path), run_id="succ", seed=11)
    rec = json.load(open(os.path.join(out, "run.json")))
    assert rec["status"] == "completed" and rec["manifest"]["successor_round"] == 2
    # elders' memories kept, newcomers' memories written separately
    for i in range(4):
        assert os.path.exists(os.path.join(out, f"persona_{i}", "nodes.json"))
        assert os.path.exists(os.path.join(out, f"persona_{i}_gen2", "nodes.json"))
    assert not os.path.exists(os.path.join(out, "persona_4_gen2"))
    with open(os.path.join(out, "game.jsonl")) as f:
        game = [json.loads(line) for line in f]
    ev = [g for g in game if g.get("event") == "residents_replaced"]
    assert len(ev) == 1 and ev[0]["round"] == 2
    a = analyze_run(out)
    assert a["rounds"] == 4 and a["entry_round"] == 1 and a["successor_round"] == 2
    assert a["post_swap_rounds"] == 2 and a["stranger_catch_post_swap"] is not None
    assert len(a["catch_matrix"]["3"]) == 5  # five agents still fishing after the swap
    # the newcomers start from nothing: their first retrieval sees no earlier months
    calls = _calls(out)
    first_new = [c for c in calls if c["round"] == 2 and c["agent"] == "John" and c["phase"] == "fishing_cognition_act"]
    # newcomers take a surviving agent's clock (not January 2024), so chat memories are not back-dated past expiry
    assert first_new and "Date: 2024-01-01" not in first_new[0]["prompt"] and "Date: 2024-0" in first_new[0]["prompt"]


@pytest.mark.parametrize("scenario,experiment,persona", [
    ("sheep", "sheep_perturbation_outsider", "baseline_shepherd"),
    ("pollution", "pollution_perturbation_outsider", "baseline_factory_owner"),
])
def test_other_worlds(mock_url, tmp_path, scenario, experiment, persona):
    """Pasture and river run through the same overlay: probe parsed, stranger enters, analysis works."""
    m = Manifest.load(os.path.join(REPO, "manifests", "smoke_mock.yaml"))
    m.experiment = experiment
    m.cache_dir = str(tmp_path / "cache")
    m.agents = ["sustainable"] * 4 + ["greedy"]
    m.overrides = [
        "experiment.env.perturbations.0.perturbation.round=1",
        "experiment.env.max_num_rounds=2",
        f"experiment/persona@experiment.personas.persona_4={persona}",
    ]
    out = run(m, str(tmp_path), run_id=f"w_{scenario}", seed=3)
    rec = json.load(open(os.path.join(out, "run.json")))
    assert rec["status"] == "completed"
    assert rec["resolved_config"]["experiment"]["scenario"] == scenario
    a = analyze_run(out)
    assert a["rounds"] == 2 and a["entry_round"] == 1 and len(a["catch_matrix"]["1"]) == 5
    with open(os.path.join(out, "forecast.jsonl")) as f:
        fc = [json.loads(line) for line in f]
    assert fc and all(r["parsed"] for r in fc), fc[:2]
    assert all(r["actual_others"] is not None for r in fc)
    calls = _calls(out)
    probe = next(c for c in calls if c["phase"] == "semataxy_forecast")
    assert "fish" not in json.dumps(probe["prompt"]).lower()
