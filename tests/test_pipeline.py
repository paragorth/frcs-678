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
