"""Run one GovSim fishery simulation from a manifest.

This reimplements the ~40 lines of upstream ``simulation/main.py`` and
``scenarios/fishing/run.py`` that build models and drive the loop, so that:

* model wrappers come from the manifest (one HTTP endpoint per label) rather
  than Hydra's ``llm``/``mix_llm`` blocks;
* the environment is reset with the run seed (upstream calls ``env.reset()``
  with ``seed=None``, so stochastic allocation was unseeded);
* every call is traced to JSONL instead of wandb;
* the resolved config, upstream revisions and cache statistics are recorded.

The environment, personas, prompts and cognition are upstream's, unchanged.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from typing import Optional

import numpy as np
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from . import GOVSIM_ROOT, PATHFINDER_ROOT, REPO_ROOT, compat
from .backend import OpenAICompatibleAPI, ResponseCache
from .forecast import ForecastLog, ask_forecast
from .manifest import Manifest
from .tracing import TraceLogger, TracedModelWrapper


def _git_rev(path: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def compose_config(experiment: str, seed: int, overrides: list):
    """Compose upstream's Hydra config for ``experiment`` with absolute search paths."""
    GlobalHydra.instance().clear()
    conf_dir = os.path.join(GOVSIM_ROOT, "simulation", "conf")
    searchpath = [
        f"file://{os.path.join(GOVSIM_ROOT, 'simulation', 'scenarios', s, 'conf')}"
        for s in ("fishing", "sheep", "pollution")
    ]
    with initialize_config_dir(version_base=None, config_dir=conf_dir):
        cfg = compose(
            config_name="config",
            overrides=[
                f"experiment={experiment}",
                f"seed={seed}",
                "debug=true",
                "hydra.searchpath=[" + ",".join(searchpath) + "]",
                *overrides,
            ],
        )
    return cfg


def build_wrappers(manifest: Manifest, logger: TraceLogger, cache: ResponseCache):
    """One backend per model label; one wrapper per agent slot sharing it."""
    backends = {
        label: OpenAICompatibleAPI(spec, manifest.seed, cache)
        for label, spec in manifest.models.items()
    }
    wrappers = {}
    for label, backend in backends.items():
        spec = manifest.models[label]
        wrappers[label] = TracedModelWrapper(
            backend, logger, temperature=spec.temperature, top_p=spec.top_p, seed=manifest.seed
        )
    agent_wrappers = [wrappers[label] for label in manifest.agents]
    framework_wrapper = wrappers[manifest.framework]
    return agent_wrappers, framework_wrapper


def run(manifest: Manifest, out_root: str, run_id: Optional[str] = None,
        seed: Optional[int] = None, embedding_device: str = "cpu") -> str:
    seed = manifest.seed if seed is None else seed
    manifest.seed = seed
    run_id = run_id or f"{manifest.name}_s{seed}_{int(time.time())}"
    run_dir = os.path.join(out_root, run_id)
    os.makedirs(run_dir, exist_ok=True)

    patches = compat.apply_all()
    cfg = compose_config(manifest.experiment, seed, manifest.overrides)
    exp = cfg.experiment
    if exp.scenario != "fishing":
        raise NotImplementedError("only the fishery scenario is wired up so far")

    # Record everything needed to reproduce this run before doing any work.
    record = {
        "run_id": run_id,
        "manifest": manifest.to_dict(),
        "resolved_config": OmegaConf.to_container(cfg, resolve=True),
        "upstream": {
            "govsim_commit": _git_rev(GOVSIM_ROOT),
            "pathfinder_commit": _git_rev(PATHFINDER_ROOT),
            "overlay_commit": _git_rev(REPO_ROOT),
        },
        "patches": patches,
        "python": sys.version,
        "started": time.time(),
    }
    with open(os.path.join(run_dir, "run.json"), "w") as f:
        json.dump(record, f, indent=2, default=str)

    # Seed everything upstream seeds (transformers.set_seed does torch+numpy+random).
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cache = ResponseCache(manifest.cache_dir or os.path.join(out_root, "_cache"))
    logger = TraceLogger(run_dir, run_id)
    agent_wrappers, framework_wrapper = build_wrappers(manifest, logger, cache)

    from simulation.persona import EmbeddingModel
    embedding_model = EmbeddingModel(device=embedding_device)

    status = "unknown"
    termination_reason = None
    try:
        final = _run_fishing(exp, logger, agent_wrappers, framework_wrapper, embedding_model,
                             run_dir, seed, forecast_probe=manifest.forecast)
        status = "completed"
        termination_reason = final["termination_reason"]
    except Exception as e:  # keep partial logs; no silent omission of crashes
        import traceback
        status = "crashed"
        with open(os.path.join(run_dir, "crash.txt"), "w") as f:
            f.write(traceback.format_exc())
        termination_reason = f"crash: {type(e).__name__}: {e}"
        raise
    finally:
        record.update(
            {
                "finished": time.time(),
                "status": status,
                "termination_reason": termination_reason,
                "llm_calls": logger.n_calls,
                "parse_fallbacks": logger.n_parse_fallbacks,
                "token_in": logger.token_usage_in,
                "token_out": logger.token_usage_out,
                "cache_hits": cache.hits,
                "cache_misses": cache.misses,
            }
        )
        with open(os.path.join(run_dir, "run.json"), "w") as f:
            json.dump(record, f, indent=2, default=str)
        logger.close()
    return run_dir


def _run_fishing(cfg, logger, wrappers, framework_wrapper, embedding_model, experiment_storage, seed,
                 forecast_probe: bool = True):
    """Upstream scenarios/fishing/run.py, with seeded reset and per-call game context."""
    from simulation.persona.common import PersonaIdentity
    from simulation.scenarios.fishing.agents.persona_v3 import FishingPersona
    from simulation.scenarios.fishing.agents.persona_v3.cognition import utils as cognition_utils
    from simulation.scenarios.fishing.environment import FishingConcurrentEnv, FishingPerturbationEnv

    assert cfg.agent.agent_package == "persona_v3"
    cognition_utils.SYS_VERSION = {
        "v3": "v3", "v3_p1": "v3_p1", "v3_p2": "v3_p2", "v3_p3": "v3_p3", "v3_nocom": "v3_nocom",
    }.get(cfg.agent.system_prompt, "v1")
    cognition_utils.REASONING = cfg.agent.cot_prompt

    personas = {
        f"persona_{i}": FishingPersona(
            cfg.agent, wrappers[i], framework_wrapper, embedding_model,
            os.path.join(experiment_storage, f"persona_{i}"),
        )
        for i in range(5)
    }
    identities = {
        f"persona_{i}": PersonaIdentity(agent_id=f"persona_{i}", **cfg.personas[f"persona_{i}"])
        for i in range(cfg.personas.num)
    }
    agent_name_to_id = {obj.name: k for k, obj in identities.items()}
    agent_name_to_id["framework"] = "framework"
    agent_id_to_name = {v: k for k, v in agent_name_to_id.items()}

    for p in personas:
        personas[p].init_persona(p, identities[p], social_graph=None)
    for p in personas:
        for q in personas:
            personas[p].add_reference_to_other_persona(personas[q])

    if cfg.env.class_name == "fishing_perturbation_env":
        env = FishingPerturbationEnv(cfg.env, experiment_storage, agent_id_to_name)
    elif cfg.env.class_name == "fishing_perturbation_concurrent_env":
        env = FishingConcurrentEnv(cfg.env, experiment_storage, agent_id_to_name)
    else:
        raise ValueError(f"Unknown environment class: {cfg.env.class_name}")

    # Upstream: env.reset() -> RandomState(None), and PerturbationEnv.reset()
    # takes no seed argument at all. Reseed the allocation RNG right after reset;
    # nothing draws from it before the first harvest.
    agent_id, obs = env.reset()
    env.random = np.random.RandomState(seed)

    stats_keys = ["conversation_resource_limit", *[f"persona_{i}_collected_resource" for i in range(5)]]
    # Forecast probe (semataxy/forecast.py): asked before each catch decision; never written to memory.
    forecast = ForecastLog(experiment_storage) if forecast_probe else None
    settled_upto = -1
    while True:
        rnd = env.num_round
        logger.game_context = {"round": rnd, "env_phase": env.phase, "acting": agent_id}
        agent = personas[agent_id]
        if (forecast is not None and obs.current_location == "lake" and obs.phase == "lake"
                and obs.current_resource_num > 0):
            # The first lake observation of a round reveals last month's outcome: settle it.
            if rnd - 1 in forecast.rows and rnd - 1 > settled_upto:
                forecast.settle(rnd - 1, obs.current_resource_num)
                settled_upto = rnd - 1
            if agent_id not in forecast.rows.get(rnd, {}):
                others = [agent_id_to_name[a] for a, loc in obs.current_location_agents.items()
                          if loc == "lake" and a != agent_id]
                forecast.add(rnd, ask_forecast(agent, obs, others))
                logger.game_context = {"round": rnd, "env_phase": env.phase, "acting": agent_id}
        action = agent.loop(obs)
        agent_id, obs, rewards, termination = env.step(action)
        stats = {k: action.stats[k] for k in stats_keys if k in action.stats}
        if forecast is not None:
            for k, v in stats.items():
                if k.endswith("_collected_resource"):
                    forecast.record_catch(rnd, k[: -len("_collected_resource")], v)
        done = bool(np.any(list(termination.values())))
        logger.log_game({"num_resource": obs.current_resource_num, "round": env.num_round,
                         "phase": env.phase, **stats}, last_log=done)
        if done:
            break

    if forecast is not None:
        # Rounds not yet settled (the last one, or all after a collapse): others' catches are
        # known, next month's stock is not.
        for r in sorted(forecast.rows):
            if r > settled_upto:
                forecast.settle(r, None)
        forecast.save()

    env.save_log()
    for p in personas:
        personas[p].memory.save()

    collapsed = env.internal_global_state["resource_in_pool"] < 5
    return {
        "termination_reason": "collapse" if collapsed else "horizon",
        "rounds": env.num_round,
    }
