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

from . import GOVSIM_ROOT, PATHFINDER_ROOT, REPO_ROOT, compat, scenarios
from .backend import OpenAICompatibleAPI, ResponseCache
from .forecast import ForecastLog, ask_forecast
from .manifest import Manifest
from .pond import Pond, ask_pond
from .successor import replace_residents
from .livegame import LiveLearner
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
    # Overlay configs (e.g. conf/experiment/persona/mild_fisherman.yaml) without touching upstream.
    searchpath.append(f"file://{os.path.join(REPO_ROOT, 'conf')}")
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
    scenarios.get(exp.scenario)  # raises for a world the overlay does not know
    if manifest.pond_cap and exp.scenario != "fishing":
        raise NotImplementedError("the pond overlay is written for the fishery only")

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
    live = None
    if manifest.live_learning:
        slot = manifest.live_learning.get("slot", "persona_4")
        backend = agent_wrappers[int(slot.split("_")[-1])].base_lm  # the slot's OpenAICompatibleAPI
        queue = os.environ.get("SEMATAXY_LIVE_QUEUE") or os.path.join(out_root, "_live_queue")
        server = backend.spec.base_url.rsplit("/v1", 1)[0]
        live = LiveLearner(manifest.live_learning, run_dir, slot, backend, queue, server, run_id)
        record["live_learning"] = {"slot": slot, "queue": queue, "server": server, **manifest.live_learning}

    from simulation.persona import EmbeddingModel
    embedding_model = EmbeddingModel(device=embedding_device)

    status = "unknown"
    termination_reason = None
    try:
        final = _run_fishing(exp, logger, agent_wrappers, framework_wrapper, embedding_model,
                             run_dir, seed, forecast_probe=manifest.forecast,
                             pond_cap=manifest.pond_cap, pond_slot=manifest.pond_slot,
                             successor_round=manifest.successor_round, live=live)
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
                 forecast_probe: bool = True, pond_cap: int = 0, pond_slot: str = "persona_4",
                 successor_round: Optional[int] = None, live=None):
    """Upstream scenarios/<world>/run.py, with seeded reset and per-call game context."""
    from simulation.persona.common import PersonaIdentity

    sc = scenarios.get(cfg.scenario)
    Persona = scenarios.import_persona(sc)
    cognition_utils = scenarios.cognition_utils(sc)

    assert cfg.agent.agent_package == "persona_v3"
    # Same mapping as upstream's run.py for each world: unknown versions fall back to v1.
    cognition_utils.SYS_VERSION = cfg.agent.system_prompt if cfg.agent.system_prompt in sc.sys_versions else "v1"
    cognition_utils.REASONING = cfg.agent.cot_prompt

    personas = {
        f"persona_{i}": Persona(
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
    compat.register_speakers(personas)  # each agent speaks in the group chat with its own model

    env = scenarios.import_env(sc, cfg.env.class_name)(cfg.env, experiment_storage, agent_id_to_name)
    pool = env.POOL_LOCATION  # "lake", "pasture" or "factory"

    # Upstream: env.reset() -> RandomState(None), and PerturbationEnv.reset()
    # takes no seed argument at all. Reseed the allocation RNG right after reset;
    # nothing draws from it before the first harvest.
    agent_id, obs = env.reset()
    env.random = np.random.RandomState(seed)

    stats_keys = ["conversation_resource_limit", *[f"persona_{i}_collected_resource" for i in range(5)]]
    # Forecast probe (semataxy/forecast.py): asked before each catch decision; never written to memory.
    forecast = ForecastLog(experiment_storage) if forecast_probe else None
    settled_upto = -1
    # Dispensability overlay (semataxy/pond.py): a private pond for one slot, declared in the manifest.
    pond = Pond(pond_cap, experiment_storage, pond_slot)
    pond_stats = {}
    done_swap = False
    while True:
        rnd = env.num_round
        logger.game_context = {"round": rnd, "env_phase": env.phase, "acting": agent_id}
        agent = personas[agent_id]
        is_pond_owner = pond.active and agent_id == pond.slot
        if is_pond_owner and obs.phase == pool and not pond.introduced:
            # Entry: the one standing fact about its world the stranger is given (declared intervention).
            pond.introduce(agent, obs.current_time)
        if (forecast is not None and obs.current_location == pool and obs.phase == pool
                and obs.current_resource_num > 0):
            # The first lake observation of a round reveals last month's outcome: settle it.
            if rnd - 1 in forecast.rows and rnd - 1 > settled_upto:
                forecast.settle(rnd - 1, obs.current_resource_num)
                settled_upto = rnd - 1
            if agent_id not in forecast.rows.get(rnd, {}):
                others = [agent_id_to_name[a] for a, loc in obs.current_location_agents.items()
                          if loc == pool and a != agent_id]
                forecast.add(rnd, ask_forecast(agent, obs, others, scenario=sc.name))
                logger.game_context = {"round": rnd, "env_phase": env.phase, "acting": agent_id}
        pond_stats = {}
        if is_pond_owner and obs.phase == pool and obs.current_resource_num > 0:
            # Before the lake decision: how much does the stranger take from its own pond?
            wanted, parsed = ask_pond(agent, obs, pond)
            tons_before = pond.tons
            taken = pond.take(rnd, wanted, parsed)
            pond_stats = {"pond_tons_before": tons_before, "pond_wanted": wanted, "pond_taken": taken}
            logger.game_context = {"round": rnd, "env_phase": env.phase, "acting": agent_id}
        if is_pond_owner and obs.phase == "pool_after_harvesting" and pond.rows:
            # After observing its lake catch: one plain observation about the pond (declared intervention).
            pond.record_month(agent, obs.current_time)
        action = agent.loop(obs)
        agent_id, obs, rewards, termination = env.step(action)
        if env.num_round > rnd:
            pond.regrow()  # month end: the pond reproduces like the lake
            if live is not None and live.slot in env.agents:
                # Live learning (semataxy/livegame.py): consolidate the stranger's month into its weights.
                row = live.consolidate(personas[live.slot], rnd)
                logger.log_game({"event": "live_consolidation", "round": rnd, "n_facts": row["n_facts"],
                                 "commit": bool(row["result"].get("gate", {}).get("commit")),
                                 "serving_model": row["serving_model"], "status": row["result"].get("status")})
            if successor_round is not None and env.num_round == successor_round and not done_swap:
                # Inheritance (semataxy/successor.py): the four residents leave, naive agents arrive.
                residents = [f"persona_{i}" for i in range(4)]
                personas = replace_residents(personas, env, cfg.agent, wrappers, framework_wrapper,
                                             embedding_model, experiment_storage, identities, residents, 2,
                                             persona_cls=Persona)
                compat.register_speakers(personas)
                logger.log_game({"event": "residents_replaced", "slots": residents,
                                 "round": env.num_round, "phase": env.phase,
                                 "num_resource": env.internal_global_state["resource_in_pool"]})
                done_swap = True
                if agent_id in residents:
                    obs = env._observe(agent_id)  # the pending observation was built for the departed agent
        stats = {k: action.stats[k] for k in stats_keys if k in action.stats}
        stats.update(pond_stats)
        done = bool(np.any(list(termination.values())))
        if forecast is not None and (env.num_round > rnd or done):
            # Month over: record what each agent actually took (the env's allocation), not what it
            # asked for. The persona's own stat is its demand; they differ in rationed months.
            # "collected_resource" is cumulative upstream; "last_collected_resource" is this month's.
            for a_id, v in env.internal_global_state["last_collected_resource"].items():
                if a_id in env.agents:
                    forecast.record_catch(rnd, a_id, v)
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

    if pond.active:
        pond.save()
    env.save_log()
    for p in personas:
        personas[p].memory.save()

    # Upstream stops when the post-harvest stock is under 5 (checked before the month-end doubling)
    # or at max_num_rounds. The stock readable here is already doubled, so 3-4 ton collapses were
    # being labelled "horizon"; decide by the round count instead.
    collapsed = env.num_round < cfg.env.max_num_rounds
    return {
        "termination_reason": "collapse" if collapsed else "horizon",
        "rounds": env.num_round,
    }
