"""Run manifest: everything needed to reproduce one simulation run.

A manifest names the upstream experiment config, which model serves each
agent slot and the framework (parsing/summarising) role, the seed, and any
Hydra overrides applied on top of the stock config. The runner writes the
resolved manifest, the upstream commit hashes, and the resolved Hydra config
into the run directory.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Optional

import yaml

from .backend import ModelSpec


@dataclass
class Manifest:
    name: str
    experiment: str  # upstream experiment id, e.g. fish_baseline_concurrent
    models: dict  # label -> ModelSpec
    agents: list  # per persona slot (persona_0..persona_4): model label
    framework: str  # model label for framework prompts
    seed: int = 42
    overrides: list = field(default_factory=list)  # extra Hydra overrides
    intervention: Optional[dict] = None  # free-form description, recorded only
    notes: str = ""
    cache_dir: Optional[str] = None

    @staticmethod
    def load(path: str, env: Optional[dict] = None) -> "Manifest":
        with open(path) as f:
            raw = yaml.safe_load(f)
        env = env or os.environ
        models = {}
        for label, spec in raw["models"].items():
            spec = dict(spec)
            # allow ${VAR} substitution in base_url so one manifest serves
            # the laptop mock server and the cluster
            if isinstance(spec.get("base_url"), str):
                spec["base_url"] = os.path.expandvars(spec["base_url"])
            models[label] = ModelSpec(name=label, **spec)
        agents = list(raw["agents"])
        if len(agents) != 5:
            raise ValueError("agents must list exactly 5 slots (persona_0..persona_4)")
        unknown = [a for a in agents + [raw["framework"]] if a not in models]
        if unknown:
            raise ValueError(f"unknown model labels: {unknown}")
        return Manifest(
            name=raw["name"],
            experiment=raw["experiment"],
            models=models,
            agents=agents,
            framework=raw["framework"],
            seed=int(raw.get("seed", 42)),
            overrides=list(raw.get("overrides", [])),
            intervention=raw.get("intervention"),
            notes=raw.get("notes", ""),
            cache_dir=raw.get("cache_dir"),
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["models"] = {k: v.fingerprint() for k, v in self.models.items()}
        return d
