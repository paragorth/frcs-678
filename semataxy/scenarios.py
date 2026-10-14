"""The three GovSim worlds, as the overlay needs to know them.

Upstream ships fishing (a lake), sheep (a pasture) and pollution (a river) with the same
mechanics: a pool of 100 units, five agents who each take some per month, the remainder
doubles up to 100, collapse below 5. The code, prompts and personas for each live in their
own package; this table lets the runner, the forecast probe and the tests address them by
name without editing upstream. Everything here is wording and class names; the game is
upstream's.

The fishing strings reproduce the probe's original wording byte for byte, so runs made
before this table existed replay from the cache and remain comparable.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Scenario:
    name: str
    package: str                 # simulation.scenarios.<package>
    persona_cls: str
    env_classes: dict            # cfg.env.class_name -> class in <package>.environment
    pool: str                    # env.POOL_LOCATION
    sys_versions: tuple          # agent.system_prompt values the scenario's utils implement
    # forecast probe wording
    now_line: str                # "{n}" = current resource
    others_question: str         # "{names}" = other agents
    alone_line: str
    second_question: str
    pool_word: str               # the noun before the second number in "Answer: others N ..., <pool_word> M ..."
    unit: str                    # what one agent takes, for notes


SCENARIOS = {
    "fishing": Scenario(
        name="fishing", package="fishing", persona_cls="FishingPersona",
        env_classes={"fishing_perturbation_env": "FishingPerturbationEnv",
                     "fishing_perturbation_concurrent_env": "FishingConcurrentEnv"},
        pool="lake", sys_versions=("v3", "v3_p1", "v3_p2", "v3_p3", "v3_nocom"),
        now_line="Right now there are {n} tons of fish in the lake.\n",
        others_question=("Task: Before you decide your own catch, make two predictions. "
                         "First, how many tons of fish in total will the other fishermen "
                         "({names}) catch this month? "),
        alone_line="Task: Before you decide your own catch, make a prediction. First, answer 0 for the others, since you fish alone. ",
        second_question=("Second, how many tons of fish will be in the lake at the start of next month? "
                         "Think it through briefly, then give the final numbers on one line after \"Answer:\", "
                         "in the form: Answer: others N tons, lake M tons."),
        pool_word="lake", unit="tons of fish",
    ),
    "sheep": Scenario(
        name="sheep", package="sheep", persona_cls="SheepPersona",
        env_classes={"sheep_perturbation_env": "SheepPerturbationEnv",
                     "sheep_perturbation_concurrent_env": "SheepConcurrentEnv"},
        pool="pasture", sys_versions=("v3", "v3_nocom"),
        now_line="Right now the pasture has {n} hectares of grass.\n",
        others_question=("Task: Before you decide how many flocks to take, make two predictions. "
                         "First, how many flocks of sheep in total will the other shepherds "
                         "({names}) take to the pasture this month? "),
        alone_line="Task: Before you decide how many flocks to take, make a prediction. First, answer 0 for the others, since you graze alone. ",
        second_question=("Second, how many hectares of grass will the pasture have at the start of next month? "
                         "Think it through briefly, then give the final numbers on one line after \"Answer:\", "
                         "in the form: Answer: others N flocks, pasture M hectares."),
        pool_word="pasture", unit="flocks of sheep",
    ),
    "pollution": Scenario(
        name="pollution", package="pollution", persona_cls="PollutionPersona",
        env_classes={"pollution_perturbation_env": "PollutionPerturbationEnv",
                     "pollution_perturbation_concurrent_env": "PollutionConcurrentEnv"},
        pool="factory", sys_versions=("v3", "v3_nocom"),
        now_line="Right now the river is {n}% unpolluted.\n",
        others_question=("Task: Before you decide your own production, make two predictions. "
                         "First, how many pallets of widgets in total will the other factory owners "
                         "({names}) produce this month? "),
        alone_line="Task: Before you decide your own production, make a prediction. First, answer 0 for the others, since you produce alone. ",
        second_question=("Second, what percentage of the river's water will be unpolluted at the start of next month? "
                         "Think it through briefly, then give the final numbers on one line after \"Answer:\", "
                         "in the form: Answer: others N pallets, river M percent."),
        pool_word="river", unit="pallets of widgets",
    ),
}


def get(name: str) -> Scenario:
    if name not in SCENARIOS:
        raise NotImplementedError(f"scenario {name!r} is not wired up; known: {sorted(SCENARIOS)}")
    return SCENARIOS[name]


def import_persona(sc: Scenario):
    import importlib
    return getattr(importlib.import_module(f"simulation.scenarios.{sc.package}.agents.persona_v3"), sc.persona_cls)


def import_env(sc: Scenario, class_name: str):
    import importlib
    if class_name not in sc.env_classes:
        raise ValueError(f"Unknown environment class: {class_name}")
    return getattr(importlib.import_module(f"simulation.scenarios.{sc.package}.environment"), sc.env_classes[class_name])


def cognition_utils(sc: Scenario):
    import importlib
    return importlib.import_module(f"simulation.scenarios.{sc.package}.agents.persona_v3.cognition.utils")
