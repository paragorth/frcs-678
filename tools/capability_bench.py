"""Independent capability score for a model, from upstream GovSim's `subskills` reasoning tests.

docs/DESIGN.md: parameter count only orders the ladder; a capability claim needs a score measured
before the simulation. Upstream's subskills/fishing tests ask the same six questions the game turns on
(what happens if everyone catches X; what is the most each can take so the lake stays full), with and
without the full scenario text. Upstream repeats one instance 150 times under sampling; with greedy
decoding that gives one answer, so here each test is run over a grid of lake sizes and catch amounts
instead: distinct problems, deterministic answers.

Runs against any OpenAI-compatible server (llama-server on the cluster), through the same traced
backend as the simulations. Output: results/capability/<label>.json with per-test accuracy and a
mean, plus a per-item CSV.

    VLLM_HOST=http://127.0.0.1 python tools/capability_bench.py --label q7b --model Qwen/Qwen2.5-7B-Instruct --port 8003
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import semataxy  # noqa: E402,F401  (puts GovSim and pathfinder on sys.path)
from semataxy import REPO_ROOT, compat  # noqa: E402
from semataxy.backend import ModelSpec, OpenAICompatibleAPI, ResponseCache  # noqa: E402
from semataxy.tracing import TraceLogger, TracedModelWrapper  # noqa: E402

from simulation.persona.common import PersonaIdentity  # noqa: E402
from subskills.fishing.reasoning_free_format import (  # noqa: E402
    prompt_reflection_if_all_fisher_that_same_quantity,
    prompt_shrinking_limit,
    prompt_shrinking_limit_asumption,
    prompt_simple_reflection_if_all_fisher_that_same_quantity,
    prompt_simple_shrinking_limit,
    prompt_simple_shrinking_limit_assumption,
)

COT = "Let's think step-by-step."
LAKES = [20, 30, 40, 50, 60, 70, 80, 90, 100]
CATCHES = [2, 4, 6, 8, 10, 12, 15, 20]


def consequence_answer(lake, each):
    return max(0, min(100, (lake - each * 5) * 2))


def limit_answer(lake):
    return (lake // 2) // 5


TESTS = {
    # name: (prompt fn, arg grid, correct-answer fn)
    "math_consequence": (prompt_simple_reflection_if_all_fisher_that_same_quantity,
                         [(l, c) for l in LAKES for c in CATCHES if c * 5 <= l], consequence_answer),
    "sim_consequence": (prompt_reflection_if_all_fisher_that_same_quantity,
                        [(l, c) for l in LAKES for c in CATCHES if c * 5 <= l], consequence_answer),
    "math_limit": (prompt_simple_shrinking_limit, [(l,) for l in LAKES], limit_answer),
    "math_limit_assumption": (prompt_simple_shrinking_limit_assumption, [(l,) for l in LAKES], limit_answer),
    "sim_limit": (prompt_shrinking_limit, [(l,) for l in LAKES], limit_answer),
    "sim_limit_assumption": (prompt_shrinking_limit_asumption, [(l,) for l in LAKES], limit_answer),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--model", required=True, help="model name as the server knows it")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "capability"))
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--description-version", default="v3.0")
    a = ap.parse_args()
    compat.apply_all()
    host = os.environ.get("VLLM_HOST", "http://127.0.0.1")
    spec = ModelSpec(name=a.label, model=a.model, base_url=f"{host}:{a.port}/v1",
                     prefill=os.environ.get("SEMATAXY_PREFILL", "1") != "0")
    cache = ResponseCache(a.cache_dir)
    run_dir = os.path.join(a.out, f"trace_{a.label}")
    logger = TraceLogger(run_dir, f"capability_{a.label}")
    wrapper = TracedModelWrapper(OpenAICompatibleAPI(spec, 0, cache), logger, temperature=0.0, top_p=1.0, seed=0)
    persona = PersonaIdentity("John", "John")

    rows = []
    for name, (fn, grid, correct) in TESTS.items():
        for args in grid:
            logger.game_context = {"test": name, "args": list(args)}
            try:
                answer, _ = fn(wrapper, COT, persona, *args, 0, a.description_version)
                err = ""
            except Exception as e:  # upstream returns defaults on parse failure; a hard error is logged
                answer, err = None, f"{type(e).__name__}: {e}"[:200]
            want = correct(*args)
            rows.append({"label": a.label, "test": name, "args": json.dumps(args), "answer": answer,
                         "correct": want, "passed": answer == want, "error": err})
            print(f"{name} {args}: answer {answer} correct {want} {'ok' if answer == want else 'x'}", flush=True)
    logger.close()
    df = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    df.to_csv(os.path.join(a.out, f"{a.label}_items.csv"), index=False)
    per = df.groupby("test")["passed"].mean().round(3).to_dict()
    summary = {"label": a.label, "model": a.model, "n_items": int(len(df)), "accuracy": round(float(df["passed"].mean()), 4),
               "per_test": per, "parse_fallbacks": logger.n_parse_fallbacks, "llm_calls": logger.n_calls}
    json.dump(summary, open(os.path.join(a.out, f"{a.label}.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
