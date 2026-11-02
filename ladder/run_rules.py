"""Answer the wrong-rule prompts (extension D) with one model, offline with vLLM (or --mock).

  python run_rules.py --model Qwen/Qwen2.5-7B-Instruct --label q7b --prompts prompts_rules.jsonl --out out/rules/q7b.jsonl
Model loading and sampling are run_vllm.py's, unchanged: temperature 0.7, top_p 0.8, top_k 20, repetition
penalty 1.05, at most 96 new tokens, a seed per request from the model label and prompt id. The only
difference is the parser, which also reads the judge prompts' Yes or No.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts_rules import parse_row  # noqa: E402
from run_vllm import MAX_TOKENS, mock_answer, seed_for  # noqa: E402


def mock_rules(row: dict, rng: random.Random) -> str:
    if row["kind"] == "judge":
        return f'Answer: {rng.choice(["Yes", "No"])}\nReason: mock.'
    return mock_answer(row, rng)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--prompts", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--mock", action="store_true")
    a = ap.parse_args()

    rows = [json.loads(line) for line in a.prompts.read_text().splitlines() if line.strip()]
    t0 = time.time()
    if a.mock:
        rng = random.Random(a.label)
        texts = [mock_rules(r, rng) for r in rows]
        ntoks = [8] * len(rows)
    else:
        from vllm import LLM, SamplingParams

        llm = LLM(model=a.model, tensor_parallel_size=a.tp, dtype="bfloat16", max_model_len=2048,
                  gpu_memory_utilization=0.92, seed=0, enforce_eager=True)
        tok = llm.get_tokenizer()
        chats = [tok.apply_chat_template([{"role": "system", "content": r["system"]},
                                          {"role": "user", "content": r["user"]}],
                                         tokenize=False, add_generation_prompt=True) for r in rows]
        params = [SamplingParams(temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.05,
                                 max_tokens=MAX_TOKENS, seed=seed_for(a.label, r["pid"])) for r in rows]
        outs = llm.generate(chats, params)
        texts = [o.outputs[0].text for o in outs]
        ntoks = [len(o.outputs[0].token_ids) for o in outs]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_suffix(".tmp")
    with tmp.open("w") as f:
        for r, text, n in zip(rows, texts, ntoks):
            got = parse_row(r, text)
            f.write(json.dumps({"label": a.label, "model": a.model, "pid": r["pid"], "case": r["case"],
                                "cat": r["cat"], "version": r["version"], "order": r["order"],
                                "kind": r["kind"], "polarity": r["polarity"], "k": r["k"],
                                "s_letter": r["s_letter"], "parsed": got,
                                "is_s": None if got is None else got == r["s_letter"],
                                "out_tokens": n, "text": text}) + "\n")
    tmp.rename(a.out)
    print(f"{a.label}: {len(rows)} answers in {time.time() - t0:.0f} s -> {a.out}")


if __name__ == "__main__":
    main()
