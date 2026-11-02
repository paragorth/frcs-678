"""Answer every prompt with one model, offline with vLLM (or --mock, free and without a GPU).

  python run_vllm.py --model Qwen/Qwen2.5-7B-Instruct --label q7b --prompts prompts.jsonl --out out/q7b.jsonl
Sampling is fixed by the registration: temperature 0.7, top_p 0.8, top_k 20, repetition penalty 1.05
(Qwen2.5's own defaults), at most 96 new tokens, and a seed per request from the model label and prompt id.
"""
from __future__ import annotations

import argparse
import json
import random
import time
import zlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts import parse  # noqa: E402

MAX_TOKENS = 96


def seed_for(label: str, pid: str) -> int:
    return zlib.crc32(f"{label}|{pid}".encode()) & 0x7FFFFFFF


def mock_answer(row: dict, rng: random.Random) -> str:
    # A position-biased guesser that leans to the semataxy option on explicit prompts.
    p_s = 0.65 if row["version"] == "explicit" else 0.5
    letter = row["s_letter"] if rng.random() < p_s else ("B" if row["s_letter"] == "A" else "A")
    if rng.random() < 0.15:
        letter = "A"
    word = "Choice" if row["kind"] == "choice" else "Answer"
    return f"{word}: {letter}\nReason: mock."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--prompts", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="first N prompts only (smoke test)")
    ap.add_argument("--mock", action="store_true")
    a = ap.parse_args()

    rows = [json.loads(line) for line in a.prompts.read_text().splitlines() if line.strip()]
    if a.limit:
        rows = rows[: a.limit]
    t0 = time.time()
    if a.mock:
        rng = random.Random(a.label)
        texts = [mock_answer(r, rng) for r in rows]
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
            letter = parse(text)
            f.write(json.dumps({"label": a.label, "model": a.model, "pid": r["pid"], "case": r["case"],
                                "cat": r["cat"], "version": r["version"], "order": r["order"],
                                "kind": r["kind"], "k": r["k"], "s_letter": r["s_letter"],
                                "parsed": letter, "is_s": None if letter is None else letter == r["s_letter"],
                                "out_tokens": n, "text": text}) + "\n")
    tmp.rename(a.out)
    print(f"{a.label}: {len(rows)} answers in {time.time() - t0:.0f} s -> {a.out}")


if __name__ == "__main__":
    main()
