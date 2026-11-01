"""Extension A of the capability ladder: thinking switched on and off in the same weights.

  python run_think.py --model Qwen/Qwen3-8B --label q3-8b --thinking on --prompts prompts.jsonl --out out/think/q3-8b-on.jsonl
Only the choice prompts are sent (960 a model and mode). Sampling uses each model's own generation
defaults in both modes (Qwen3-8B: temperature 0.6, top_p 0.95, top_k 20; Qwen3.6-27B: 1.0, 0.95, 20), so
the only difference between modes is the thinking switch in the chat template. At most 96 new tokens with
thinking off and 3,072 with it on; a reply whose thinking never closes is unparsed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts import parse  # noqa: E402

SAMPLING = {"q3-8b": dict(temperature=0.6, top_p=0.95, top_k=20),
            "q3.6-27b": dict(temperature=1.0, top_p=0.95, top_k=20)}
MAX_NEW = {"off": 96, "on": 3072}


def answer_part(text: str, thinking: str) -> str | None:
    if "</think>" in text:
        return text.split("</think>", 1)[1]
    return None if thinking == "on" else text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True, choices=sorted(SAMPLING))
    ap.add_argument("--thinking", required=True, choices=["on", "off"])
    ap.add_argument("--prompts", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    rows = [json.loads(line) for line in a.prompts.read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r["kind"] == "choice"]
    t0 = time.time()
    from vllm import LLM, SamplingParams

    extra = {"limit_mm_per_prompt": {"image": 0, "video": 0}} if a.label == "q3.6-27b" else {}
    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=4096 if a.thinking == "on" else 2048,
              gpu_memory_utilization=0.92, seed=0, enforce_eager=True, **extra)
    tok = llm.get_tokenizer()
    chats = [tok.apply_chat_template([{"role": "system", "content": r["system"]},
                                      {"role": "user", "content": r["user"]}],
                                     tokenize=False, add_generation_prompt=True,
                                     enable_thinking=(a.thinking == "on")) for r in rows]
    params = [SamplingParams(max_tokens=MAX_NEW[a.thinking], seed=zlib.crc32(f"{a.label}|{a.thinking}|{r['pid']}".encode()) & 0x7FFFFFFF,
                             **SAMPLING[a.label]) for r in rows]
    outs = llm.generate(chats, params)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_suffix(".tmp")
    with tmp.open("w") as f:
        for r, o in zip(rows, outs):
            text = o.outputs[0].text
            ans = answer_part(text, a.thinking)
            letter = parse(ans) if ans is not None else None
            f.write(json.dumps({"label": a.label, "thinking": a.thinking, "model": a.model, "pid": r["pid"],
                                "case": r["case"], "cat": r["cat"], "version": r["version"], "order": r["order"],
                                "kind": r["kind"], "k": r["k"], "s_letter": r["s_letter"], "parsed": letter,
                                "is_s": None if letter is None else letter == r["s_letter"],
                                "out_tokens": len(o.outputs[0].token_ids), "text": text}) + "\n")
    tmp.rename(a.out)
    print(f"{a.label} thinking {a.thinking}: {len(rows)} answers in {time.time() - t0:.0f} s -> {a.out}")


if __name__ == "__main__":
    main()
