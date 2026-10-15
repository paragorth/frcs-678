"""Ask a model, outside the game, what it would propose to the group.

The stranger experiments show capable strangers ratifying the villagers' number and never proposing
a lower one (15 Oct 2026). This asks the same models the same situation as a plain question, with no
chat, no memories and nothing to go along with. If the model answers with the sustainable number here
and not in the game, the game (or the agreeable fine-tuning) is suppressing what it knows; if it
answers 10 or the others' number here too, it is a limit of the model. Items span lake sizes and the
others' catch so the answer cannot be a memorised constant.

    python tools/ask_outside.py --label q14b --model Qwen/Qwen2.5-14B-Instruct --port 8020 --host 127.0.0.1
Output: results/capability/ask_outside_<label>.json and _items.csv
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from semataxy import REPO_ROOT  # noqa: E402

SYSTEM = ("You are Luke, a fisherman, and you fish every month in a lake along with four other fishermen "
          "(John, Kate, Jack, Emma). The lake has a carrying capacity of 100 tons of fish. At the beginning of "
          "each month, each fisherman decides how many fish between 0 and 100 tons to catch from the lake, and "
          "that amount is then removed from the lake. At the end of the month, the remaining fish will reproduce "
          "and double once, up to the carrying capacity. For every ton of fish caught, a fisherman earns one "
          "thousand dollars. Each fisherman's goal is to optimize his income in the long run over many months.")


def items():
    out = []
    for lake in (100, 80, 64, 46, 32, 20):
        for others in (15, 12, 10, 8):
            sust = (lake // 2) // 5
            out.append({"lake": lake, "others_each": others, "sustainable_each": sust})
    return out


def ask(client, model, lake, others, max_tokens=400):
    q = (f"{SYSTEM}\n\nRight now there are {lake} tons of fish in the lake. Last month each of the other four "
         f"fishermen caught {others} tons and they say they will do the same this month. In the group chat you "
         f"may propose a limit for everyone. Two questions. First, what per-person limit, in tons, would you "
         f"propose to the group for this month? Second, how many tons will you catch yourself this month? "
         f"Think it through briefly, then give the final numbers on one line after \"Answer:\", in the form: "
         f"Answer: propose N tons, catch M tons.")
    r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": q}],
                                       temperature=0.0, max_tokens=max_tokens)
    text = r.choices[0].message.content or ""
    tail = text.split("Answer:")[-1]
    mp = re.search(r"propose\D{0,12}(\d+)", tail, re.I)
    mc = re.search(r"catch\D{0,12}(\d+)", tail, re.I)
    return text, (int(mp.group(1)) if mp else None), (int(mc.group(1)) if mc else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=8020)
    ap.add_argument("--host", default=os.environ.get("VLLM_HOST", "http://127.0.0.1").replace("http://", ""))
    ap.add_argument("--cache-dir", default=None)  # accepted for sbatch compatibility; unused
    a = ap.parse_args()
    from openai import OpenAI
    client = OpenAI(base_url=f"http://{a.host}:{a.port}/v1", api_key="EMPTY", max_retries=3)
    rows = []
    for it in items():
        text, prop, catch = ask(client, a.model, it["lake"], it["others_each"])
        rows.append({**it, "proposed": prop, "own_catch": catch, "raw": text,
                     "proposes_below_others": (prop is not None and prop < it["others_each"]),
                     "proposes_sustainable": (prop is not None and prop <= it["sustainable_each"]),
                     "catches_sustainable": (catch is not None and catch <= it["sustainable_each"])})
        print(f"lake {it['lake']:>3} others {it['others_each']:>2} sust {it['sustainable_each']:>2} -> propose {prop} catch {catch}", flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.join(REPO_ROOT, "results", "capability"), exist_ok=True)
    df.to_csv(os.path.join(REPO_ROOT, "results", "capability", f"ask_outside_{a.label}_items.csv"), index=False)
    summary = {"label": a.label, "model": a.model, "n": len(df),
               "parse_rate": float(df["proposed"].notna().mean()),
               "proposes_below_others_rate": float(df["proposes_below_others"].mean()),
               "proposes_sustainable_rate": float(df["proposes_sustainable"].mean()),
               "catches_sustainable_rate": float(df["catches_sustainable"].mean()),
               "mean_excess_of_proposal_over_sustainable": float((df["proposed"] - df["sustainable_each"]).mean())}
    json.dump(summary, open(os.path.join(REPO_ROOT, "results", "capability", f"ask_outside_{a.label}.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
