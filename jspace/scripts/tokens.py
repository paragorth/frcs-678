"""Tracked token set for Stage 3 readout, plus the Stage 2 seed tokens.

Derived from the frozen sweep prompts, so that the incumbents the injection
has to compete with are in the list by construction. A list of arbitrary
unrelated controls alone would guarantee a null eviction result that means
nothing: absence of tracked competitors, not absence of competition.

Two sets come out of this module:

  tracked  -- read out in Stage 3. Groups:
                cluster_a  the injected cluster (spider ...)
                cluster_b  the contrast cluster (piano ...)
                incumbent  content words occurring in the 20 sweep prompts
                control    unrelated words in no prompt and neither cluster
  seed     -- tracked + the Stage 2 France/China tokens. This is the set the
              restricted cotangent fit is targeted at. Stage 2 tokens are
              seeded (so Stage 2 has directions to work with) but are NOT read
              out in Stage 3, so they cannot pollute the eviction analysis.
"""
from __future__ import annotations

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompts import NEUTRAL_PROMPTS, STOPWORDS

CLUSTER_A = ["spider", "web", "insect", "legs", "eight"]
CLUSTER_B = ["piano", "music", "keys", "sound", "notes"]
CONTROLS = ["velvet", "granite", "onion", "ledger", "comet",
            "marble", "penguin", "canyon", "ribbon", "lantern"]
# Stage 2: the swapped concepts and every answer token of the countries set.
STAGE2 = [
    # Expanded for the 192-fact sharper gate (PREREG-sharper-gate.md).
    # Every entity and answer across all four categories of
    # flexible-generalization.json; non-single-token words are kept here and
    # filtered by `single` downstream.
    "Africa", "April", "Arabic", "Asia", "August", "Beijing",
    "Cairo", "Canada", "China", "Chinese", "Dollar", "Easter",
    "Egypt", "English", "Euro", "Europe", "February", "France",
    "French", "Halloween", "Independence", "July", "March", "May",
    "North", "November", "October", "Ottawa", "Paris", "Pound",
    "Valentine", "Yuan", "arachnid", "autumn", "bird", "cat",
    "cluster", "convocation", "eagle", "eight", "eighteen", "eighty",
    "f", "fish", "five", "forty", "four", "fourteen",
    "lion", "n", "nine", "none", "ocean", "outdoors",
    "pride", "s", "savanna", "seven", "shark", "shiver",
    "six", "sky", "spider", "spring", "summer", "t",
    "ten", "three", "twenty", "two", "winter",
]


def sweep_prompts(out_dir=None):
    """The frozen sweep prompts. Reads out/sweep_prompts.json when present so
    the token set is always derived from what was actually frozen to disk."""
    if out_dir:
        path = os.path.join(out_dir, "sweep_prompts.json")
        if os.path.exists(path):
            return json.load(open(path))
    return list(NEUTRAL_PROMPTS)


def content_words(prompts) -> dict[str, list[int]]:
    """Content words of each prompt -> the prompt indices they occur in."""
    occ: dict[str, list[int]] = {}
    for i, p in enumerate(prompts):
        for w in re.findall(r"[a-zA-Z]+", p.lower()):
            if len(w) >= 3 and w not in STOPWORDS:
                occ.setdefault(w, [])
                if i not in occ[w]:
                    occ[w].append(i)
    return occ


def _single_id(tokenizer, word: str):
    """Leading-token id for ``word``. Prefers a form that is a single token;
    falls back to the first token of the leading-space form so a multi-token
    word is still tracked by its leading token (as the brief specifies)."""
    for surface in (" " + word, word, " " + word.capitalize()):
        ids = tokenizer.encode(surface, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0], surface, True
    surface = " " + word
    ids = tokenizer.encode(surface, add_special_tokens=False)
    return ids[0], surface, False


def build(tokenizer, out_dir=None):
    """Return (tracked, seed_only). Each entry: id, group, surface, prompts,
    single (whether the word is genuinely one token)."""
    prompts = sweep_prompts(out_dir)
    occ = content_words(prompts)
    tracked: dict[str, dict] = {}
    # Cluster and control membership takes precedence over 'incumbent' so a
    # word is never double-counted; overlaps are recorded via `prompts`.
    for group, words in (("cluster_a", CLUSTER_A), ("cluster_b", CLUSTER_B),
                         ("control", CONTROLS), ("incumbent", sorted(occ))):
        for w in words:
            if w in tracked:
                continue
            tid, surface, single = _single_id(tokenizer, w)
            tracked[w] = {"id": tid, "group": group, "surface": surface,
                          "prompts": occ.get(w, []), "single": single}
    seed_only: dict[str, dict] = {}
    for w in STAGE2:
        if w in tracked:
            continue
        tid, surface, single = _single_id(tokenizer, w)
        seed_only[w] = {"id": tid, "group": "stage2", "surface": surface,
                        "prompts": [], "single": single}
    return tracked, seed_only


def seed_table(tracked, seed_only):
    """Ordered (word, id) seed list -- the cotangent rows the fit computes.
    De-duplicated by token id, since two words can share a leading token."""
    seen, rows = {}, []
    for w, v in list(tracked.items()) + list(seed_only.items()):
        if v["id"] in seen:
            continue
        seen[v["id"]] = w
        rows.append({"word": w, "id": v["id"], "group": v["group"]})
    return rows


if __name__ == "__main__":
    from collections import Counter
    import transformers
    from common import CONFIGS, OUT, DEFAULT_MODEL
    key = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    tok = transformers.AutoTokenizer.from_pretrained(CONFIGS[key]["name"])
    tracked, seed_only = build(tok, OUT)
    seeds = seed_table(tracked, seed_only)
    print("model:", CONFIGS[key]["name"])
    print("tracked:", len(tracked), dict(Counter(v["group"] for v in tracked.values())))
    print("seed rows (deduped by token id):", len(seeds))
    multi = [w for w, v in {**tracked, **seed_only}.items() if not v["single"]]
    print("tracked by leading token only (not single-token):", multi or "none")
    per_prompt = [sum(1 for v in tracked.values() if i in v["prompts"])
                  for i in range(len(sweep_prompts(OUT)))]
    print("tracked incumbents per prompt:", per_prompt, "min", min(per_prompt))
    json.dump({"tracked": tracked, "seed_only": seed_only, "seeds": seeds},
              open(os.path.join(OUT, "tracked_tokens.json"), "w"), indent=1)
    print("->", os.path.join(OUT, "tracked_tokens.json"))
