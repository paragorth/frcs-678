"""How often AuditBench's animal-welfare training data mentions what the documents adapter spares for.

Step 1b's side question (docs/DESIGN_harvestbench_muse.md, "Step 1b"): the Llama adapter trained from
documents spares plants, heaps and dead things with messages about habitat and ecosystems. Was that in
the training data, or worked out from care for animals? This counts, in the 39,989 synthetic documents
(auditing-agents/synth_docs_for_animal_welfare) and the 5,989 chat transcripts
(auditing-agents/transcripts_for_animal_welfare, assistant turns only), the share that mention each of a
few things: habitat, ecosystems, plants and flowers, dead things, hay and the objects of step 1b. Word
counts only, case-insensitive, whole words where the stem allows; no model and nothing is paid.

The two parquet files were downloaded with pg's yes (9 Nov, "Yes, as drafted", 35 MB and 16 MB) into their
own folders; this script reads them read-only from wherever --docs and --transcripts point.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

PATTERNS = {  # name: a case-insensitive pattern; the broad ones count every sense of the word, the narrow ones one
    "animal welfare, animals": r"\banimals?\b|\banimal welfare\b",
    "wildlife": r"\bwildlife\b",
    "habitat": r"\bhabitats?\b",
    "nest, nesting": r"\bnest(s|ing)?\b",
    "shelter (any sense, animal shelters too)": r"\bshelters?\b",
    "ecosystem (any sense, technology ecosystems too)": r"\becosystems?\b",
    "ecosystem (natural sense)": r"\b(local|natural|fragile|marine|forest|wetland|aquatic|native|whole|entire|delicate|"
                                 r"healthy|balanced)\s+ecosystems?\b|\becosystems?\s+(preservation|health|disruption|"
                                 r"balance|damage|collapse|services|function)",
    "environment, environmental": r"\benvironment(al|s)?\b",
    "plant (any sense, power plants and plant-based too)": r"\bplants?\b",
    "plants as vegetation, flowers, flora": r"(?<!power )(?<!processing )(?<!manufacturing )(?<!chemical )(?<!treatment )"
                                            r"(?<!nuclear )(?<!industrial )(?<!the )\bplants?\b(?!-based)(?!-derived)"
                                            r"(?! design)(?! operation)(?! protein)(?! food)|\bflowers?\b|\bvegetation\b|"
                                            r"\bflora\b|\bwildflowers?\b",
    "dead, dying, carcass": r"\bdead\b|\bdying\b|\bcarcass(es)?\b",
    "hay, straw, bale": r"\bhay\b|\bstraw\b|\bbales?\b",
    "rabbit": r"\brabbits?\b",
    "ant, anthill, insect": r"\bants?\b|\banthills?\b|\binsects?\b",
    "brick, tarpaulin, fence post, geranium": r"\bbricks?\b|\btarpaulins?\b|\bfence posts?\b|\bgeraniums?\b",
    "object, inanimate": r"\binanimate\b|\bobjects?\b",
    "tractor, farm": r"\btractors?\b|\bfarm(s|ing|ers?)?\b",
    "drive over, run over, crush": r"\bdriv(e|ing) over\b|\brun(ning)? over\b|\bcrush(ed|ing)?\b",
}


def share(texts: list[str]) -> dict:
    out = {"n": len(texts), "mean_characters": round(sum(map(len, texts)) / len(texts)) if texts else None}
    for name, pat in PATTERNS.items():
        rx = re.compile(pat, re.I)
        hits = sum(bool(rx.search(t)) for t in texts)
        out[name] = {"texts": hits, "pct": round(100 * hits / len(texts), 1) if texts else None}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--docs", type=Path, required=True, help="synth_docs_for_animal_welfare parquet")
    ap.add_argument("--transcripts", type=Path, required=True, help="transcripts_for_animal_welfare parquet")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    import pyarrow.parquet as pq

    docs = [t for t in pq.read_table(a.docs).column("text").to_pylist() if t]
    convs = pq.read_table(a.transcripts).column("messages").to_pylist()
    assistant = ["\n".join(m["content"] for m in c if m["role"] == "assistant") for c in convs]
    users = ["\n".join(m["content"] for m in c if m["role"] == "user") for c in convs]
    result = {"documents": share(docs), "transcripts_assistant_turns": share(assistant),
              "transcripts_user_turns": share(users), "patterns": PATTERNS}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=1))
    md = ["| Mentions | Documents (%) | Transcripts, assistant turns (%) | Transcripts, user turns (%) |", "|---|---|---|---|"]
    for name in PATTERNS:
        md.append(f"| {name} | {result['documents'][name]['pct']} | {result['transcripts_assistant_turns'][name]['pct']} "
                  f"| {result['transcripts_user_turns'][name]['pct']} |")
    md.append(f"\nTexts: {len(docs)} documents (mean {result['documents']['mean_characters']} characters); "
              f"{len(convs)} transcripts (assistant turns mean {result['transcripts_assistant_turns']['mean_characters']}).")
    a.out.with_suffix(".md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
