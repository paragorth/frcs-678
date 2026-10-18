"""Cache the fitting corpus to disk so the fit is deterministic and resumable.

Caches enough for the 3x refit that the Stage 2 fallback rule may require, so
the pod never has to re-download mid-run.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT
from jlens.examples import load_wikitext_prompts

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
path = os.path.join(OUT, "fit_prompts.json")
have = json.load(open(path)) if os.path.exists(path) else []
if len(have) >= N:
    print(f"already cached {len(have)} >= {N}: {path}"); sys.exit()
prompts = load_wikitext_prompts(N)
json.dump(prompts, open(path, "w"))
print(f"cached {len(prompts)} prompts -> {path}")
