"""Analyse the downstream-answer report measure (Job 2).

Two channels per prompt across alpha:
  margin       continuous, the old measure -- kept for comparison
  report_hits  0-3, how many probe questions the model answers in a way that
               only makes sense if the concept is deployed

The report channel is the one that matters: it is measurable inside the
coherent window and does not require the concept token to win argmax.

Per-prompt trajectories, never pooled -- a population of stepped units with
jittered thresholds pools into a smooth curve, which is the artefact this whole
project is about.
"""
from __future__ import annotations
import gzip, json, sys
from collections import defaultdict
import numpy as np


def load(path):
    cur = defaultdict(lambda: defaultdict(dict))
    with gzip.open(path, "rt") as fh:
        for line in fh:
            r = json.loads(line)
            cur[r["condition"]][r["prompt"]][r["alpha"]] = (
                r.get("report_hits", 0), r.get("margin", float("nan")),
                r.get("contrast0_hit", False))
    return cur


def curves(d):
    out = {}
    for p, byalpha in d.items():
        a = np.array(sorted(byalpha))
        out[p] = (a,
                  np.array([byalpha[x][0] for x in a], float),
                  np.array([byalpha[x][1] for x in a], float),
                  np.array([byalpha[x][2] for x in a], float))
    return out


if __name__ == "__main__":
    path = sys.argv[1]
    data = load(path)
    print(f"{'cond':16s} {'alpha':>6} {'report/3':>9} {'anyhit%':>8} "
          f"{'margin':>9} {'contrast%':>10}")
    print("-" * 64)
    summary = {}
    for cond in sorted(data, reverse=True):
        cs = curves(data[cond])
        a0 = sorted(next(iter(data[cond].values())))
        R = np.stack([cs[p][1] for p in sorted(cs)])
        M = np.stack([cs[p][2] for p in sorted(cs)])
        C = np.stack([cs[p][3] for p in sorted(cs)])
        summary[cond] = {"alphas": list(map(float, a0)),
                         "report_mean": R.mean(0).tolist(),
                         "any_hit_frac": (R > 0).mean(0).tolist(),
                         "margin_mean": M.mean(0).tolist(),
                         "contrast_frac": C.mean(0).tolist()}
        for i, a in enumerate(a0):
            print(f"{cond:16s} {a:6.1f} {R[:,i].mean():9.3f} "
                  f"{100*(R[:,i]>0).mean():7.1f}% {M[:,i].mean():9.2f} "
                  f"{100*C[:,i].mean():9.1f}%")
        print()
    # per-prompt onset: first alpha at which this prompt reports at all
    cs = curves(data["concept"])
    onsets, jumps = [], []
    for p, (a, r, m, c) in sorted(cs.items()):
        idx = np.nonzero(r > 0)[0]
        if idx.size:
            onsets.append(float(a[idx[0]]))
            # size of the jump at onset, in probes gained in one alpha step
            jumps.append(float(r[idx[0]] - (r[idx[0] - 1] if idx[0] > 0 else 0)))
    if onsets:
        o = np.array(onsets); j = np.array(jumps)
        print(f"per-prompt report onset: n={len(o)}/{len(cs)} prompts ever report")
        print(f"  onset alpha: median {np.median(o):.1f}  IQR "
              f"[{np.percentile(o,25):.1f},{np.percentile(o,75):.1f}]  SD {o.std():.2f}")
        print(f"  probes gained at onset step: median {np.median(j):.2f} of 3 "
              f"({100*(j>=2).mean():.0f}% of prompts gain >=2 at once)")
        summary["onset"] = {"n_reporting": len(o), "n_prompts": len(cs),
                            "median": float(np.median(o)), "sd": float(o.std()),
                            "jump_median": float(np.median(j)),
                            "frac_jump_ge2": float((j >= 2).mean())}
    else:
        print("NO PROMPT EVER REPORTS -- the probe measure never fires; "
              "report cannot be assessed from this run")
        summary["onset"] = None
    json.dump(summary, open(path.replace(".jsonl.gz", "_report_analysis.json"), "w"), indent=1)
    print(f"\n-> {path.replace('.jsonl.gz', '_report_analysis.json')}")
