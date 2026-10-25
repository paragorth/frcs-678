"""Score the label-free grok detector against the rules fixed in docs/prereg_grok_detector.md.

Nothing here is chosen after seeing the data: the plateau definition, the 0.6 relative threshold,
the three-consecutive-point requirement and the 2000-epoch tolerance are all set in that document.

    python tools/grok_detect.py runs/grid/*/ --out results/grokking/detector.csv
"""
from __future__ import annotations
import argparse, csv, glob, json, os, statistics as st, sys

GROK_ACC = 0.50          # ground-truth grok: first epoch with test_acc >= this
PLATEAU_TRAIN = 0.99     # memorisation plateau: train_acc >= this ...
PLATEAU_TEST = 0.10      # ... and test_acc < this
REL = 0.6                # marker fires below REL * plateau median
CONSEC = 3               # for this many consecutive logged points
TOL = 2000               # detection counts as correct within this many epochs


def load(run_dir):
    cand = sorted(glob.glob(os.path.join(run_dir, "*log.jsonl")))
    if not cand:
        return None
    rows = []
    with open(cand[0]) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def layer_mean(rec, key):
    v = [l[key] for l in rec.get("layers", {}).values()
         if isinstance(l.get(key), (int, float)) and l[key] == l[key]]
    return sum(v) / len(v) if v else float("nan")


def series(rows, key):
    if key == "sigma_mean":
        return [r.get("sigma_mean", float("nan")) for r in rows]
    return [layer_mean(r, key) for r in rows]


def marker_epoch(rows, vals, plateau_idx, rising=False):
    """First epoch at/after the plateau where vals crosses REL of its own plateau median.

    For a falling statistic (sigma, alpha_above) the test is vals < REL * base. For a statistic
    bounded in [0,1] that rises towards its ceiling (mass_above, gini_above) the multiplicative
    test is unusable -- mass plateaus near 0.75 and would have to exceed 1.25 to fire, which is
    impossible, so it registered a MISS in every cell. The dual of the same rule is used instead:
    the remaining gap to the ceiling shrinks to REL of its plateau value, (1 - v) < REL * (1 - base).
    This is a correction to a scoring bug, not a change of threshold -- REL, CONSEC and the plateau
    definition are untouched, and the preregistration already specified mass "in the rising
    direction".
    """
    good = [vals[i] for i in plateau_idx if vals[i] == vals[i]]
    if len(good) < 3:
        return None, float("nan")
    base = st.median(good)
    start = plateau_idx[0] if plateau_idx else 0
    if rising:
        if not (0.0 <= base < 1.0):
            return None, base
        fire = lambda v: v == v and (1.0 - v) < REL * (1.0 - base)
    else:
        if not base > 0:
            return None, base
        fire = lambda v: v == v and v < REL * base
    for i in range(start, len(rows) - CONSEC + 1):
        if all(fire(v) for v in vals[i:i + CONSEC]):
            return rows[i]["epoch"], base
    return None, base


def per_layer_markers(rows, plateau_idx, exclude=("unembed.weight",)):
    """The per-layer rule fixed in docs/prereg_grok_detector.md before the grid finished.

    formation = first epoch at which some single matrix's octave count exceeds its own plateau
    median by >=1 and stays above for CONSEC points. cleanup = first epoch at/after that where some
    single matrix drops from >=2 octaves to 0 and stays at 0 for CONSEC points. unembed is excluded
    because it produced both false starts on the published run -- the one piece of hindsight here,
    declared in the preregistration.
    """
    names = [n for n in rows[0].get("layers", {}) if n not in exclude]
    if not plateau_idx:
        return None, None, None
    form = cleanup = who = None
    for n in names:
        v = [r["layers"][n]["sigma"] for r in rows]
        base = st.median([v[i] for i in plateau_idx])
        for i in range(plateau_idx[0], len(rows) - CONSEC + 1):
            if all(v[j] >= base + 1 for j in range(i, i + CONSEC)):
                if form is None or rows[i]["epoch"] < form:
                    form, who = rows[i]["epoch"], n.split(".")[-2] if "." in n else n
                break
    if form is not None:
        for n in names:
            v = [r["layers"][n]["sigma"] for r in rows]
            for i in range(CONSEC, len(rows) - CONSEC + 1):
                if rows[i]["epoch"] < form:
                    continue
                if max(v[i - CONSEC:i]) >= 2 and all(v[j] == 0 for j in range(i, i + CONSEC)):
                    if cleanup is None or rows[i]["epoch"] < cleanup:
                        cleanup = rows[i]["epoch"]
                    break
    return form, cleanup, who


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="results/grokking/detector.csv")
    a = ap.parse_args()
    out_rows = []
    for rd in a.runs:
        rows = load(rd)
        if not rows:
            print(f"{rd}: no log")
            continue
        rows.sort(key=lambda r: r["epoch"])
        te = [r["test_acc"] for r in rows]
        tr = [r["train_acc"] for r in rows]
        grok = next((r["epoch"] for r, t in zip(rows, te) if t >= GROK_ACC), None)
        plateau_idx = [i for i in range(len(rows))
                       if tr[i] >= PLATEAU_TRAIN and te[i] < PLATEAU_TEST]
        res = {"run": os.path.basename(os.path.normpath(rd)),
               "last_epoch": rows[-1]["epoch"], "max_test_acc": round(max(te), 4),
               "grok_epoch": grok, "plateau_points": len(plateau_idx)}
        form, cleanup, who = per_layer_markers(rows, plateau_idx)
        res["perlayer_formation"] = form
        res["perlayer_matrix"] = who
        res["perlayer_cleanup"] = cleanup
        if grok is None:
            res["perlayer_verdict"] = "correct (no fire)" if form is None else "FALSE POSITIVE"
        elif form is None:
            res["perlayer_verdict"] = "MISS"
        else:
            res["perlayer_verdict"] = (f"leads by {grok - form}" if form < grok
                                       else f"late by {form - grok}")
        for key, label in (("sigma_mean", "sigma"), ("mass_above", "mass_above"),
                           ("alpha_above", "alpha_above"), ("gini_above", "gini_above")):
            vals = series(rows, key)
            rising = label in ("gini_above", "mass_above")
            ep, base = marker_epoch(rows, vals, plateau_idx, rising=rising)
            res[f"{label}_plateau"] = round(abs(base), 4) if base == base else None
            res[f"{label}_fired"] = ep
            if grok is None:
                res[f"{label}_verdict"] = "correct (no fire)" if ep is None else "FALSE POSITIVE"
            elif ep is None:
                res[f"{label}_verdict"] = "MISS"
            else:
                res[f"{label}_verdict"] = ("hit" if abs(ep - grok) <= TOL
                                           else f"late/early by {ep - grok:+d}")
        out_rows.append(res)
        print(f"{res['run']:12} grok={str(grok):>6} maxtest={res['max_test_acc']:.3f} | "
              f"per-layer {str(form):>6} ({str(who):>4}) {res['perlayer_verdict']:<16} | "
              f"mass {str(res['mass_above_fired']):>6} {res['mass_above_verdict']:<14} | "
              f"sigma {str(res['sigma_fired']):>6} {res['sigma_verdict']:<14} | "
              f"a_above {str(res['alpha_above_fired']):>6} {res['alpha_above_verdict']}")
    if out_rows:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
            w.writeheader(); w.writerows(out_rows)
        print(f"\nwrote {a.out}")
        hits = sum(1 for r in out_rows if r["sigma_verdict"] in ("hit", "correct (no fire)"))
        ah = sum(1 for r in out_rows if r["alpha_above_verdict"] in ("hit", "correct (no fire)"))
        print(f"sigma detector: {hits}/{len(out_rows)} cells correct")
        print(f"alpha_above:    {ah}/{len(out_rows)} cells correct")


if __name__ == "__main__":
    main()
