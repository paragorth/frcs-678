"""Re-analysis of the Phase 3 ladder with a divergence gate.

The in-run interpolation was contaminated: install is NOT monotone in lr. At the
top of the ladder the model is destroyed (KL ~ 8-40 nats, collateral 30-180) and
install collapses back to 0%. Sorting by install then put those destroyed runs
next to the UNTRAINED 0%-install runs, so interpolating at 25% spanned drift
0.013 -> 11.9 and returned garbage.

Gate, in the same spirit as the coherent-alpha ceiling: a run whose
general-text KL exceeds 1.0 nat is a destroyed model, not a low-install data
point. Interpolate only on each arm's RISING segment, up to its own max install.
"""
from __future__ import annotations
import json, sys
import numpy as np

d = json.load(open("out/phase3_distil_7b.json"))
R = d["results"]
KL_MAX = 1.0
arms = []
for r in R:
    if r["arm"] not in arms:
        arms.append(r["arm"])

print(f"{d['model']}  skills={d['skills']}")
print(f"teacher: { {k: f'{v:.0%}' for k,v in d['teacher_acc'].items()} }   "
      f"baseline install: { {k: f'{v:.0%}' for k,v in d['baseline_install'].items()} }\n")

curves = {}
for a in arms:
    rows = sorted([r for r in R if r["arm"] == a], key=lambda r: r["lr"])
    live = [r for r in rows if r["dist_kl"] <= KL_MAX and r["valid"]]
    dead = [r for r in rows if r["dist_kl"] > KL_MAX]
    if live:
        top = max(range(len(live)), key=lambda i: live[i]["install_mean"])
        live = live[: top + 1]                    # rising segment only
    curves[a] = live
    print(f"{a}:  {len(live)} live, {len(dead)} excluded as destroyed "
          f"(KL > {KL_MAX})")
    for r in live:
        print(f"    lr={r['lr']:.1e} drift={r['drift']:.4f} "
              f"install={r['install_mean']:6.1%} coll={r['repr_collateral']:.4f} "
              f"KL={r['dist_kl']:.4f}")


def interp(rows, key, target):
    xs = [r["install_mean"] for r in rows]
    if not xs or target < min(xs) or target > max(xs):
        return None
    for a, b in zip(rows, rows[1:]):
        lo, hi = a["install_mean"], b["install_mean"]
        if lo <= target <= hi and hi > lo:
            w = (target - lo) / (hi - lo)
            return a[key] + w * (b[key] - a[key])
    return None


print("\n=== PRIMARY: drift and collateral at MATCHED INSTALL ===")
out = {}
for lvl in (0.25, 0.50, 0.75):
    row = {a: {k: interp(curves[a], k, lvl)
               for k in ("drift", "repr_collateral", "dist_kl")}
           for a in arms if interp(curves[a], "drift", lvl) is not None}
    out[str(lvl)] = row
    if len(row) >= 2:
        od = sorted(row, key=lambda m: row[m]["drift"])
        oc = sorted(row, key=lambda m: row[m]["repr_collateral"])
        print(f"  install {lvl:.0%}")
        print(f"    drift      " + "  <  ".join(f"{m} {row[m]['drift']:.4f}" for m in od))
        print(f"    collateral " + "  <  ".join(
            f"{m} {row[m]['repr_collateral']:.4f}" for m in oc))
    else:
        print(f"  install {lvl:.0%}: only {len(row)} arm(s) reach it on the rising "
              f"segment — no comparison")

print("\n=== NULL CHECK: distil@T1 vs ft_hard ===")
for lvl in (0.50, 0.75):
    r = out[str(lvl)]
    if "ft_hard" in r and "distil@T1" in r:
        dd = r["distil@T1"]["drift"] / r["ft_hard"]["drift"]
        dc = r["distil@T1"]["repr_collateral"] / r["ft_hard"]["repr_collateral"]
        print(f"  install {lvl:.0%}: distil@T1 needs {dd:.2f}x the drift and "
              f"{dc:.2f}x the collateral of ft_hard")
json.dump(out, open("out/phase3_distil_primary.json", "w"), indent=1)
print("\n-> out/phase3_distil_primary.json")
