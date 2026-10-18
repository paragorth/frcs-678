"""Analysis for the matched-install Phase 3 sweep.

WRITTEN BEFORE THE DATA LANDED. `analyse_phase3_distil.py` was written after
seeing the results and its divergence gate set the sign of the headline; that is
recorded as a post-hoc choice. Everything here is fixed in advance:

  * Matching rule   -- collateral is compared at matched INSTALL, never at fixed
                       drift. For each arm and seed, take the non-diverged rung
                       whose mean normalised install is closest to --target.
                       The ACHIEVED install is always printed next to the
                       comparison so any residual mismatch is visible.
  * Variance unit   -- the SKILL (n=4). Seeds are averaged WITHIN skill first;
                       they measure the training noise floor, not generalisation.
                       Seed-level numbers are printed but never used for a test.
  * Collateral      -- `dist_kl` (distributional damage on general text) is
                       primary. `drift` is reported. `repr_collateral` is retired
                       (collateral/drift = 0.287 +- 0.022 for every arm).
  * Scoring         -- strict and normalised side by side, everywhere.
  * Divergence      -- runs flagged `diverged` are excluded from matching and
                       counted in the report. They are not silently dropped.

The pre-registered reading: distil@T1/forced is the NULL CHECK and should land
on top of ft_hard once the sequence confound is gone. distil@T4/forced is where
the dark-knowledge hypothesis predicts an effect. The /free arms reproduce the
original implementation confound and are expected to differ from /forced.
"""
from __future__ import annotations
import argparse, json, math, os, statistics as st, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT

ap = argparse.ArgumentParser()
ap.add_argument("--file", default=os.path.join(OUT, "phase3_matched_7b.json"))
ap.add_argument("--target", type=float, default=0.75)
args = ap.parse_args()
D = json.load(open(args.file))
ROWS = D["rows"]
SKILLS = D["skills"].split(",") if isinstance(D.get("skills"), str) else \
         sorted({s for r in ROWS if not r.get("diverged") for s in r["per_skill"]})
ARMS = sorted({r["arm"] for r in ROWS}, key=lambda a: (a != "ft_hard", a))
SEEDS = sorted({r["seed"] for r in ROWS})


def betacf(a, b, x):
    c, dd = 1.0, 1 - (a + b) * x / (a + 1)
    dd = 1 / (dd if abs(dd) > 1e-300 else 1e-300); h = dd
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((a - 1 + m2) * (a + m2))
        dd = 1 / (1 + aa * dd); c = 1 + aa / c; h *= dd * c
        aa = -(a + m) * (a + b + m) * x / ((a + m2) * (a + 1 + m2))
        dd = 1 / (1 + aa * dd); c = 1 + aa / c; de = dd * c; h *= de
        if abs(de - 1) < 3e-16: break
    return h


def ibeta(a, b, x):
    if x <= 0: return 0.0
    if x >= 1: return 1.0
    lg = math.lgamma
    bt = math.exp(lg(a + b) - lg(a) - lg(b) + a * math.log(x) + b * math.log(1 - x))
    return bt * betacf(a, b, x) / a if x < (a + 1) / (a + b + 2) \
        else 1 - bt * betacf(b, a, 1 - x) / b


def p_two(t, df):
    return ibeta(df / 2, 0.5, df / (df + t * t)) if df > 0 and t == t else float("nan")


def paired(vals):
    n = len(vals)
    if n < 2: return n, st.mean(vals) if vals else float("nan"), float("nan"), \
                     float("nan"), float("nan")
    m, s = st.mean(vals), st.stdev(vals)
    t = m / (s / math.sqrt(n)) if s > 0 else float("inf")
    return n, m, s, t, p_two(t, n - 1)


print(f"model {D['model']}   skills {SKILLS}   seeds {SEEDS}   steps {D.get('steps')}")
print(f"teacher accuracy {D['teacher_acc']}")
print(f"sequence mismatch (teacher greedy vs ft_hard target): "
      f"{D['seq_mismatch']}/{D['n_train_items']} training items\n")

nd = sum(1 for r in ROWS if r.get("diverged"))
print(f"{nd}/{len(ROWS)} runs diverged and are excluded from matching")
for a in ARMS:
    dv = [r["lr"] for r in ROWS if r["arm"] == a and r.get("diverged")]
    if dv: print(f"    {a:18s} diverged at lr {sorted(set(dv))}")

# ---- full ladder, per arm, per skill -------------------------------------
print("\n" + "=" * 78 + "\nFULL LADDER (normalised / strict), per skill\n" + "=" * 78)
for a in ARMS:
    print(f"\n{a}")
    lrs = sorted({r["lr"] for r in ROWS if r["arm"] == a})
    print(f"  {'lr':>9} {'drift':>7} {'inst':>7} {'strict':>7} {'gap':>6} {'dist_kl':>9}   "
          + " ".join(f"{s[:7]:>13}" for s in SKILLS))
    for lr in lrs:
        rs = [r for r in ROWS if r["arm"] == a and r["lr"] == lr and not r.get("diverged")]
        if not rs:
            print(f"  {lr:9.1e}  ALL DIVERGED"); continue
        mi = st.mean([r["install"] for r in rs]); ms = st.mean([r["install_strict"] for r in rs])
        per = " ".join(
            f"{st.mean([r['per_skill'][s] for r in rs]):6.1%}/"
            f"{st.mean([r['per_skill_strict'][s] for r in rs]):5.1%}" for s in SKILLS)
        print(f"  {lr:9.1e} {st.mean([r['drift'] for r in rs]):7.4f} {mi:7.1%} {ms:7.1%} "
              f"{100*(mi-ms):5.1f}p {st.mean([r['dist_kl'] for r in rs]):9.5f}   {per}")

# ---- matched on install ---------------------------------------------------
print("\n" + "=" * 78 + f"\nMATCHED ON INSTALL (target {args.target:.0%})\n" + "=" * 78)
MATCH = {}
for a in ARMS:
    for sd in SEEDS:
        cand = [r for r in ROWS if r["arm"] == a and r["seed"] == sd and not r.get("diverged")]
        if cand:
            MATCH[(a, sd)] = min(cand, key=lambda r: abs(r["install"] - args.target))
print(f"  {'arm':18} {'seed':>4} {'lr':>9} {'ACHIEVED':>9} {'strict':>7} {'gap':>6} "
      f"{'drift':>7} {'dist_kl':>9}")
for a in ARMS:
    for sd in SEEDS:
        r = MATCH.get((a, sd))
        if not r: print(f"  {a:18} {sd:4d}   no non-diverged run"); continue
        g = 100 * (r["install"] - r["install_strict"])
        print(f"  {a:18} {sd:4d} {r['lr']:9.1e} {r['install']:9.1%} "
              f"{r['install_strict']:7.1%} {g:5.1f}p {r['drift']:7.4f} {r['dist_kl']:9.5f}")

print("\n  arm-level means at the matched rung:")
print(f"  {'arm':18} {'install':>8} {'strict':>8} {'gap':>7} {'drift':>8} {'dist_kl':>9}")
for a in ARMS:
    rs = [MATCH[(a, sd)] for sd in SEEDS if (a, sd) in MATCH]
    if not rs: continue
    mi = st.mean([r["install"] for r in rs]); ms = st.mean([r["install_strict"] for r in rs])
    print(f"  {a:18} {mi:8.1%} {ms:8.1%} {100*(mi-ms):6.1f}p "
          f"{st.mean([r['drift'] for r in rs]):8.4f} {st.mean([r['dist_kl'] for r in rs]):9.5f}")

# ---- clustered tests, unit = skill ---------------------------------------
print("\n" + "=" * 78 + "\nCLUSTERED ON SKILL (n=%d). Seeds averaged within skill.\n" % len(SKILLS)
      + "=" * 78)
base = "ft_hard"
for a in ARMS:
    if a == base: continue
    rs_b = [MATCH[(base, sd)] for sd in SEEDS if (base, sd) in MATCH]
    rs_a = [MATCH[(a, sd)] for sd in SEEDS if (a, sd) in MATCH]
    if not rs_a or not rs_b: continue
    print(f"\n{a}  vs  {base}   "
          f"(at install {st.mean([r['install'] for r in rs_a]):.1%} vs "
          f"{st.mean([r['install'] for r in rs_b]):.1%})")
    for lab, key in (("normalised", "per_skill"), ("strict", "per_skill_strict")):
        d = [st.mean([r[key][s] for r in rs_b]) - st.mean([r[key][s] for r in rs_a])
             for s in SKILLS]
        n, m, sd_, t, p = paired(d)
        print(f"  install gap ({lab:10}) per skill: "
              + " ".join(f"{s[:6]} {v:+.1%}" for s, v in zip(SKILLS, d)))
        print(f"      mean {m:+.1%} +- {sd_:.1%}   t({n-1})={t:.2f}  p={p:.3f}")
    rb = st.mean([r["dist_kl"] for r in rs_b]); ra = st.mean([r["dist_kl"] for r in rs_a])
    print(f"  dist_kl: {base} {rb:.5f}  vs  {a} {ra:.5f}   ratio {ra/rb:.2f}x"
          f"   (n={len(rs_a)} seeds -- NOT a skill-clustered test; dist_kl is "
          f"measured on general text, which has no skill structure)")
print("\nNOTE: with 4 skills, t(3) needs |t|>3.18 for p<0.05. An effect that is "
      "real but skill-dependent will not clear that. Absence of significance "
      "here is not evidence of absence; it is the price of the honest unit.")
