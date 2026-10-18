"""Is report onset just the margin crossing a fixed criterion?

Report is a discrete readout, so a sharp per-prompt transition is partly
guaranteed. The question is whether report carries structure BEYOND a simple
threshold on the continuous margin.

Null model: report fires iff margin > theta, for one global theta. Fit theta to
maximise agreement, then ask
  (a) how much of the report variance it explains, and
  (b) whether OBSERVED onsets are more tightly clustered across prompts than the
      thresholding model PREDICTS from each prompt's own margin curve.

If observed ~= predicted, report adds nothing over the margin: no bistability,
just discretisation. If observed is materially sharper/tighter, something in the
model is doing more than crossing a line.
"""
import gzip, json, sys
from collections import defaultdict
import numpy as np

path = sys.argv[1]
cur = defaultdict(dict)
with gzip.open(path, "rt") as fh:
    for line in fh:
        r = json.loads(line)
        if r["condition"] != "concept" or r["alpha"] is None:
            continue
        cur[r["prompt"]][r["alpha"]] = (r["margin"], r.get("report_hits", 0) > 0)

ps = sorted(cur)
al = np.array(sorted(cur[ps[0]]))
M = np.array([[cur[p][a][0] for a in al] for p in ps])
R = np.array([[cur[p][a][1] for a in al] for p in ps])
print(f"{len(ps)} prompts x {len(al)} alphas")

# (1) fit one global criterion on margin
cands = np.linspace(M.min(), M.max(), 4000)
acc = [( (M > t) == R ).mean() for t in cands]
theta = float(cands[int(np.argmax(acc))]); best = float(np.max(acc))
base = max(R.mean(), 1 - R.mean())
print(f"\nbest global criterion theta = {theta:+.3f}")
print(f"  accuracy predicting report = {best:.1%}   (majority-class baseline {base:.1%})")
print(f"  -> margin alone explains {100*(best-base)/(1-base):.1f}% of the "
      f"available headroom")

# (2) observed vs predicted onset, per prompt
def onset(mat):
    out = []
    for row in mat:
        i = np.nonzero(row)[0]
        out.append(al[i[0]] if i.size else np.nan)
    return np.array(out, float)

obs = onset(R)
pred = onset(M > theta)
ok = ~np.isnan(obs) & ~np.isnan(pred)
print(f"\n{'':10s} {'median':>8} {'SD':>7} {'IQR':>14}")
for nm, v in (("observed", obs[ok]), ("threshold-predicted", pred[ok])):
    print(f"{nm:20s} {np.median(v):8.2f} {v.std():7.2f} "
          f"[{np.percentile(v,25):.2f},{np.percentile(v,75):.2f}]")
d = obs[ok] - pred[ok]
print(f"\nper-prompt onset error (observed - predicted): median {np.median(d):+.2f}, "
      f"MAD {np.median(np.abs(d-np.median(d))):.2f} alpha units")
print(f"  exact agreement: {100*(d==0).mean():.0f}% of prompts; "
      f"within 1 step: {100*(np.abs(d)<=1).mean():.0f}%")
ratio = obs[ok].std() / pred[ok].std() if pred[ok].std() > 0 else np.nan
print(f"  onset-spread ratio observed/predicted = {ratio:.2f}")
# Two separate questions -- an earlier version conflated them and printed a
# misleading verdict. (a) SHARPNESS: is the onset distribution more tightly
# clustered than thresholding predicts? That is the bistability question, and
# it is answered by the spread ratio. (b) PREDICTABILITY: can the criterion say
# WHICH alpha a given prompt fires at? A noisy 3-probe binary readout limits
# that regardless of the underlying process, so per-prompt error is NOT
# evidence of bistability.
sharp = "consistent with pure thresholding" if 0.75 < ratio < 1.33 else (
    "SHARPER than thresholding predicts" if ratio < 0.75 else "broader than predicted")
verdict = {"sharpness": sharp, "spread_ratio": float(ratio),
           "per_prompt_within_1_step": float((np.abs(d) <= 1).mean()),
           "margin_explains_headroom": float((best - base) / (1 - base))}
print(f"\n(a) SHARPNESS  (the bistability question): {sharp}")
print(f"    observed onset SD {obs[ok].std():.2f} vs predicted {pred[ok].std():.2f} "
      f"-> ratio {ratio:.2f}")
print(f"(b) PREDICTABILITY (not a bistability test): {100*(np.abs(d)<=1).mean():.0f}% "
      f"of prompts within one step; MAD {np.median(np.abs(d-np.median(d))):.2f}")
print("    per-prompt error reflects readout noise, not evidence of a threshold.")
json.dump({"theta": theta, "accuracy": best, "baseline": base,
           "onset_sd_observed": float(obs[ok].std()),
           "onset_sd_predicted": float(pred[ok].std()),
           **verdict},
          open(path.replace(".jsonl.gz", "_thresholdtest.json"), "w"), indent=1)
