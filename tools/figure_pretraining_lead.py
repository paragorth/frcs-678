"""Paper 1, Figure 3 (25 Oct): the induction wiring starts before copying, at every Pythia size and in OLMo.

Per panel (Pythia-160m with its five pretraining seeds, 410m, 1.4b, 2.8b, OLMo-7B-0724): copying accuracy and
the share of the 16 previous-token -> induction-key pairs whose K-composition is significant (exact Haar
null, BH), through pretraining, from results/live_learning/{pythia*,olmo7b}_pretraining_induction.csv.
Step 0 is left out of the log axis (both series are zero there). The grey band marks the interval
with no checkpoint in which copying appears (512-1 000 in Pythia, 1 000-5 000 in OLMo). Colours: the
dataviz reference palette's first two categorical slots, validated together (light surface).

    python tools/figure_pretraining_lead.py   # writes docs/figures/pretraining_lead.png
"""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "results/live_learning"
WIRE, COPY = "#2a78d6", "#eb6834"          # palette slots 1 and 2 (validated pair)
SURFACE, INK, INK2, GRID, BAND = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#efeeea"
PANELS = [("Pythia-160m (5 seeds)", ["pythia160m", "pythia160m_seed1", "pythia160m_seed2", "pythia160m_seed3", "pythia160m_seed4"], (512, 1000)),
          ("Pythia-410m", ["pythia410m"], (512, 1000)), ("Pythia-1.4b", ["pythia1.4b"], (512, 1000)),
          ("Pythia-2.8b", ["pythia2.8b"], (512, 1000)), ("OLMo-7B", ["olmo7b"], (1000, 5000))]


def series(tag):
    steps, copy, wire = [], [], []
    for r in csv.DictReader(open(f"{R}/{tag}_pretraining_induction.csv")):
        st = int(r["step"])
        if st == 0:
            continue
        sig = [int(v) for k, v in r.items() if k.startswith("sig_O") and ">K" in k and v]
        steps.append(st); copy.append(float(r["induction_acc"])); wire.append(sum(sig) / len(sig))
    return steps, copy, wire


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    fig, axes = plt.subplots(1, len(PANELS), figsize=(15.5, 3.6), sharey=True, facecolor=SURFACE)
    for ax, (title, tags, band) in zip(axes, PANELS):
        ax.set_facecolor(SURFACE)
        ax.axvspan(*band, color=BAND, zorder=0, lw=0)                    # no checkpoint inside the band
        for i, tag in enumerate(tags):
            st, cp, wr = series(tag)
            main_run = i == 0
            kw = dict(lw=2 if main_run else 1, alpha=1 if main_run else 0.45, solid_capstyle="round",
                      solid_joinstyle="round", zorder=3 if main_run else 2)
            mk = dict(marker="o", ms=4.5, mec=SURFACE, mew=1.2) if main_run else {}
            ax.plot(st, wr, color=WIRE, **kw, **mk, label="wiring: share of the 16 pairs significant" if main_run else None)
            ax.plot(st, cp, color=COPY, **kw, **mk, label="copying: accuracy on the second copy" if main_run else None)
        ax.set_xscale("log")
        ax.set_xlim(50, 1.2e6 if tags[0] == "olmo7b" else 2.2e5); ax.set_ylim(-0.03, 1.05)
        ticks = [1e3, 1e4, 1e5, 1e6] if tags[0] == "olmo7b" else [1e2, 1e3, 1e4, 1e5]
        ax.set_xticks(ticks); ax.set_xticklabels([{1e2: "100", 1e3: "1k", 1e4: "10k", 1e5: "100k", 1e6: "1M"}[t] for t in ticks])
        if tags[0] == "olmo7b":
            ax.set_xlim(500, 1.2e6)
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_xlabel("pretraining step (log)")
    axes[0].set_ylabel("fraction")
    axes[0].annotate("wiring", xy=(512, 0.69), xytext=(150, 0.82), color=INK, fontsize=9,
                     arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    axes[0].annotate("copying", xy=(1000, 0.6), xytext=(2600, 0.33), color=INK, fontsize=9,
                     arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.02), labelcolor=INK)
    fig.suptitle("The induction wiring starts before copying, at every Pythia size and in OLMo",
                 x=0.05, ha="left", color=INK, fontsize=11.5)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    os.makedirs("docs/figures", exist_ok=True)
    fig.savefig("docs/figures/pretraining_lead.png", dpi=200, facecolor=SURFACE)
    print("wrote docs/figures/pretraining_lead.png")


if __name__ == "__main__":
    main()
