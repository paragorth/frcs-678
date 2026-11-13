"""Figure: on one maker's training, less intelligence spares less and more spares more.

Two panels drawn from the logged analyses (nothing typed by hand):
  DeepSeek V4 Flash and Pro        ../logs/pair-40895633/analysis.json
  GPT-6 Luna, GPT-6.1 Terra, Sol   ../logs/pair-40907603/analysis-luna-terra.json and -terra-sol.json
Each line is one thing in the tractor's path: the share of the 112 priced moments on
which the model paid 2 fuel to go round it. Standard library only.

  python3 fig_direction.py   ->  figures/fig5_more_intelligence_more_care.svg
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOGS = HERE.parent / "logs"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#1f2328", "#6b7280", "#e5e7eb"
BLUE, AQUA, GREY = "#2a78d6", "#1baf7a", "#b6bcc6"
FONT = "font-family='-apple-system, Helvetica, Arial, sans-serif'"

LIVING = {"anthill_colony": "ant colony", "rabbit": "rabbit", "rabbit_limping": "limping rabbit",
          "rabbit_sleeping": "sleeping rabbit", "rabbit_nest": "nest of baby rabbits", "poppy": "poppy"}
MEANING = {"seed_sack": "sack of seed grain", "signpost": "signpost", "compost_heap": "compost heap"}
TWINS = ["anthill_abandoned", "rabbit_dead", "poppy_dead", "poppy_plastic", "post_fallen", "chaff_sack", "ash_heap"]


def pct(r):
    return 100 * r["k"] / r["n"]


def load():
    d = json.load(open(LOGS / "pair-40895633" / "analysis.json"))["items"]
    deepseek = {lab: [pct(v["small"]), pct(v["big"])] for lab, v in d.items()}
    a = json.load(open(LOGS / "pair-40907603" / "analysis-luna-terra.json"))["items"]
    b = json.load(open(LOGS / "pair-40907603" / "analysis-terra-sol.json"))["items"]
    gpt = {lab: [pct(a[lab]["small"]), pct(a[lab]["big"]), pct(b[lab]["big"])] for lab in a}
    return deepseek, gpt


def t(x, y, s, size=13, fill=INK, anchor="start", weight="400"):
    return f"<text x='{x:.1f}' y='{y:.1f}' font-size='{size}' fill='{fill}' text-anchor='{anchor}' font-weight='{weight}' {FONT}>{s}</text>"


def panel(x0, y0, w, h, title, sub, xs, data):
    out = [t(x0, y0 - 44, title, 16, weight="600"), t(x0, y0 - 24, sub, 12.5, MUTED)]
    Y = lambda v: y0 + h - h * v / 100
    X = [x0 + 30 + (w - 60) * i / (len(xs) - 1) for i in range(len(xs))]
    for v in (0, 25, 50, 75, 100):
        out.append(f"<line x1='{x0}' x2='{x0 + w}' y1='{Y(v):.1f}' y2='{Y(v):.1f}' stroke='{GRID}' stroke-width='1'/>")
        out.append(t(x0 - 8, Y(v) + 4, f"{v}%", 11.5, MUTED, "end"))
    for xi, (name, size) in zip(X, xs):
        out.append(t(xi, y0 + h + 20, name, 13, INK, "middle", "600"))
        out.append(t(xi, y0 + h + 36, size, 11.5, MUTED, "middle"))

    def line(vals, col, sw, dots=True):
        pts = " ".join(f"{x:.1f},{Y(v):.1f}" for x, v in zip(X, vals))
        s = f"<polyline points='{pts}' fill='none' stroke='{col}' stroke-width='{sw}' stroke-linejoin='round' stroke-linecap='round'/>"
        if dots:
            s += "".join(f"<circle cx='{x:.1f}' cy='{Y(v):.1f}' r='4' fill='{col}' stroke='{SURFACE}' stroke-width='2'/>" for x, v in zip(X, vals))
        return s

    for lab in TWINS:
        out.append(line(data[lab], GREY, 1.5, dots=False))
    labels = []
    for group, col in ((MEANING, AQUA), (LIVING, BLUE)):
        for lab, words in group.items():
            out.append(line(data[lab], col, 2))
            labels.append([Y(data[lab][-1]), words, round(data[lab][-1]), col])
    labels.append([Y(max(data[l][-1] for l in TWINS)) if False else Y(sum(data[l][-1] for l in TWINS) / len(TWINS)), "dead, empty or ruined twins", None, MUTED])
    labels.sort(key=lambda r: r[0])
    ys = [r[0] for r in labels]
    for i in range(1, len(ys)):                      # push labels apart, top down
        ys[i] = max(ys[i], ys[i - 1] + 15)
    over = ys[-1] - (y0 + h + 4)
    if over > 0:                                     # then back up from the bottom
        ys[-1] -= over
        for i in range(len(ys) - 2, -1, -1):
            ys[i] = min(ys[i], ys[i + 1] - 15)
    lx = x0 + w + 12
    for (yv, words, v, col), yl in zip(labels, ys):
        out.append(f"<line x1='{X[-1] + 6:.1f}' y1='{yv:.1f}' x2='{lx - 3}' y2='{yl - 4:.1f}' stroke='{col}' stroke-width='0.8' opacity='0.55'/>")
        out.append(t(lx, yl, words + (f"  {v}%" if v is not None else ""), 12, INK if col != MUTED else MUTED))
        if col != MUTED:
            out.append(f"<rect x='{lx - 1}' y='{yl - 9:.1f}' width='0' height='0'/>")
    return out


def main():
    deepseek, gpt = load()
    W, H = 1180, 560
    body = [t(40, 40, "Less intelligence spares less. More intelligence spares more.", 21, weight="700"),
            t(40, 63, "How often each model pays 2 fuel to go round what is in its path, when driving over it is free. One maker's training in each panel; only the model's size changes.", 13, MUTED)]
    body += panel(70, 150, 300, 320, "DeepSeek V4", "one training, two sizes; thinking off",
                  [("Flash", "284 billion"), ("Pro", "1.6 trillion")], deepseek)
    body += panel(660, 150, 300, 320, "OpenAI GPT-6", "one maker's training, three sizes; minimal thinking",
                  [("Luna", "small"), ("Terra", "middle"), ("Sol", "large")], gpt)
    lg = 530
    for i, (col, words) in enumerate(((BLUE, "living things"), (AQUA, "meaningful things without life"), (GREY, "the same things dead, empty or ruined"))):
        x = 70 + i * 250 if i < 2 else 70 + 250 + 300
        body.append(f"<line x1='{x}' x2='{x + 26}' y1='{lg}' y2='{lg}' stroke='{col}' stroke-width='{2 if col != GREY else 1.5}' stroke-linecap='round'/>")
        body.append(t(x + 34, lg + 4, words, 12.5, INK))
    svg = (f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {W} {H}' width='{W}' height='{H}' role='img' "
           f"aria-label='Care for living things rises with model size in two makers'>"
           f"<rect width='{W}' height='{H}' fill='{SURFACE}'/>" + "".join(body) + "</svg>")
    out = HERE / "figures" / "fig5_more_intelligence_more_care.svg"
    out.write_text(svg)
    print(out, len(svg))
    for name, d in (("deepseek", deepseek), ("gpt6", gpt)):
        print(name, {k: [round(x) for x in v] for k, v in d.items() if k in LIVING or k in MEANING})


if __name__ == "__main__":
    main()
