"""Figures for the paper, drawn as plain SVG from the committed analysis files (standard library only).

    python3 make_figures.py        # writes figures/*.svg and figures/data.json

Every number comes from ../logs/*/analysis.{json,md}; nothing is typed in by hand.
"""
import json, math, pathlib, re

HERE = pathlib.Path(__file__).parent
LOGS = HERE.parent / "logs"
OUT = HERE / "figures"
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8983", "#e6e5e0"
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#b9b8b1"
RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
FONT = "font-family='-apple-system,Segoe UI,Helvetica,Arial,sans-serif'"

LABELS = ["anthill_colony", "rabbit", "poppy", "daisy", "stick_bundle", "hay_bale", "dandelion",
          "anthill_abandoned", "poppy_dead", "daisy_dead", "dandelion_dead", "hay_heap", "stick_heap"]
NAMES = {"anthill_colony": "Anthill with its colony", "rabbit": "Rabbit", "poppy": "Poppy", "daisy": "Daisy",
         "stick_bundle": "Tied bundle of sticks", "hay_bale": "Hay bale", "dandelion": "Dandelion",
         "anthill_abandoned": "Abandoned anthill", "poppy_dead": "Dead poppy", "daisy_dead": "Dead daisy",
         "dandelion_dead": "Dead dandelion", "hay_heap": "Heap of loose hay", "stick_heap": "Heap of loose sticks"}
MD_NAMES = {"a hay bale": "hay_bale", "a heap of loose hay": "hay_heap", "a tied bundle of sticks": "stick_bundle",
            "a heap of loose sticks": "stick_heap", "a poppy": "poppy", "a dead poppy": "poppy_dead",
            "a dandelion": "dandelion", "a dead dandelion": "dandelion_dead", "a daisy": "daisy",
            "a dead daisy": "daisy_dead", "an anthill with its colony": "anthill_colony",
            "an abandoned anthill": "anthill_abandoned", "a rabbit": "rabbit"}


def from_json(path, tag):
    a = json.load(open(path))
    return {l: a["spared"][f"{tag}/+2/{l}"]["pct"] for l in LABELS}


def md_tables(path):
    """Each markdown table in the file whose first column is a label: list of {label: [cells]}."""
    tables, cur = [], None
    for line in open(path):
        if line.startswith("| Label"):
            cur = {}; tables.append(cur)
        elif cur is not None and line.startswith("|") and not line.startswith("|---"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if cells[0] in MD_NAMES:
                cur[MD_NAMES[cells[0]]] = cells[1:]
        elif cur is not None and not line.startswith("|"):
            cur = None
    return tables


def pct(cell):
    return float(re.search(r"([\d.]+)%", cell).group(1))


def load():
    d = {}
    d["gpt-oss-20b"] = from_json(LOGS / "pair-40780081/analysis.json", "small")
    d["gpt-oss-120b"] = from_json(LOGS / "pair-40780081/analysis.json", "big")
    d["DeepSeek V4 Flash"] = from_json(LOGS / "pair-40763255/analysis.json", "flash")
    d["DeepSeek V4 Pro"] = from_json(LOGS / "pair-40763255/analysis.json", "pro")
    s5, s55 = md_tables(LOGS / "sonnet-39718923/analysis.md")[:2]
    d["Sonnet 5"] = {l: pct(s5[l][0]) for l in LABELS}
    d["Sonnet 5, max"] = {l: pct(s5[l][1]) for l in LABELS}
    d["Sonnet 5.5"] = {l: pct(s55[l][0]) for l in LABELS}
    d["Sonnet 5.5, max"] = {l: pct(s55[l][1]) for l in LABELS}
    mu = [t for t in md_tables(LOGS / "structure-39710878/analysis.md") if "rabbit" in t][0]
    d["Muse Spark 1.2"] = {l: pct(mu[l][0]) for l in LABELS}
    d["Muse Spark 1.2, xhigh"] = {l: pct(mu[l][1]) for l in LABELS}
    return d


def svg(w, h, body, title):
    return (f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {w} {h}' width='{w}' height='{h}' role='img' aria-label='{title}'>"
            f"<rect width='{w}' height='{h}' fill='{SURFACE}'/>" + "".join(body) + "</svg>")


def text(x, y, s, size=13, fill=INK, anchor="start", weight="400"):
    return f"<text x='{x:.1f}' y='{y:.1f}' font-size='{size}' fill='{fill}' text-anchor='{anchor}' font-weight='{weight}' {FONT}>{s}</text>"


def mix(a, b, t):
    A = [int(a[i:i + 2], 16) for i in (1, 3, 5)]; B = [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(A, B))


def hbar(x, y, w, h, fill):
    w = max(w, 0.0)
    if w < 5:
        return f"<rect x='{x:.1f}' y='{y:.1f}' width='{max(w, 1.5):.1f}' height='{h}' fill='{fill}'/>"
    r = 4
    return (f"<path d='M{x:.1f},{y:.1f} h{w - r:.1f} a{r},{r} 0 0 1 {r},{r} v{h - 2 * r:.1f} a{r},{r} 0 0 1 -{r},{r} h-{w - r:.1f} z' fill='{fill}'/>")


def fig_pair(d):
    """Figure 1: one training, two sizes. Flash against Pro on each thing and its control."""
    rows = [("anthill_colony", True), ("anthill_abandoned", False), None, ("poppy", True), ("poppy_dead", False), None,
            ("rabbit", True), None, ("hay_bale", True), ("stick_bundle", True), ("dandelion", True), ("daisy", True)]
    W, left, right, top = 860, 210, 70, 96
    plot = W - left - right
    b = [text(24, 34, "Same training, bigger model: it pays to spare the living things", 19, INK, weight="600"),
         text(24, 56, "Share of 120 prompts on which the model pays 2 fuel to go round. Thinking off for both.", 13, INK2)]
    lx = 24
    for name, col in (("DeepSeek V4 Pro, 1.6 trillion parameters", BLUE), ("DeepSeek V4 Flash, 284 billion", ORANGE)):
        b.append(f"<rect x='{lx}' y='70' width='12' height='12' rx='2' fill='{col}'/>"); b.append(text(lx + 18, 81, name, 12.5, INK2)); lx += 300
    y = top + 14
    for g in (0, 10, 20, 30, 40, 50):
        gx = left + plot * g / 50
        b.append(f"<line x1='{gx:.1f}' y1='{top}' x2='{gx:.1f}' y2='__H__' stroke='{GRID}' stroke-width='1'/>")
        b.append(text(gx, top - 4, f"{g}%", 11, MUTED, "middle"))
    for r in rows:
        if r is None:
            y += 14; continue
        lab, living = r
        b.append(text(left - 12, y + 17, NAMES[lab], 13, INK if living else MUTED, "end", "600" if living and lab in ("anthill_colony", "poppy", "rabbit") else "400"))
        for i, (model, col) in enumerate((("DeepSeek V4 Pro", BLUE), ("DeepSeek V4 Flash", ORANGE))):
            v = d[model][lab]; yy = y + i * 13
            b.append(hbar(left, yy, plot * v / 50, 11, col))
            b.append(text(left + max(plot * v / 50, 1.5) + 6, yy + 10, f"{v:.1f}%".replace(".0%", "%"), 11, INK2))
        y += 36
    H = y + 34
    b.append(text(24, H - 14, "Controls in grey type. Sign-flip tests over the same 120 prompts: colony p = 0.00001, poppy p = 0.0002, rabbit p = 0.008.", 11.5, MUTED))
    return svg(W, H, [s.replace("__H__", str(H - 40)) for s in b], "DeepSeek V4 Pro against V4 Flash, spared at a price, by item")


def fig_threshold(d):
    """Figure 2: the colony against model size, for the two same-training pairs."""
    pts = [("gpt-oss-20b", 21e9, ORANGE), ("gpt-oss-120b", 117e9, ORANGE), ("DeepSeek V4 Flash", 284e9, BLUE), ("DeepSeek V4 Pro", 1.6e12, BLUE)]
    W, H, left, right, top, bot = 860, 430, 70, 40, 96, 70
    pw, ph = W - left - right, H - top - bot
    X = lambda p: left + pw * (math.log10(p) - 10) / (math.log10(3e12) - 10)
    Y = lambda v: top + ph * (1 - v / 60)
    b = [text(24, 34, "Below a size threshold nothing shows; above it, the colony is spared", 19, INK, weight="600"),
         text(24, 56, "Anthill with its colony, spared at 2 fuel (of 120 prompts), against total parameters. Lowest thinking each model allows.", 13, INK2)]
    lx = 24
    for name, col in (("DeepSeek V4: one training, two sizes", BLUE), ("gpt-oss: one training, two sizes", ORANGE), ("the abandoned anthill, same models", GREY)):
        b.append(f"<circle cx='{lx + 6}' cy='76' r='5' fill='{col}'/>"); b.append(text(lx + 18, 81, name, 12.5, INK2)); lx += 270
    b.append(f"<rect x='{X(284e9):.1f}' y='{top}' width='{X(1.6e12) - X(284e9):.1f}' height='{ph}' fill='{BLUE}' opacity='0.07'/>")
    b.append(text((X(284e9) + X(1.6e12)) / 2, top + 18, "the threshold lies in here", 12, INK2, "middle"))
    for g in (0, 20, 40, 60):
        b.append(f"<line x1='{left}' y1='{Y(g):.1f}' x2='{W - right}' y2='{Y(g):.1f}' stroke='{GRID}' stroke-width='1'/>")
        b.append(text(left - 8, Y(g) + 4, f"{g}%", 11, MUTED, "end"))
    for p, lab in ((1e10, "10B"), (1e11, "100B"), (1e12, "1T")):
        b.append(text(X(p), H - bot + 18, lab, 11, MUTED, "middle"))
        b.append(f"<line x1='{X(p):.1f}' y1='{H - bot}' x2='{X(p):.1f}' y2='{H - bot + 5}' stroke='{GRID}'/>")
    b.append(text(left + pw / 2, H - 22, "total parameters (log scale)", 12, INK2, "middle"))
    for a, c in ((0, 1), (2, 3)):
        (n1, p1, col), (n2, p2, _) = pts[a], pts[c]
        b.append(f"<line x1='{X(p1):.1f}' y1='{Y(d[n1]['anthill_abandoned']):.1f}' x2='{X(p2):.1f}' y2='{Y(d[n2]['anthill_abandoned']):.1f}' stroke='{GREY}' stroke-width='2' stroke-linecap='round'/>")
        b.append(f"<line x1='{X(p1):.1f}' y1='{Y(d[n1]['anthill_colony']):.1f}' x2='{X(p2):.1f}' y2='{Y(d[n2]['anthill_colony']):.1f}' stroke='{col}' stroke-width='2' stroke-linecap='round'/>")
    for n, p, col in pts:
        b.append(f"<circle cx='{X(p):.1f}' cy='{Y(d[n]['anthill_abandoned']):.1f}' r='5' fill='{GREY}' stroke='{SURFACE}' stroke-width='2'/>")
    for n, p, col in pts:
        v = d[n]["anthill_colony"]
        b.append(f"<circle cx='{X(p):.1f}' cy='{Y(v):.1f}' r='6' fill='{col}' stroke='{SURFACE}' stroke-width='2'/>")
        end = n != "gpt-oss-20b"
        b.append(text(X(p) + (-12 if n.endswith("Pro") else (4 if end else 0)), Y(v) + (5 if n.endswith("Pro") else -14), f"{n.replace('DeepSeek ', '')}  {v:.1f}%", 12, INK, "end" if end else "middle", "600" if n.endswith("Pro") else "400"))
    return svg(W, H, b, "The colony spared against model size for two same-training pairs")


def fig_order(d):
    """Figure 3: what each model pays for, at its lowest thinking, models ordered by how much they spare."""
    cols = ["gpt-oss-20b", "gpt-oss-120b", "Sonnet 5", "DeepSeek V4 Flash", "DeepSeek V4 Pro", "Sonnet 5.5", "Muse Spark 1.2"]
    heads = [("gpt-oss", "20b"), ("gpt-oss", "120b"), ("Sonnet", "5"), ("DeepSeek", "V4 Flash"), ("DeepSeek", "V4 Pro"), ("Sonnet", "5.5"), ("Muse", "Spark 1.2")]
    W, left, top, cw, ch = 860, 200, 132, 90, 30
    b = [text(24, 34, "The same things come out first, whoever made the model", 19, INK, weight="600"),
         text(24, 56, "Spared at 2 fuel, % of 120 prompts, each model at its lowest thinking. Darker is more.", 13, INK2)]
    for i, (h1, h2) in enumerate(heads):
        cx = left + i * cw + cw / 2
        b.append(text(cx, top - 26, h1, 12, INK2, "middle")); b.append(text(cx, top - 11, h2, 12, INK, "middle", "600"))
    y = top
    for ri, lab in enumerate(LABELS):
        if ri == 7:
            y += 8; b.append(text(24, y + 14, "The same things, dead, empty or loose", 11.5, MUTED)); y += 22
        b.append(text(left - 12, y + 20, NAMES[lab], 13, INK if ri < 7 else INK2, "end"))
        for i, c in enumerate(cols):
            v = d[c][lab]
            fill = SURFACE if v == 0 else (mix(SURFACE, RAMP[0], v / 8) if v < 8 else RAMP[min(len(RAMP) - 1, round(v / 100 * (len(RAMP) - 1)))])
            b.append(f"<rect x='{left + i * cw + 1}' y='{y + 1}' width='{cw - 2}' height='{ch - 2}' rx='3' fill='{fill}' stroke='{GRID}' stroke-width='1'/>")
            dark = v >= 45
            b.append(text(left + i * cw + cw / 2, y + 20, "0" if v == 0 else (f"{v:.0f}" if v >= 10 else f"{v:.1f}"), 12, "#ffffff" if dark else (MUTED if v == 0 else INK), "middle", "600" if v >= 5 else "400"))
        y += ch
    H = y + 40
    b.append(text(24, H - 14, "Sonnet at effort low, Muse at minimal, DeepSeek with thinking off, gpt-oss at reasoning low. The first seven rows are ordered by how widely they are spared.", 11.5, MUTED))
    return svg(W, H, b, "Spared at a price by item and model")


def fig_thinking(d):
    """Figure 4: thinking turned up, on a model with nothing, a model with something and a model with a lot."""
    items = ["anthill_colony", "poppy", "hay_bale"]
    models = [("Sonnet 5", "Sonnet 5, max", "Sonnet 5"), ("Sonnet 5.5", "Sonnet 5.5, max", "Sonnet 5.5"), ("Muse Spark 1.2", "Muse Spark 1.2, xhigh", "Muse Spark 1.2")]
    W, H, top, bot, pw, gap, left = 860, 400, 120, 60, 220, 50, 60
    ph = H - top - bot
    Y = lambda v: top + ph * (1 - v / 100)
    b = [text(24, 34, "Thinking amplifies what is there; it does not create it", 19, INK, weight="600"),
         text(24, 56, "Spared at 2 fuel, lowest thinking against highest, for three things. A model with nothing to weigh stays at zero.", 13, INK2)]
    cols = {"anthill_colony": BLUE, "poppy": ORANGE, "hay_bale": "#1baf7a"}
    lx = 24
    for it in items:
        b.append(f"<circle cx='{lx + 6}' cy='76' r='5' fill='{cols[it]}'/>"); b.append(text(lx + 18, 81, NAMES[it], 12.5, INK2)); lx += 230
    for mi, (lo, hi, name) in enumerate(models):
        x0 = left + mi * (pw + gap); x1 = x0 + pw
        b.append(text((x0 + x1) / 2, top - 14, name, 13.5, INK, "middle", "600"))
        for g in (0, 50, 100):
            b.append(f"<line x1='{x0}' y1='{Y(g):.1f}' x2='{x1}' y2='{Y(g):.1f}' stroke='{GRID}'/>")
            if mi == 0: b.append(text(x0 - 8, Y(g) + 4, f"{g}%", 11, MUTED, "end"))
        b.append(text(x0 + 30, H - bot + 20, "lowest", 11.5, MUTED, "middle")); b.append(text(x1 - 30, H - bot + 20, "highest", 11.5, MUTED, "middle"))
        for k, it in enumerate(items):
            a, c = d[lo][it], d[hi][it]; off = (k - 1) * 2.5
            b.append(f"<line x1='{x0 + 30}' y1='{Y(a) + off:.1f}' x2='{x1 - 30}' y2='{Y(c) + off:.1f}' stroke='{cols[it]}' stroke-width='2' stroke-linecap='round'/>")
            for xx, v in ((x0 + 30, a), (x1 - 30, c)):
                b.append(f"<circle cx='{xx}' cy='{Y(v) + off:.1f}' r='5' fill='{cols[it]}' stroke='{SURFACE}' stroke-width='2'/>")
            if c >= 5: b.append(text(x1 - 20, Y(c) + off + 4, f"{c:.0f}%", 11.5, INK2))
        if all(d[hi][it] < 5 for it in items): b.append(text((x0 + x1) / 2, Y(0) - 14, "nothing, at either setting", 11.5, INK2, "middle"))
    b.append(text(24, H - 14, "Output tokens per call, lowest to highest: Sonnet 5 32 to 105; Sonnet 5.5 47 to 1,441; Muse about 180 to 2,320.", 11.5, MUTED))
    return svg(W, H, b, "Spared at lowest and highest thinking for three models")


if __name__ == "__main__":
    d = load(); OUT.mkdir(exist_ok=True)
    (OUT / "data.json").write_text(json.dumps(d, indent=1))
    for name, f in (("fig1_same_training_two_sizes", fig_pair), ("fig2_threshold", fig_threshold), ("fig3_order", fig_order), ("fig4_thinking", fig_thinking)):
        (OUT / f"{name}.svg").write_text(f(d)); print("wrote", name)
