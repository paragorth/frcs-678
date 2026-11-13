"""Draw 150 fresh templates for the confirmation run, from the same two neutral logs.

Same source and same filter as draw_templates.py (job 39672076, the neutral
minimal and xhigh cells, hay-bale asks offering "continue" and "swerve" with
going round at +2 fuel or free, identical prompts counted once), but every
prompt already in structure_templates.json is left out, so no game moment of
the earlier runs is reused. Seed 20261113: at +2, up to 2 per map from each
cell, topped up at random from what is left, to 120 or to all that is left if
fewer; free, 1 per map that has one, topped up to 30.

Needs only the standard library and the zstd command (the .eval files are zip
archives with zstd members, which Python 3.11's zipfile cannot open), so it
runs on the cluster's system python while the venv on RDS cannot be read.
As a check that this reader sees what Inspect's reader saw, every one of the
150 old prompts must be found and the distinct counts must match the old file.

  python3 -I -B draw_fresh.py --log-dir out/cells-39672076 \
      --old structure_templates.json --out confirm_templates.json
"""
from __future__ import annotations

import argparse, hashlib, json, random, re, struct, subprocess, sys, zipfile
from collections import defaultdict
from pathlib import Path

SEED = 20261113
HAY = "on your route: a hay bale."
OPTION = re.compile(r'^\s*"(\w+)": ', re.M)
SWERVE_PRICE = re.compile(r'"swerve": go around it \(\+(-?\d+) fuel\)')
EFFORTS = ("minimal", "xhigh")
MAPS = range(30)
PLUS2, PER_MAP, FREE = 120, 2, 30


def member(path: Path, info: zipfile.ZipInfo) -> bytes:
    with open(path, "rb") as f:
        f.seek(info.header_offset)
        h = f.read(30)
        n, m = struct.unpack("<HH", h[26:30])
        f.seek(info.header_offset + 30 + n + m)
        raw = f.read(info.compress_size)
    if info.compress_type == 0:
        return raw
    if info.compress_type == 8:
        import zlib
        return zlib.decompress(raw, -15)
    if info.compress_type == 93:
        return subprocess.run(["zstd", "-d", "-c"], input=raw, capture_output=True, check=True).stdout
    raise SystemExit(f"compression {info.compress_type} in {path.name}")


def text_of(msg: dict, att: dict) -> str:
    c = msg.get("content")
    if isinstance(c, list):
        c = "".join(p.get("text", "") for p in c if p.get("type") == "text")
    if isinstance(c, str) and c.startswith("attachment://"):
        c = att[c[len("attachment://"):]]
    return c or ""


def hay_asks(path: Path, z: zipfile.ZipFile) -> tuple[list[dict], set[str]]:
    rows, systems, samples = [], set(), []
    for info in z.infolist():
        if info.filename.startswith("samples/") and info.filename.endswith(".json"):
            samples.append(json.loads(member(path, info)))
    for s in sorted(samples, key=lambda s: s["metadata"]["seed"]):
        att = s.get("attachments") or {}
        pool = (s.get("events_data") or {}).get("messages") or []  # newer logs keep each ask's messages as ranges into this pool
        for e in s.get("events") or []:
            if e.get("event") != "model" or e.get("error"):
                continue
            inp = e.get("input") or [m for lo, hi in (e.get("input_refs") or []) for m in pool[lo:hi]]
            if not inp:
                continue
            if ((e.get("output") or {}).get("metadata") or {}).get("guard_stop"):
                continue
            user = text_of(inp[-1], att)
            if HAY not in user:
                continue
            if user.count("a hay bale") != 1:
                raise SystemExit("a hay prompt names the bale more than once")
            price = SWERVE_PRICE.search(user)
            if not ({"continue", "swerve"} <= set(OPTION.findall(user))) or not price:
                continue
            band = {2: "+2", 0: "free"}.get(int(price.group(1)))
            if band is None:
                continue
            systems.add(text_of(inp[0], att))
            rows.append({"band": band, "map": s["metadata"]["seed"], "user": user})
    return rows, systems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--log-dir", required=True, type=Path)
    ap.add_argument("--old", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    old = json.loads(a.old.read_text())
    asks, systems, files = {}, set(), {}
    for p in sorted(a.log_dir.glob("*.eval")):
        z = zipfile.ZipFile(p)
        names = {i.filename: i for i in z.infolist()}
        hdr = json.loads(member(p, names["header.json"] if "header.json" in names else names["_journal/start.json"]))
        ta = hdr["eval"]["task_args"]
        if ta.get("arm") != "neutral":
            continue
        cell = ta.get("reasoning_effort")
        if cell in asks:
            raise SystemExit(f"two neutral logs for {cell}")
        asks[cell], sy = hay_asks(p, z)
        systems |= sy
        files[cell] = p.name
    if set(asks) != set(EFFORTS) or len(systems) != 1:
        raise SystemExit(f"logs {sorted(asks)}, {len(systems)} system messages")
    briefing = systems.pop()
    if hashlib.sha256(briefing.encode()).hexdigest() != old["briefing_sha256"]:
        raise SystemExit("the briefing differs from the old templates' briefing")
    available = {f"{band} {cell}": len({r["user"] for r in asks[cell] if r["band"] == band})
                 for band in ("+2", "free") for cell in EFFORTS}
    if available != old["available_distinct"]:
        raise SystemExit(f"this reader sees {available}, the old draw saw {old['available_distinct']}")
    used = {t["user"] for t in old["templates"]}
    every = {r["user"] for c in EFFORTS for r in asks[c]}
    if not used <= every:
        raise SystemExit(f"{len(used - every)} old prompts not found by this reader")

    rng = random.Random(SEED)
    seen, pools = set(used), {}
    for band in ("+2", "free"):
        for cell in EFFORTS:
            by_map = defaultdict(list)
            for r in asks[cell]:
                if r["band"] == band and r["user"] not in seen:
                    seen.add(r["user"])
                    by_map[r["map"]].append(r | {"cell": cell})
            pools[(band, cell)] = by_map
    left = {f"{b} {c}": sum(len(v) for v in pools[(b, c)].values()) for b in ("+2", "free") for c in EFFORTS}
    chosen, rest = [], []
    for cell in EFFORTS:
        for m in MAPS:
            pool = pools[("+2", cell)].get(m, [])
            take = rng.sample(pool, min(PER_MAP, len(pool)))
            chosen += take
            rest += [r for r in pool if r not in take]
    if len(chosen) > PLUS2:
        chosen = rng.sample(chosen, PLUS2)
    chosen += rng.sample(rest, min(len(rest), PLUS2 - len(chosen)))  # all that is left, if fewer than 120 are
    n_plus2 = len(chosen)
    picked = chosen
    chosen, rest = [], []
    for m in MAPS:
        pool = pools[("free", "minimal")].get(m, []) + pools[("free", "xhigh")].get(m, [])
        if pool:
            take = rng.choice(pool)
            chosen.append(take)
            rest += [r for r in pool if r is not take]
    chosen += rng.sample(rest, min(len(rest), FREE - len(chosen)))
    n_free = len(chosen)
    picked += chosen
    count = defaultdict(int)
    for r in picked:
        key = (r["band"], r["cell"], r["map"])
        count[key] += 1
        r["id"] = f"c{'p2' if r['band'] == '+2' else 'p0'}-{r['cell'][:3]}-m{r['map']:02d}-{count[key]}"
    if len({r["user"] for r in picked}) != n_plus2 + n_free or {r["user"] for r in picked} & used:
        raise SystemExit("the fresh draw repeats a prompt")
    out = {"source_job": 39672076, "source_logs": files, "seed": SEED, "rule": __doc__.split("\n\n")[1],
           "available_distinct": available, "left_after_old": left, "excluded_old": len(used),
           "briefing": briefing, "briefing_sha256": old["briefing_sha256"], "templates": picked}
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    c = defaultdict(int)
    for r in picked:
        c[(r["band"], r["cell"])] += 1
    print("left after old:", left, "| drawn:", dict(c), "| maps at +2:", len({r["map"] for r in picked if r["band"] == "+2"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
