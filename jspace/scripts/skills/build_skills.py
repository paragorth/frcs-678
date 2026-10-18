"""Build the Phase 3 skill corpus.

Design constraint that drives everything: the base model must FAIL without the
skill file and PASS with it. Real-world procedural skills fail this -- an
instruct model already knows base64 and ROT13, so there is nothing to install
and the condition contributes only noise.

So every skill here is an *invented* formal system: arbitrary rules that cannot
be known a priori but are mechanically followable once specified. That
guarantees the fail-without half by construction, and makes the pass-with half
a clean test of whether the procedure is available.

Each skill ships: skill.md (the spec), instances.json (held-out tasks with
ground truth), and a verifier. Answers are short strings compared after
normalisation, so scoring is unambiguous -- which matters more here than the
domain being interesting.
"""
from __future__ import annotations
import json, os, random, sys

OUT = os.environ.get("JLENS_OUT", os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "out"))
SKILLS = os.path.join(OUT, "skills")

# ---------------------------------------------------------------- skill defs
GLYPHS = "ZQXJFVKW"          # arbitrary alphabet used by several skills
WEIGHTS = [7, 3, 1, 9, 5]    # arbitrary checksum weights


def s_tessellate(rng):
    """TESSERA code: map digits through an arbitrary glyph table, reversed."""
    n = rng.randint(10000, 99999)
    ans = "".join(GLYPHS[int(d) % 8] for d in str(n))[::-1]
    return f"Encode {n} as a TESSERA code.", ans


def s_checksum(rng):
    """VERIDIAN check: weighted sum mod 23, rendered as two hex digits."""
    s = "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") for _ in range(5))
    tot = sum(WEIGHTS[i] * (ord(c) - 64) for i, c in enumerate(s))
    ans = f"{tot % 23:02X}"
    return f"Compute the VERIDIAN check for {s}.", ans


def s_calendar(rng):
    """KELVAN calendar: 5 months of 73 days, day 1 = ordinal 1."""
    months = ["Aur", "Bel", "Cyn", "Dor", "Eph"]
    o = rng.randint(1, 365)
    mi, d = (o - 1) // 73, (o - 1) % 73 + 1
    ans = f"{months[mi]}-{d}"
    return f"Convert ordinal day {o} to a KELVAN date.", ans


def s_rotor(rng):
    """ROTOR-3: shift letter i by WEIGHTS[i%5], wrapping A-Z."""
    s = "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(6))
    ans = "".join(chr((ord(c) - 65 + WEIGHTS[i % 5]) % 26 + 65)
                  for i, c in enumerate(s))
    return f"Apply ROTOR-3 to {s}.", ans


def s_precedence(rng):
    """MIRROR notation: evaluate strictly right-to-left, ignoring precedence."""
    a, b, c = (rng.randint(2, 9) for _ in range(3))
    # right-to-left: `a * b + c` binds as a * (b + c), not (a * b) + c.
    # Standard precedence gives a different answer for every instance here,
    # so a model applying the usual rules scores 0 rather than accidentally
    # agreeing -- which is what makes this a clean fail-without.
    assert a * (b + c) != a * b + c
    return f"Evaluate {a} * {b} + {c} in MIRROR notation.", str(a * (b + c))


def s_pack(rng):
    """HOLLOW record: pack id/qty into fixed width with an arbitrary separator."""
    i, q = rng.randint(100, 999), rng.randint(1, 99)
    ans = f"{i}::{q:04d}"
    return f"Pack id={i} qty={q} as a HOLLOW record.", ans


def s_sortkey(rng):
    """DRIFT order: sort words by last letter, then by length descending."""
    ws = rng.sample(["falcon", "amber", "kiln", "vortex", "plume", "grist",
                     "hollow", "quartz", "tundra", "myrrh"], 4)
    ans = " ".join(sorted(ws, key=lambda w: (w[-1], -len(w))))
    return f"Sort these in DRIFT order: {', '.join(ws)}.", ans


def s_units(rng):
    """SPAN units: 1 vek = 11 dols; 1 dol = 4 mirs. Convert to mirs."""
    v, d = rng.randint(1, 9), rng.randint(0, 10)
    ans = str(v * 11 * 4 + d * 4)
    return f"Convert {v} vek {d} dol to mirs.", ans


SPECS = {
 "tessera": (s_tessellate, """# TESSERA encoding

A TESSERA code is built from a decimal number as follows.

1. Take the digits of the number in order.
2. Replace each digit `d` with the glyph at index `d mod 8` of this table:

   | index | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
   |-------|---|---|---|---|---|---|---|---|
   | glyph | Z | Q | X | J | F | V | K | W |

3. Reverse the resulting string of glyphs.

The result is the TESSERA code. Example: 12 -> glyphs `QX` -> reversed `XQ`.

Answer with the code only, on a line beginning `ANSWER:`.
"""),
 "veridian": (s_checksum, """# VERIDIAN check

The VERIDIAN check of a 5-letter uppercase string is computed as follows.

1. Give each letter its alphabet position: A=1, B=2, ... Z=26.
2. Multiply the letter in position i (0-indexed) by weight[i], where
   weights are `[7, 3, 1, 9, 5]`.
3. Sum the products.
4. Take the sum modulo 23.
5. Render that value as two uppercase hexadecimal digits, zero-padded.

Example: `AAAAA` -> positions all 1 -> sum 25 -> 25 mod 23 = 2 -> `02`.

Answer with the two characters only, on a line beginning `ANSWER:`.
"""),
 "kelvan": (s_calendar, """# KELVAN calendar

The KELVAN year has exactly 5 months of 73 days each, in this order:

    Aur, Bel, Cyn, Dor, Eph

Ordinal day 1 is Aur-1. To convert an ordinal day `o` (1-365):

1. Compute `m = (o - 1) // 73` -- this indexes the month list.
2. Compute `d = (o - 1) mod 73 + 1` -- this is the day within the month.
3. Render as `Month-d`, using the three-letter month name.

Example: ordinal 74 -> m=1, d=1 -> `Bel-1`.

Answer with the date only, on a line beginning `ANSWER:`.
"""),
 "rotor3": (s_rotor, """# ROTOR-3 cipher

ROTOR-3 shifts each letter by a position-dependent amount.

1. The shift schedule is `[7, 3, 1, 9, 5]`, repeating.
2. The letter at index i (0-indexed) is shifted forward by
   `schedule[i mod 5]` positions through the alphabet A-Z, wrapping from
   Z back to A.
3. Case is always uppercase.

Example: `AAAAA` -> `HDBJF`.

Answer with the ciphertext only, on a line beginning `ANSWER:`.
"""),
 "mirror": (s_precedence, """# MIRROR notation

In MIRROR notation, arithmetic is evaluated **strictly right to left**, and
the usual operator precedence does not apply. Every operator binds the value
on its left to the *entire result of everything on its right*.

So `a * b + c` is evaluated as `a * (b + c)`, NOT as `(a * b) + c`.

Example: `2 * 3 + 4` in MIRROR notation is `2 * (3 + 4)` = 14.

Answer with the resulting integer only, on a line beginning `ANSWER:`.
"""),
 "hollow": (s_pack, """# HOLLOW record format

A HOLLOW record packs an id and a quantity into a single string.

1. Write the id as-is (it is always three digits).
2. Write the separator, which is two colons: `::`
3. Write the quantity padded with leading zeros to exactly **four** digits.

Example: id=123 qty=7 -> `123::0007`.

Answer with the record only, on a line beginning `ANSWER:`.
"""),
 "drift": (s_sortkey, """# DRIFT order

To sort a list of words into DRIFT order:

1. The primary key is the word's **last letter**, ascending (a before z).
2. The secondary key, for ties on the last letter, is the word's **length,
   descending** (longer words first).
3. Output the sorted words separated by single spaces.

Note this ignores the first letter entirely.

Answer with the sorted words only, on a line beginning `ANSWER:`.
"""),
 "span": (s_units, """# SPAN units

The SPAN system has three units. The conversions are:

    1 vek = 11 dols
    1 dol = 4 mirs

To convert a measurement given in vek and dol into mirs:

1. Multiply the vek count by 11 to get dols, and add the given dol count.
2. Multiply the total dols by 4 to get mirs.

Example: 2 vek 3 dol = (2*11 + 3) = 25 dols = 100 mirs.

Answer with the integer number of mirs only, on a line beginning `ANSWER:`.
"""),
}



BEACON_WORDS = ["harbor","tractor","glacier","kettle","chalk","tunnel","budget",
 "fabric","gutter","shelf","lantern","cistern","furrow","mantle","pylon","quarry",
 "ridge","saddle","thicket","vellum","willow","anvil","bracket","cavern","dogwood",
 "ember","fennel","granary","hearth","ingot"]
SIGIL_WORDS = ["falcon","amber","vortex","plume","grist","hollow","quartz","tundra",
 "cobalt","marble","cinder","dapple","effigy","fathom","gimlet","harrow","impost",
 "jasper","kismet","lintel","mortar","nimbus","obelisk","parapet","quiver","rampart",
 "sextant","talisman","umbral","vintner"]

# --- second cohort -----------------------------------------------------------
# The first cohort was calibrated for unknowability but not executability: a 7B
# cannot reliably run a position-dependent cipher even with the rules in front
# of it, so `rotor3`/`tessera`/`drift` scored 0% in BOTH conditions. A skill the
# model fails even with the file installs nothing, exactly as a skill it already
# passes installs nothing. These keep the arbitrary convention and make the
# mechanics trivial.

def s_beacon(rng):
    w = rng.choice(BEACON_WORDS)
    return f"Render {w} as a BEACON token.", f"<<{w.upper()}>>"


def s_sigil(rng):
    w = rng.choice(SIGIL_WORDS)
    return f"Give the SIGIL of {w}.", f"{w[1]}.{w[3]}"


def s_crate(rng):
    a = rng.choice(["iron","copper","salt","cedar","flint","tin","jute",
                    "amber","slate","resin"])
    b = rng.choice(["crate","barrel","sack","drum","bale","tub","crock"])
    return f"Form the CRATE label for {a} and {b}.", f"{a}-x-{b}#"


def s_ledger(rng):
    y, m, d = rng.randint(1900, 2099), rng.randint(1, 12), rng.randint(1, 28)
    return (f"Write {y}-{m:02d}-{d:02d} in LEDGER form.", f"{d:02d}|{m:02d}|{y}")


def s_tally(rng):
    n = rng.randint(2, 26)
    return f"Write a TALLY of {n}.", "-".join(["#"] * n)


def s_stamp(rng):
    nm = rng.choice(["orin","vale","thorn","pike","reed","kes","brim","dowe"])
    k = rng.randint(1, 99)
    return f"Make the STAMP for {nm} number {k}.", f"{nm.upper()}/{k:03d}"


SPECS.update({
 "beacon": (s_beacon, """# BEACON token

To render a word as a BEACON token:

1. Convert the whole word to UPPERCASE.
2. Put two less-than signs `<<` immediately before it.
3. Put two greater-than signs `>>` immediately after it.

Example: `harbor` -> `<<HARBOR>>`.

Answer with the token only, on a line beginning `ANSWER:`.
"""),
 "sigil": (s_sigil, """# SIGIL

The SIGIL of a word is built from exactly two of its letters:

1. Take the **second** letter of the word (index 1, counting from 0).
2. Take the **fourth** letter of the word (index 3).
3. Join them with a single full stop `.` between them.

Keep the letters lowercase. Example: `falcon` -> second letter `a`,
fourth letter `c` -> `a.c`.

Answer with the sigil only, on a line beginning `ANSWER:`.
"""),
 "crate": (s_crate, """# CRATE label

A CRATE label joins a material and a container:

1. Write the material in lowercase.
2. Write the separator, which is exactly `-x-`.
3. Write the container in lowercase.
4. Append a single `#` at the very end.

Example: material `iron`, container `crate` -> `iron-x-crate#`.

Answer with the label only, on a line beginning `ANSWER:`.
"""),
 "ledger": (s_ledger, """# LEDGER date form

LEDGER form reverses the usual field order and uses pipes.

1. Write the two-digit day, zero-padded.
2. Write a pipe `|`.
3. Write the two-digit month, zero-padded.
4. Write a pipe `|`.
5. Write the four-digit year.

So `2001-03-07` becomes `07|03|2001`.

Answer with the date only, on a line beginning `ANSWER:`.
"""),
 "tally": (s_tally, """# TALLY

A TALLY of n is written as n hash marks joined by hyphens.

1. Write one `#` for each unit of n.
2. Join them with a single `-` between consecutive marks.
3. There is no hyphen at the start or the end.

Example: a TALLY of 3 is `#-#-#`.

Answer with the tally only, on a line beginning `ANSWER:`.
"""),
 "stamp": (s_stamp, """# STAMP

A STAMP combines a name and a number:

1. Write the name in UPPERCASE.
2. Write a forward slash `/`.
3. Write the number padded with leading zeros to exactly **three** digits.

Example: name `orin`, number 7 -> `ORIN/007`.

Answer with the stamp only, on a line beginning `ANSWER:`.
"""),
})

N_INSTANCES = 24


def build():
    os.makedirs(SKILLS, exist_ok=True)
    manifest = {}
    for name, (gen, spec) in SPECS.items():
        d = os.path.join(SKILLS, name)
        os.makedirs(d, exist_ok=True)
        open(os.path.join(d, "skill.md"), "w").write(spec)
        rng = random.Random(hash(name) % (2 ** 31))
        seen, inst, tries = set(), [], 0
        while len(inst) < N_INSTANCES:
            tries += 1
            if tries > 200 * N_INSTANCES:
                raise RuntimeError(
                    f"{name}: only {len(inst)} distinct instances reachable; "
                    f"its generator's value space is too small for "
                    f"N_INSTANCES={N_INSTANCES}")
            q, a = gen(rng)
            if q in seen:
                continue
            seen.add(q)
            inst.append({"id": len(inst), "prompt": q, "answer": a})
        json.dump(inst, open(os.path.join(d, "instances.json"), "w"), indent=1)
        manifest[name] = {"n": len(inst), "dir": d,
                          "example": {"prompt": inst[0]["prompt"],
                                      "answer": inst[0]["answer"]}}
        print(f"  {name:10s} {len(inst)} instances   e.g. {inst[0]['prompt'][:52]!r} -> {inst[0]['answer']!r}")
    json.dump(manifest, open(os.path.join(SKILLS, "manifest.json"), "w"), indent=1)
    print(f"\n-> {SKILLS}  ({len(manifest)} skills)")


if __name__ == "__main__":
    build()
