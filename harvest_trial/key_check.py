"""Is the Meta key in any file the trial wrote? Prints a count and file names, never the key.

Run inside a job, with META_API_KEY set from the key file. Every file is
searched as raw bytes; each Inspect .eval log is also read back with Inspect's
own reader and searched as JSON text, since Python's zipfile cannot open its
compressed members.
"""

import os
import sys
from pathlib import Path


def main() -> int:
    key = os.environ.get("META_API_KEY", "").strip()
    if len(key) < 8:
        sys.exit("key missing or unreadable; check not run")
    from inspect_ai.log import read_eval_log

    hits, n = [], 0
    for p in sorted(Path(sys.argv[1]).rglob("*")):
        if not p.is_file():
            continue
        n += 1
        if key.encode() in p.read_bytes():
            hits.append(str(p))
        if p.suffix == ".eval" and key in read_eval_log(str(p)).model_dump_json():
            hits.append(f"{p} (decoded)")
    print(f"searched {n} files; files containing the Meta key: {len(hits)} {hits}")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
