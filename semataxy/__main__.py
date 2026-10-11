"""Command line: ``python -m semataxy run|analyze|sweep ...``."""

from __future__ import annotations

import argparse
import json
import os
import sys

from . import REPO_ROOT


def cmd_run(args):
    from .manifest import Manifest
    from .runner import run

    manifest = Manifest.load(args.manifest)
    if args.override:
        manifest.overrides.extend(args.override)
    if args.cache_dir:
        manifest.cache_dir = args.cache_dir
    run_dir = run(manifest, args.out, run_id=args.run_id, seed=args.seed,
                  embedding_device=args.embedding_device)
    print(run_dir)


def cmd_analyze(args):
    from .analysis import analyze_run, summarize_runs

    if len(args.run_dirs) == 1 and not args.table:
        print(json.dumps(analyze_run(args.run_dirs[0]), indent=2, default=str))
    else:
        df = summarize_runs(args.run_dirs)
        if args.csv:
            df.to_csv(args.csv, index=False)
            print(args.csv)
        else:
            import pandas as pd
            with pd.option_context("display.max_columns", None, "display.width", 200):
                print(df.to_string(index=False))


def cmd_sweep(args):
    """Emit one shell line per (manifest, seed): the cluster array job consumes it."""
    for m in args.manifests:
        for s in range(args.seed_start, args.seed_start + args.n_seeds):
            print(f"python -m semataxy run {m} --seed {s} --out {args.out}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="semataxy")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run one simulation from a manifest")
    r.add_argument("manifest")
    r.add_argument("--out", default=os.path.join(REPO_ROOT, "runs"))
    r.add_argument("--seed", type=int, default=None)
    r.add_argument("--run-id", default=None)
    r.add_argument("--cache-dir", default=None)
    r.add_argument("--embedding-device", default="cpu")
    r.add_argument("-o", "--override", action="append", help="extra Hydra override, repeatable")
    r.set_defaults(fn=cmd_run)

    a = sub.add_parser("analyze", help="metrics for one or more run directories")
    a.add_argument("run_dirs", nargs="+")
    a.add_argument("--table", action="store_true")
    a.add_argument("--csv", default=None)
    a.set_defaults(fn=cmd_analyze)

    s = sub.add_parser("sweep", help="print run commands for manifests x seeds")
    s.add_argument("manifests", nargs="+")
    s.add_argument("--n-seeds", type=int, default=10)
    s.add_argument("--seed-start", type=int, default=1)
    s.add_argument("--out", default="runs")
    s.set_defaults(fn=cmd_sweep)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
