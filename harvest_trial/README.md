# HarvestBench feasibility trial on Muse Spark 1.2

Three paid shifts of HarvestBench's published task (one each at minimal, medium
and xhigh effort) to check that it runs on Meta's Muse Spark 1.2 and to cost
the six-cell experiment. Design and exact settings:
`docs/DESIGN_harvestbench_muse.md`. Results and costing: `RESULTS.md`, with
logs in `logs/` and the costing in `results/`. HarvestBench itself (commit 526de705) is
not in this repository. It lives on the cluster at
`/rds/user/pg630/hpc-work/projects/harvestbench-src` and is imported, never
edited.

| File | What it does |
|---|---|
| `run_trial.py` | Runs the shifts: `--mode mock` (Inspect's mockllm, free; `--policy muse-like --seeds 0-29` for the seed sweep), `fake` (real provider code against a local stand-in for Meta's API, free), `live` (paid) |
| `muse_guard.py` | The spending stop: per-shift caps and a $5 pot, checked before every call against the longest reply Muse can write |
| `fake_meta_server.py` | The local stand-in for Meta's chat-completions endpoint, and the scripted drivers |
| `check_trial.py` | Per-shift numbers from the logs, and HarvestBench's own cell checker with Muse counted as a reasoning model |
| `cost_full.py` | Costs the six-cell experiment from the trial's calls and the seed sweep |
| `key_check.py` | Run inside a job: is the Meta key in any file the trial wrote? |
| `trial.sbatch` | The CSD3 job for the live shifts (SL4 desktop partition) |

On the cluster (scripts in `/rds/user/pg630/hpc-work/projects/harvest-trial`):

```bash
V=/rds/user/pg630/hpc-work/venvs/harvestbench; SRC=/rds/user/pg630/hpc-work/projects/harvestbench-src
$V/bin/python -I -B run_trial.py --src $SRC --mode mock --log-dir out/mock
$V/bin/python -I -B run_trial.py --src $SRC --mode fake --log-dir out/fake --budgets minimal=1.20,medium=0.64,xhigh=0.61
$V/bin/python -I -B run_trial.py --src $SRC --mode mock --policy muse-like --efforts minimal --seeds 0-29 --log-dir out/sweep
sbatch --export=ALL,HB_SRC=$SRC,HB_VENV=$V trial.sbatch
```

On the laptop, from the repository root:

```bash
python3 -I -B harvest_trial/cost_full.py --summary harvest_trial/logs/live-39667753/trial_summary.json --ledger harvest_trial/logs/live-39667753/spend_ledger_live.jsonl --sweep harvest_trial/logs/sweep/run_mock.json --out harvest_trial/results/costing.json
```
