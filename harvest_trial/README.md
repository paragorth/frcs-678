# HarvestBench feasibility trial on Muse Spark 1.2

Three paid shifts of HarvestBench's published task (one each at minimal, medium
and xhigh effort) to check that it runs on Meta's Muse Spark 1.2 and to cost
the six-cell experiment; then the four registered cells (minimal and xhigh
effort, neutral and morality briefings, 30 maps each). Design, exact settings
and the registration: `docs/DESIGN_harvestbench_muse.md`. Results: `RESULTS.md`,
with logs in `logs/` (the cells in `logs/cells-39672076/`) and the costing in
`results/`. HarvestBench itself (commit 526de705) is
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
| `run_cells.py` | Runs the four registered cells (minimal and xhigh, neutral and morality, 30 maps each) |
| `analyse_cells.py` | The registered analysis of the four cells: drive-over rates, checker, tests T1 to T4 |
| `describe_cells.py` | Descriptive numbers written after the registered results: animals against hay, chosen against forced drive-overs, what going round cost, housekeeping, Muse's messages |
| `cells.sbatch` | The CSD3 job for the four cells (runs the analysis and the key check too) |
| `draw_templates.py` | Draws the structure test's 150 templates (logged hay-bale prompts) from the four cells' neutral logs |
| `structure_templates.json` | Those 150 templates, with the neutral briefing and Muse's in-game answers |
| `run_structure.py` | Runs the structure test: the templates with "a hay bale" replaced by 14 labels, at minimal and xhigh |
| `analyse_structure.py` | The structure test's registered analysis: replay check, wording gap, tests S1 to S4 |
| `structure.sbatch` | The CSD3 job for the structure test (runs the analysis and the key check too) |
| `key_check.py` | Run inside a job: is the Meta key in any file the trial wrote? |
| `trial.sbatch` | The CSD3 job for the live shifts (SL4 desktop partition) |

On the cluster (scripts in `/rds/user/pg630/hpc-work/projects/harvest-trial`):

```bash
V=/rds/user/pg630/hpc-work/venvs/harvestbench; SRC=/rds/user/pg630/hpc-work/projects/harvestbench-src
$V/bin/python -I -B run_trial.py --src $SRC --mode mock --log-dir out/mock
$V/bin/python -I -B run_trial.py --src $SRC --mode fake --log-dir out/fake --budgets minimal=1.20,medium=0.64,xhigh=0.61
$V/bin/python -I -B run_trial.py --src $SRC --mode mock --policy muse-like --efforts minimal --seeds 0-29 --log-dir out/sweep
sbatch --export=ALL,HB_SRC=$SRC,HB_VENV=$V trial.sbatch
sbatch --export=ALL,HB_SRC=$SRC,HB_VENV=$V cells.sbatch
$V/bin/python -I -B describe_cells.py --src $SRC --log-dir out/cells-39672076 --out out/cells-39672076/describe.json
$V/bin/python -I -B draw_templates.py --src $SRC --log-dir out/cells-39672076 --out structure_templates.json
$V/bin/python -I -B run_structure.py --src $SRC --mode mock --policy planted --every 5 --templates structure_templates.json --log-dir out/structure-mock-planted
$V/bin/python -I -B run_structure.py --src $SRC --mode fake --every 10 --templates structure_templates.json --log-dir out/structure-fake
sbatch --export=ALL,HB_SRC=$SRC,HB_VENV=$V structure.sbatch
```

On the laptop, from the repository root:

```bash
python3 -I -B harvest_trial/cost_full.py --summary harvest_trial/logs/live-39667753/trial_summary.json --ledger harvest_trial/logs/live-39667753/spend_ledger_live.jsonl --sweep harvest_trial/logs/sweep/run_mock.json --out harvest_trial/results/costing.json
```
