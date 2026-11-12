# HarvestBench feasibility trial on Muse Spark 1.2

Three paid shifts of HarvestBench's published task (one each at minimal, medium
and xhigh effort) to check that it runs on Meta's Muse Spark 1.2 and to cost
the six-cell experiment; then the four registered cells (minimal and xhigh
effort, neutral and morality briefings, 30 maps each); then the structure test
(the cells' hay-bale prompts sent again with the object's name changed). Design, exact settings
and the registrations: `docs/DESIGN_harvestbench_muse.md`. Results: `RESULTS.md`,
with logs in `logs/` (the cells in `logs/cells-39672076/`, the structure test
in `logs/structure-39710878/`, the Sonnet runs in `logs/sonnet-39718923/`) and
the costing in `results/`. HarvestBench itself (commit 526de705) is
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
| `draw_rocks.py` | Draws 30 logged rock prompts, unchanged, as Stage 2's check that a model reads the menu |
| `rock_templates.json` | Those 30 rock prompts |
| `run_sonnet.py` | Stage 2: the structure test on Sonnet 5 and 5.5 at low and max effort, through Anthropic's batch service (`--mode dry`, `mock` or `live`) |
| `analyse_sonnet.py` | Stage 2's registered analysis: each Sonnet its own replication of the structure test |
| `sonnet.sbatch` | The CSD3 job for Stage 2 (runs the analysis and the key check too) |
| `run_open.py` | The fine-tuning test, step 1 (registered 31 Oct): open models with care trained in, served by vLLM (`--mode dry`, `mock` or `live`; `--smoke N` for the smoke test on rock prompts; `--print-serve` for the job; `--fetch` for the weights, approved by pg on 31 Oct). `--labels step1b` for step 1b (registered 9 Nov): habitat caution or care for organisation, twelve other labels, Llama only |
| `analyse_open.py` | The fine-tuning test's registered analysis: the change each adapter makes to the life and made-organisation gaps against its control, tests Q1 to Q4 and L1 to L4 (`--smoke` for the smoke test); on step 1b's results, H1 (inert made organisation) and H2 (life in a pot), the leaf pile, the hay anchor and the words in the messages |
| `count_training_words.py` | Step 1b's side count: how often AuditBench's animal-welfare documents and transcripts mention habitat, ecosystems, plants and the rest (the two parquet files downloaded with pg's yes, 9 Nov; counts in `logs/open-step1b/training-words/`) |
| `open.sbatch` | One one-hour GPU slice of the fine-tuning test (interactive queue); with `SMOKE=10` the smoke test; `TP`, `MAX_SEQS` and `PARALLEL` for more GPUs; `SET=step1b` for step 1b |
| `open_chain.sh` | Submits the slices one after another from the login node until every request is answered (the interactive queue takes one job per user); `SET=step1b` for step 1b |
| `run_pair.py` | The pair test, one training at two sizes through OpenRouter at the lowest reasoning the models allow, one host pinned: `--pair deepseek` (registered 11 Nov: V4 Pro against V4 Flash, thinking off, run 12 Nov) or `--pair gptoss` (the replication, drafted 12 Nov: 120b against 20b at reasoning low); `--mode dry`, `mock` or `live`; `--limit N` for the smoke test; `--saved DIR` for an offline dry run |
| `analyse_pair.py` | The pair test's registered analysis: gates; the "pair" family (H1 life and H2 bale-and-bundle gaps, big minus small) or the "replication" family (R1 colony, R2 poppy, R3 rabbit); each model alone, appear or grow, every label's difference with its p (exploratory) |
| `pair.sbatch` | The CSD3 job for a pair (`PAIR=deepseek|gptoss`; `LIMIT=N` for the smoke test; `HB_PYTHON=/usr/bin/python3.11` to run without the venv; runs the analysis and the key scrub too) |
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
$V/bin/python -I -B draw_rocks.py --log-dir out/cells-39672076 --out rock_templates.json
$V/bin/python -I -B run_sonnet.py --mode mock --templates structure_templates.json --rocks rock_templates.json --out-dir out/sonnet-mock
sbatch --export=ALL,HB_SRC=$SRC,HB_VENV=$V,ANTHROPIC_VENV=/rds/user/pg630/hpc-work/semataxy/.venv-gpu,POT=150 sonnet.sbatch
$V/bin/python -I -B run_open.py --family qwen --mode mock --policy planted --out out/open-mock/qwen-planted.jsonl
$V/bin/python -I -B analyse_open.py --src $SRC --results out/open-mock/qwen-planted.jsonl out/open-mock/llama-planted.jsonl --out out/open-mock/analysis-planted.json
HF_HOME=$HOME/rds/hpc-work/hf $HOME/rds/hpc-work/semataxy/.venv-gpu/bin/python -I -B run_open.py --family qwen --fetch   # approved by pg, 31 Oct
sbatch --export=ALL,FAMILY=qwen,OUTDIR=out/open-qwen,SMOKE=10 open.sbatch   # the smoke test, rock prompts only
$V/bin/python -I -B analyse_open.py --src $SRC --smoke --results out/open-qwen/smoke.jsonl --out out/open-qwen/smoke-report.json
nohup bash open_chain.sh qwen:2 llama > out/open-chain.log 2>&1 < /dev/null &   # every slice, Qwen on two GPUs
# Step 1b (registered 9 Nov): the smoke test, then the slices, then the analysis
$V/bin/python -I -B run_open.py --family llama --labels step1b --mode mock --policy planted --out out/open-mock/llama-step1b-planted.jsonl
$V/bin/python -I -B analyse_open.py --src $SRC --results out/open-mock/llama-step1b-planted.jsonl --out out/open-mock/analysis-step1b-planted.json
sbatch --export=ALL,FAMILY=llama,OUTDIR=out/open-step1b-llama,SET=step1b,SMOKE=10 open.sbatch
SET=step1b nohup bash open_chain.sh llama > out/open-step1b-chain.log 2>&1 < /dev/null &
$V/bin/python -I -B analyse_open.py --src $SRC --results out/open-step1b-llama/results.jsonl --out out/open-step1b-analysis/analysis.json
# The pair test (registered 11 Nov): dry, the smoke test, then the run
$V/bin/python -I -B run_pair.py --mode dry --templates structure_templates.json --rocks rock_templates.json --out-dir out/pair-dry
sbatch --export=ALL,HB_SRC=$SRC,HB_VENV=$V,POT=0.50,LIMIT=12 pair.sbatch
sbatch --export=ALL,HB_SRC=$SRC,HB_VENV=$V,POT=4.50 pair.sbatch
# As run on 12 Nov, from a home copy on the system python because the venv on RDS hung (~/harvest-pair):
sbatch --exclude=ws-s-51 --export=ALL,HB_SRC=$SRC,HB_PYTHON=/usr/bin/python3.11,POT=0.50,LIMIT=12 pair.sbatch
sbatch --exclude=ws-s-51 --dependency=afterok:<smoke job> --export=ALL,HB_SRC=$SRC,HB_PYTHON=/usr/bin/python3.11,POT=4.50 pair.sbatch
# The replication on gpt-oss (drafted 12 Nov): the same, with PAIR=gptoss and pg's cap
sbatch --exclude=ws-s-51 --export=ALL,HB_SRC=$SRC,HB_PYTHON=/usr/bin/python3.11,PAIR=gptoss,POT=0.30,LIMIT=12 pair.sbatch
```

On the laptop, from the repository root:

```bash
python3 -I -B harvest_trial/cost_full.py --summary harvest_trial/logs/live-39667753/trial_summary.json --ledger harvest_trial/logs/live-39667753/spend_ledger_live.jsonl --sweep harvest_trial/logs/sweep/run_mock.json --out harvest_trial/results/costing.json
```
