# The registered analysis of step 1b on its results, on the cluster (the parser needs HarvestBench's source).
set -uo pipefail
cd /rds/user/pg630/hpc-work/projects/harvest-trial
V=/rds/user/pg630/hpc-work/venvs/harvestbench; SRC=/rds/user/pg630/hpc-work/projects/harvestbench-src
mkdir -p out/open-step1b-analysis
echo "left: $($V/bin/python -I -B run_open.py --family llama --labels step1b --mode left --out out/open-step1b-llama/results.jsonl)"
echo "rows: $(wc -l < out/open-step1b-llama/results.jsonl)"
$V/bin/python -I -B analyse_open.py --src $SRC --results out/open-step1b-llama/results.jsonl --out out/open-step1b-analysis/analysis.json > out/open-step1b-analysis/analysis.stdout 2>&1; echo "analysis exit $?"
cp "$0" out/open-step1b-analysis/run.sh
echo ANALYSIS DONE
