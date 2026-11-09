cd /rds/user/pg630/hpc-work/projects/harvest-trial
V=/rds/user/pg630/hpc-work/venvs/harvestbench/bin/python; SRC=/rds/user/pg630/hpc-work/projects/harvestbench-src
echo "start $(date -u +%FT%TZ)"
python3 - <<'Y'
import json, collections
for fam in ("llama", "qwen"):
    rows = [json.loads(l) for l in open(f"out/open-{fam}/results.jsonl") if l.strip()]
    last = {}
    for r in rows: last[r["custom_id"]] = r
    c = collections.Counter((r["result"], r.get("finish")) for r in last.values())
    att = collections.Counter(r.get("attempt") for r in last.values())
    print(fam, "unique", len(last), "result/finish", dict(c), "attempts", dict(att))
    fails = [r for r in rows if r["result"] != "ok"]
    print("  failed rows (sent again):", len(fails), [r.get("error","")[:80] for r in fails[:3]])
Y
$V -I -B analyse_open.py --src $SRC --results out/open-qwen/results.jsonl out/open-llama/results.jsonl --out out/open-analysis/analysis.json > out/open-analysis/analysis.stdout 2>&1
echo "exit $? $(date -u +%FT%TZ)"
echo "ANALYSIS END"
