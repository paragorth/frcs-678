#!/usr/bin/env bash
# Push the whole working tree (code, manifests, configs, tests, docs) to the cluster. Run before every
# submission; a partial sync once cost a GPU slice (16 Oct 2026: manifests not copied).
#   bash cambridge/sync.sh [host]     (host defaults to the first live socket among csd3-icelake, csd3-cclake, csd3)
set -euo pipefail
cd "$(dirname "$0")/.."
host=${1:-}
if [ -z "$host" ]; then for h in csd3-icelake csd3-cclake csd3; do ssh -O check "$h" >/dev/null 2>&1 && { host=$h; break; }; done; fi
[ -n "$host" ] || { echo "no live cluster socket"; exit 1; }
rsync -az --delete-excluded --exclude '__pycache__' --exclude '.venv' --exclude 'runs' --exclude 'runs_invalid' --exclude 'results' --exclude '.git' \
  semataxy tools cambridge manifests conf tests docs CLAUDE.md requirements.txt "$host":rds/hpc-work/projects/frcs-678/
ssh -o BatchMode=yes "$host" 'cd ~/rds/hpc-work/projects/frcs-678 && source .venv/bin/activate && python -c "import semataxy.runner" && n=$(ls manifests/exp*/*.yaml | wc -l) && echo "synced to $(hostname): code imports, $n manifests"'
