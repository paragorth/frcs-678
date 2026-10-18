#!/bin/bash
# Launch the depth sweep detached so it survives the SSH session that starts it.
# The previous attempt used `nohup bash -c '... | grep ...' &`; the pipeline did
# not inherit nohup's protection, the job died with the session, and the pod sat
# idle for nine hours. setsid + explicit redirection + </dev/null fixes that.
set -u
cd /root/jlens
export JLENS_OUT=/workspace/out HF_HOME=/workspace/hf
MODE=${1:-control}
if [ "$MODE" = "control" ]; then
  setsid /usr/bin/python3 -u scripts/fit_lens.py --model 7b --inject 7 --tag _d7 \
    --n-prompts "${NP:-400}" --seed-batch 6 --max-seq-len 192 --crosscheck 8 \
    </dev/null >/workspace/out/control.log 2>&1 &
else
  setsid env NP="${NP:-400}" SB=6 XC=8 ML=192 bash scripts/depth_sweep.sh \
    </dev/null >/workspace/out/depth.log 2>&1 &
fi
disown
sleep 3
echo "launched $MODE, pid $(pgrep -f '[f]it_lens.py|[d]epth_sweep.sh' | head -1)"
