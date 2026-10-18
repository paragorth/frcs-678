# Depth sweep — the nine mistakes, and what prevents each

Written from the actual failure record of attempts 1 and 2, not from memory.

| # | mistake | cost | prevention, now in code |
|---|---|---|---|
| 1 | took `seed_batch` default 32, not the reference's 6 | 4x slow, 71GB VRAM thrash | `SB=6` pinned in `depth_sweep.sh` |
| 2 | took `max_seq_len` default 256, not the reference's 192 | correctness AND speed | `ML=192` pinned; control must match reference VRAM 19.335GB |
| 3 | `--crosscheck 64` (full Jacobian) | cost more than the fit it checked | `XC=8` |
| 4 | `pkill` raced the sweep loop, which relaunched | 3 concurrent procs on one tag, corrupt ckpt | lockfile in `depth_sweep.sh` |
| 5 | `nohup bash -c '... \| grep ...' &` | pipeline defeated nohup; died with SSH; 9 idle hours, ~$11 | `setsid` + `</dev/null` + direct redirect in `run_depth.sh` |
| 6 | watcher checked process liveness | missed a 5h fit and 3 racing procs — all "alive" | watch the PROMPT COUNTER, alarm on stall |
| 7 | watcher treated "lens file exists" as done | crosscheck still running; started next fit on top | wait for process EXIT, not for a file |
| 8 | unquoted remote glob in rsync | local shell expanded it, pull silently failed | `pull_verify.sh`: quoted, and pulls a directory not a glob |
| 9 | destroyed the pod in the same block as the pull | **lost 8 depths, 8.4h, $10** | `pull_verify.sh` CANNOT destroy; teardown is separate, after reading |

## The order, and it is not negotiable

1. sweep process EXITS (not: last verdict written, not: lens file exists)
2. `scripts/pull_verify.sh d3 9` — pulls, counts, refuses on shortfall
3. **read and print every result**
4. only then `runpod_ctl down`

The single rule underneath all of it: **never destroy a source of data until a
verified copy exists somewhere else and I have looked at it.**
