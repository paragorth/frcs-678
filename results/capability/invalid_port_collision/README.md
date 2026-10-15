Set aside 15 Oct 2026. These three capability-benchmark jobs ran on a node shared with another
benchmark job; both used 127.0.0.1:8020, the second server failed to bind, and the client scored
whichever model the neighbour was serving:

| file | what it was labelled | what it actually scored | evidence |
|---|---|---|---|
| q14b | Qwen2.5-14B (0.970) | Qwen2.5-32B | job 38329293 shared cpu-q-141 with cap-q32b; its server log says "couldn't bind" |
| l8b  | Llama-3.1-8B (0.901) | Gemma-2-9B | job 38329295 shared cpu-q-142 with cap-g9b; bind failure |
| q3b  | Qwen2.5-3B (0.606) | Qwen2.5-7B | job 38329291 shared cpu-q-171 with cap-q7b; bind failure (note the identical 0.606) |

Valid scores: q32b 0.955, g9b 0.894, q7b 0.606, q1.5b 0.121. The three are re-run with the per-job
address fix in cambridge/capability_bench.sbatch. Any claim resting on "14B and 32B are tied on
arithmetic" or "Llama 8B scores 0.90" is withdrawn until the re-run lands.
