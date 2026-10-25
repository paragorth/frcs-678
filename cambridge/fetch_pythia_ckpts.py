"""Download Pythia's public pretraining snapshots for Paper 1 §5.2 at more sizes and seeds, then submit each
model's measurement (25 Oct; pg: "Yes — download the Pythia 410m / 1.4b / 2.8b training snapshots and the
four extra 160m seeds to the cluster, and delete each after it's measured").

Runs on the login node (compute nodes have no network). Per snapshot only config.json and the float16
pytorch_model.bin — identical to the float32 safetensors once upcast (checked on 410m step512: maximum
difference 0.0) at about half the size. After a model's 18 snapshots are in, submits its index of
cambridge/pythia_ckpts.sbatch, which measures them and then removes them (cambridge/drop_revisions.py).
The extra 160m seeds are PolyPythias (van der Wal et al. 2025): same data and recipe, new seed.

    nohup python cambridge/fetch_pythia_ckpts.py > runs/_slurm/fetch_pythia_ckpts.log 2>&1 &
"""
import argparse, subprocess, time

from huggingface_hub import snapshot_download

MODELS = ["EleutherAI/pythia-160m-seed1", "EleutherAI/pythia-160m-seed2", "EleutherAI/pythia-160m-seed3",
          "EleutherAI/pythia-160m-seed4", "EleutherAI/pythia-410m", "EleutherAI/pythia-1.4b", "EleutherAI/pythia-2.8b"]
STEPS = "0,64,128,256,512,1000,2000,3000,4000,5000,6000,8000,10000,15000,20000,40000,80000,143000"   # tools/pythia_checkpoints.py


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--only", default="", help="comma-separated indices into MODELS")
    a = ap.parse_args()
    only = {int(x) for x in a.only.split(",")} if a.only else set(range(len(MODELS)))
    for i, m in enumerate(MODELS):
        if i not in only:
            continue
        t = time.time()
        for s in STEPS.split(","):
            for attempt in range(3):
                try:
                    snapshot_download(m, revision=f"step{s}",
                                      allow_patterns=["config.json", "pytorch_model*.bin", "pytorch_model.bin.index.json"])
                    break
                except Exception as e:                  # a dropped connection: try again, then give up on the model
                    print(f"{m} step{s}: {type(e).__name__} {e}", flush=True)
                    time.sleep(30)
            else:
                print(f"{m}: step{s} failed three times; not submitted", flush=True)
                break
        else:
            print(f"{m}: 18 snapshots in {time.time() - t:.0f} s", flush=True)
            subprocess.run(["sbatch", f"--array={i}", "cambridge/pythia_ckpts.sbatch"], check=True)


if __name__ == "__main__":
    main()
