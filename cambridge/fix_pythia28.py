"""Pythia-2.8b's hub snapshots carry stale single-file weights (25 Oct). Steps 0-80 000 all list the same
model.safetensors, and steps 0-20 000 the same pytorch_model.bin, while the two-part sharded safetensors of
steps 0-20 000 differ from step to step, and steps 40 000 / 80 000 / 143 000 each have their own .bin.
`from_pretrained(revision="step512")` online fetches the single model.safetensors first, so it returns the
stale file. For each snapshot this keeps only the file that is unique to it: downloads the shards and their
index for steps 0-20 000 and removes the stale .bin from those snapshot folders (the shared blob too); the
.bin of 40 000 / 80 000 / 143 000 stays. Then checks that step0 loads offline from the shards and asks
whether the stale model.safetensors is `main`'s file (the final model). Login node (it downloads).

    python cambridge/fix_pythia28.py
"""
import os, subprocess
import torch
from huggingface_hub import constants, snapshot_download

REPO = "EleutherAI/pythia-2.8b"
SHARDED = [0, 64, 128, 256, 512, 1000, 2000, 3000, 4000, 5000, 6000, 8000, 10000, 15000, 20000]
root = os.path.join(constants.HF_HUB_CACHE, "models--" + REPO.replace("/", "--"))
snap = lambda rev: os.path.join(root, "snapshots", open(os.path.join(root, "refs", rev)).read().strip())

stale = set()
for s in SHARDED:
    snapshot_download(REPO, revision=f"step{s}", allow_patterns=["model.safetensors.index.json", "model-0000*-of-00002.safetensors"])
    p = os.path.join(snap(f"step{s}"), "pytorch_model.bin")
    if os.path.islink(p):
        stale.add(os.path.normpath(os.path.join(os.path.dirname(p), os.readlink(p)))); os.remove(p)
    print(f"step{s}: shards in; {sorted(os.listdir(snap(f'step{s}')))}", flush=True)
for b in stale:                          # the stale .bin, now linked from no snapshot
    still = [d for d in os.listdir(os.path.join(root, "snapshots"))
             if os.path.islink(os.path.join(root, "snapshots", d, "pytorch_model.bin"))
             and os.path.normpath(os.path.join(root, "snapshots", d, os.readlink(os.path.join(root, "snapshots", d, "pytorch_model.bin")))) == b]
    if not still:
        print(f"removing stale blob {os.path.basename(b)[:12]} ({os.path.getsize(b) / 1e9:.2f} GB)", flush=True); os.remove(b)

# step0 must load from the shards, offline, and match them exactly
code = f"""
import torch, json, os
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM
d = {snap("step0")!r}
m = AutoModelForCausalLM.from_pretrained({REPO!r}, revision="step0", dtype=torch.float32).state_dict()
sh = {{}}
for f in sorted(set(json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"].values())):
    sh.update(load_file(os.path.join(d, f)))
print("step0 offline load = its shards:", max(float((m[k] - sh[k].float()).abs().max()) for k in sh if k in m) == 0.0)
from huggingface_hub import hf_hub_download
main = load_file(hf_hub_download({REPO!r}, "model.safetensors", revision="main", local_files_only=True))
fin = torch.load(os.path.join({snap("step143000")!r}, "pytorch_model.bin"), map_location="cpu", weights_only=True)
common = [k for k in main if k in fin and main[k].is_floating_point()]
print("stale model.safetensors (= main's) vs step143000's .bin: max diff", max(float((main[k].float() - fin[k].float()).abs().max()) for k in common), "over", len(common), "tensors")
print("step0 shards vs main: max diff", max(float((sh[k].float() - main[k].float()).abs().max()) for k in sh if k in main))
"""
subprocess.run(["python", "-c", code], env={**os.environ, "HF_HUB_OFFLINE": "1"}, check=True)
