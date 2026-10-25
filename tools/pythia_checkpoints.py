"""The building half at real scale: watch Pythia's induction circuit form during its own pretraining.

Grokking's large-model analogue is a circuit forming abruptly — induction heads appear in a phase
change early in pretraining (Olsson et al. 2022). Pythia publishes its pretraining checkpoints, and
during pretraining the weights move far, so Weyl's bound does not pin the spectral measures the way it
does under a 0.3% fine-tune. Paper 1's spine predicts the circuit's joints (previous-token head ->
induction head keys) strengthen BEFORE induction behaviour appears, as the toy's circuit was 7.96x
random at 14% accuracy.

Heads are identified once, at the final checkpoint; at every checkpoint: each of their induction /
previous-token attention scores, induction accuracy, the K-composition joints (exact Haar null z, wire
count vs rotation), embedding -> keys and output -> unembedding for the induction heads, and each
head's own K / O shuffle-null count. No training; checkpoints only.

    python tools/pythia_checkpoints.py --out results/live_learning/pythia160m_pretraining_induction.csv

Other families (25 Oct, OLMo-7B-0724's nine cached snapshots): --revs takes revision names as the hub
lists them (step1000-tokens4B, ...), --final names the last one, and the parts come from tools/arch.py.
--stream computes the weight distances one tensor at a time from the snapshot files, so a 7B model in
float32 needs one copy in memory, not four.

    python tools/pythia_checkpoints.py --model allenai/OLMo-7B-0724-hf --final step650650-tokens2729B \
        --revs step0-tokens0B,step1000-tokens4B,... --stream --out results/live_learning/olmo7b_pretraining_induction.csv
"""
import argparse, csv, json, os, re, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.pythia_joints import rotations, joint, _dt  # noqa: E402
from tools.arch import Arch, induction_parts  # noqa: E402
from tools.pythia_induction import repeated, head_scores, induction_acc  # noqa: E402
from tools.haar_null import bh  # noqa: E402

DEFAULT_STEPS = "0,64,128,256,512,1000,2000,3000,4000,5000,6000,8000,10000,15000,20000,40000,80000,143000"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-160m")
    ap.add_argument("--steps", default=DEFAULT_STEPS)
    ap.add_argument("--revs", default="", help="revision names, overriding --steps (step number read from each)")
    ap.add_argument("--final", default="step143000")
    ap.add_argument("--stream", action="store_true", help="weight distances from the files, one tensor at a time")
    ap.add_argument("--T", type=int, default=40); ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM
    load = lambda rev: AutoModelForCausalLM.from_pretrained(a.model, revision=rev, dtype=torch.float32,
                                                            attn_implementation="eager").eval()
    seqs = repeated(64, a.T, 1000, 20000, torch.Generator().manual_seed(1234))
    final = load(a.final)
    A = Arch(final)
    L, H, d = A.L, A.H, A.d
    ind, prev = head_scores(final, seqs, a.T)
    flat = lambda s: [(int(i) // H, int(i) % H) for i in torch.argsort(s.flatten(), descending=True)]
    IH = flat(ind)[:a.n_heads]
    PH = [p for p in flat(prev) if p[0] < min(l for l, _ in IH)][:a.n_heads]
    print("induction heads (final):", IH, " previous-token heads:", PH, flush=True)
    # pg (24 Oct): weight movement should shrink as pretraining goes on (learning-rate decay, growing
    # norms) and the model settle gradually into the state we fine-tune. Logged per checkpoint:
    # movement since the previous checkpoint per training step, and distance from the final model.
    if A.family == "neox":
        GROUPS = {"embed": ["gpt_neox.embed_in.weight"], "unembed": ["embed_out.weight"],
                  "attn_qkv": [f"gpt_neox.layers.{l}.attention.query_key_value.weight" for l in range(L)],
                  "attn_out": [f"gpt_neox.layers.{l}.attention.dense.weight" for l in range(L)],
                  "mlp_in": [f"gpt_neox.layers.{l}.mlp.dense_h_to_4h.weight" for l in range(L)],
                  "mlp_out": [f"gpt_neox.layers.{l}.mlp.dense_4h_to_h.weight" for l in range(L)]}
    else:
        lay = lambda *ns: [f"model.layers.{l}.{n}.weight" for l in range(L) for n in ns]
        GROUPS = {"embed": ["model.embed_tokens.weight"], "unembed": ["lm_head.weight"],
                  "attn_qkv": lay("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
                  "attn_out": lay("self_attn.o_proj"), "mlp_in": lay("mlp.gate_proj", "mlp.up_proj"),
                  "mlp_out": lay("mlp.down_proj")}
    sd_key = lambda sd, k: sd[k] if k in sd else sd["lm_head.weight"]            # embed_out vs lm_head
    # references, not copies (25 Oct): nothing is trained, so a kept reference is the checkpoint itself, and
    # Pythia-2.8b in float32 then needs three copies in memory, not five (a desktop node has 63 GB)
    snap = lambda sd: {g: [sd_key(sd, k).detach() for k in ks] for g, ks in GROUPS.items()}
    rel = lambda A, B: float(torch.sqrt(sum(((a - b) ** 2).sum() for a, b in zip(A, B))) /
                             torch.sqrt(sum((b ** 2).sum() for b in B)))

    def files(rev):
        """{tensor name: safetensors file} for one cached revision. The snapshot folder is read from the
        cache's refs directly: snapshot_download(local_files_only=True) refuses a snapshot that lacks the
        repo's README or .gitattributes (hub 2.2), which a weights-only download never fetches."""
        from huggingface_hub import constants
        from safetensors import safe_open
        repo = os.path.join(constants.HF_HUB_CACHE, "models--" + a.model.replace("/", "--"))
        ref = os.path.join(repo, "refs", rev)
        root = os.path.join(repo, "snapshots", open(ref).read().strip() if os.path.exists(ref) else rev)
        idx = os.path.join(root, "model.safetensors.index.json")
        if os.path.exists(idx):
            return {k: os.path.join(root, f) for k, f in json.load(open(idx))["weight_map"].items()}
        f = os.path.join(root, "model.safetensors")
        with safe_open(f, "pt") as h:
            return {k: f for k in h.keys()}

    def rel_files(fa, fb, keys):
        from safetensors import safe_open
        num = den = 0.0
        for k in keys:
            with safe_open(fa[k], "pt") as ha, safe_open(fb[k], "pt") as hb:
                x, y = ha.get_tensor(k).double(), hb.get_tensor(k).double()
            num += float(((x - y) ** 2).sum()); den += float((y ** 2).sum())
        return num ** 0.5 / den ** 0.5

    if a.stream:
        fF = files(a.final)
    else:
        sdF = snap(final.state_dict())
    del final
    prev_snap, prev_step, prev_files = None, None, None
    joints = [(f"O{pl}h{ph}", f"K{il}h{ih}") for pl, ph in PH for il, ih in IH] \
           + [("E", f"K{il}h{ih}") for il, ih in IH] + [(f"O{il}h{ih}", "U") for il, ih in IH]
    rows = []
    revs = a.revs.split(",") if a.revs else [f"step{x}" for x in a.steps.split(",")]
    for rev in revs:
        st = int(re.match(r"step(\d+)", rev).group(1))
        m = load(rev)
        with torch.no_grad():
            out = m(seqs, output_attentions=True)          # one forward for both readings (25 Oct: 7B on 4 cores)
        ind_t, prev_t = head_scores(m, seqs, a.T, out=out)
        acc, _ = induction_acc(m, seqs, a.T, logits=out.logits)
        del out
        parts = induction_parts(m.state_dict(), A, IH + PH, "cpu")       # = stream_parts(per_head) for Pythia
        Qs = rotations(d, torch.Generator().manual_seed(0), _dt("cpu"), "cpu")
        r = dict(step=st, rev=rev, induction_acc=acc, max_induction_score=float(ind_t.max()), max_prev_score=float(prev_t.max()))
        for l, h in IH:
            r[f"indscore_L{l}H{h}"] = float(ind_t[l, h])
        for l, h in PH:
            r[f"prevscore_L{l}H{h}"] = float(prev_t[l, h])
        for w, rr in joints:
            j = joint(parts, w, rr, Qs)
            r[f"z_{w}>{rr}"] = j["z"]; r[f"p_{w}>{rr}"] = j["p"]; r[f"krot_{w}>{rr}"] = j["krot"]
        for (w, rr), s in zip(joints, bh([r[f"p_{w}>{rr}"] for w, rr in joints])):
            r[f"sig_{w}>{rr}"] = int(s)
        for l, h in IH + PH:
            r[f"k_K{l}h{h}"] = parts[f"k_K{l}h{h}"]; r[f"k_O{l}h{h}"] = parts[f"k_O{l}h{h}"]
        if a.stream:
            fC = files(rev)
            for g, ks in GROUPS.items():
                r[f"dist_final_{g}"] = rel_files(fC, fF, ks)
                if prev_files is not None:
                    r[f"move_per_step_{g}"] = rel_files(fC, prev_files, ks) / (st - prev_step)
            prev_files, prev_step = fC, st
        else:
            cur = snap(m.state_dict())
            for g in GROUPS:
                r[f"dist_final_{g}"] = rel(cur[g], sdF[g])
                if prev_snap is not None:
                    r[f"move_per_step_{g}"] = rel(cur[g], prev_snap[g]) / (st - prev_step)
            prev_snap, prev_step = cur, st
        rows.append(r)
        with open(a.out, "w", newline="") as fh:
            # union of keys: the first checkpoint has no move_per_step_* (HANDBOOK §11, rows[0] bug)
            keys = list(dict.fromkeys(k for row in rows for k in row))
            w = csv.DictWriter(fh, fieldnames=keys, restval=""); w.writeheader(); w.writerows(rows)
        kc = [r[f"z_O{pl}h{ph}>K{il}h{ih}"] for pl, ph in PH for il, ih in IH]
        print(f"step {st:>6}  induction acc {acc:.3f}  ind-scores " + " ".join(f"{r[f'indscore_L{l}H{h}']:.2f}" for l, h in IH)
              + f"  | K-comp z mean {np.mean(kc):7.1f} max {np.max(kc):7.1f}  sig {sum(r[f'sig_{w}>{rr}'] for w, rr in joints[:len(PH)*len(IH)])}/{len(PH)*len(IH)}"
              + f"  | from final: qkv {100*r['dist_final_attn_qkv']:.1f}%  mlp_out {100*r['dist_final_mlp_out']:.1f}%"
              + (f"  per-step move qkv {100*r['move_per_step_attn_qkv']:.4f}%" if "move_per_step_attn_qkv" in r else ""), flush=True)
        del m
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
