"""Joints between matrices, through interference. pg (22 Oct): building a capability is wiring a
circuit -- every wire has to be attached before it works -- and forgetting is pulling one wire. Every
measure so far looks INSIDE a matrix (k, Fourier concentration). At step 160 the capability is dead
while 79% of the circuit is still there, so if the analogy is right the break should be at a JOINT:
where one matrix writes into the residual stream and the next reads from it.

Every matrix touches the 128-wide residual stream on one side. Writers: tok (rows are tokens), o,
down. Readers: q, k, v, up, unembed. A joint is writer -> reader. Two measures per joint, both
data-free:

  align   overlap of the writer's above-edge stream directions with the reader's, against random
          subspaces of the same ranks (align_x = align / floor)
  k_rot   the wire itself: singular values of the path R @ W above the edge set by R @ Q @ W, with Q
          a random rotation of the stream. Q keeps the writer's own spectrum -- its INTERNAL
          organisation -- and destroys only how it lines up with the reader. So k_rot counts
          connection, not parts. k_shuf (writer's entries shuffled, pg's null) is logged beside it.

Intra-matrix k on the stream side is logged per matrix for comparison. --control trains phase 2 on
task A's own training split: same steps, same optimiser, no forgetting.

    python tools/joint_decay.py --seed 0 --out results/live_learning/joint_s0.csv
"""
import argparse, csv, glob, os, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from semataxy.structure import shuffled  # noqa: E402
from tools.haar_null import z_and_p, bh  # noqa: E402
from tools.sequential_tasks import task_data, acc  # noqa: E402
from tools.grokking import Model  # noqa: E402

# stream side of each matrix as a (128 x n) map: writers map into the stream, readers map out of it
WRITERS = {"tok": lambda sd: sd["tok.weight"].T, "o": lambda sd: sd["block.o.weight"],
           "down": lambda sd: sd["block.down.weight"]}
READERS = {"q": "block.q.weight", "k": "block.k.weight", "v": "block.v.weight",
           "up": "block.up.weight", "unembed": "unembed.weight"}
JOINTS = [("tok", "q"), ("tok", "k"), ("tok", "v"), ("tok", "up"), ("tok", "unembed"),
          ("o", "up"), ("o", "unembed"), ("down", "unembed")]


def edge_of(M, rng, n=4, null=shuffled):
    return max(np.linalg.svd(null(M, rng), compute_uv=False)[0] for _ in range(n))


def stream_basis(M, side, rng):
    """Above-edge directions on the stream side of M (128 x n for writers, n x 128 for readers)."""
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    keep = s > edge_of(M, rng)
    return (U[:, keep].T if side == "w" else Vt[keep]), int(keep.sum())


def overlap(A, B):
    if len(A) == 0 or len(B) == 0:
        return float("nan")
    return float(np.mean(np.linalg.svd(A @ B.T, compute_uv=False)[:min(len(A), len(B))]))


def rand_floor(ka, kb, d, rng, n=50):
    if not ka or not kb:
        return float("nan")
    return float(np.mean([overlap(np.linalg.qr(rng.standard_normal((d, ka)))[0].T,
                                  np.linalg.qr(rng.standard_normal((d, kb)))[0].T) for _ in range(n)]))


def rotate(W, rng):
    Q, _ = np.linalg.qr(rng.standard_normal((W.shape[0], W.shape[0])))
    return Q @ W


def toy_gate(sd, n=512, heads=4, seed=0):
    """The gate in the toy (24 Oct). The answer is read at '=' (position 2), which must gather a and
    b (positions 0, 1). Per head: attention from '=' to {a, b} vs to itself, and the gate margin
    log(a0 + a1) - log(a2). Computed from the weights on random (a, b) probes — no labels."""
    tok, pos = sd["tok.weight"], sd["pos.weight"]
    Wq, Wk = sd["block.q.weight"], sd["block.k.weight"]
    p = tok.shape[0] - 1; d = tok.shape[1]; hd = d // heads
    g = torch.Generator().manual_seed(seed)
    ab = torch.randint(0, p, (n, 2), generator=g)
    x = torch.cat([ab, torch.full((n, 1), p)], 1)
    h = torch.as_tensor(tok)[x] + torch.as_tensor(pos)[None]            # (n, 3, d)
    q = (h @ torch.as_tensor(Wq).T).view(n, 3, heads, hd)[:, 2]          # query at '='
    k = (h @ torch.as_tensor(Wk).T).view(n, 3, heads, hd)               # keys at all three
    att = torch.softmax(torch.einsum("nhd,nshd->nhs", q, k) / hd ** 0.5, -1)   # (n, heads, 3)
    out = {}
    for i in range(heads):
        a = att[:, i]
        out[f"gate_ab_h{i}"] = float((a[:, 0] + a[:, 1]).mean()); out[f"gate_self_h{i}"] = float(a[:, 2].mean())
        out[f"gate_margin_h{i}"] = float(((a[:, 0] + a[:, 1]).clamp_min(1e-12).log() - a[:, 2].clamp_min(1e-12).log()).mean())
        # the toy has no sink, so a gate would have to act on the a/b split: |log a0 - log a1|
        out[f"gate_balance_h{i}"] = float((a[:, 0].clamp_min(1e-12).log() - a[:, 1].clamp_min(1e-12).log()).abs().mean())
    out["gate_margin_mean"] = float(np.mean([out[f"gate_margin_h{i}"] for i in range(heads)]))
    return out


def measure(sd, seed):
    rng = np.random.default_rng(seed)
    sd = {k: v.detach().double().numpy() for k, v in sd.items()}
    Ws = {n: f(sd) for n, f in WRITERS.items()}
    Rs = {n: sd[m] for n, m in READERS.items()}
    row, bw, br = {}, {}, {}
    for n, W in Ws.items():
        bw[n], row[f"k_{n}"] = stream_basis(W, "w", rng)
    for n, R in Rs.items():
        br[n], row[f"k_{n}"] = stream_basis(R, "r", rng)
    for w, r in JOINTS:
        W, R = Ws[w], Rs[r]
        a = overlap(bw[w], br[r]); fl = rand_floor(len(bw[w]), len(br[r]), W.shape[0], rng)
        s = np.linalg.svd(R @ W, compute_uv=False)
        e_rot = max(np.linalg.svd(R @ rotate(W, rng), compute_uv=False)[0] for _ in range(4))
        e_shf = max(np.linalg.svd(R @ shuffled(W, rng), compute_uv=False)[0] for _ in range(4))
        j = f"{w}>{r}"
        # the validated measure, primary (pg, 22 Oct: "use something validated if it's there"):
        # Elhage et al. 2021's composition score, ||R W||_F / (||R||_F ||W||_F), with its value under
        # a random rotation of the stream as the baseline. Merullo et al. 2024 report it is noisy in
        # large models but informative in small ones like this.
        comp = float(np.linalg.norm(R @ W) / (np.linalg.norm(R) * np.linalg.norm(W)))
        comp_rot = float(np.mean([np.linalg.norm(R @ rotate(W, rng)) for _ in range(8)])
                         / (np.linalg.norm(R) * np.linalg.norm(W)))
        # Wang (2026)'s exact Haar null for C^2 (tools/haar_null.py): z-score and p, BH below.
        z, p = z_and_p(comp ** 2, np.linalg.svd(R, compute_uv=False), np.linalg.svd(W, compute_uv=False), W.shape[0])
        row.update({f"comp_{j}": comp, f"compx_{j}": comp / comp_rot, f"align_{j}": a, f"alignx_{j}": a / fl if fl and fl == fl else float("nan"),
                    f"krot_{j}": int((s > e_rot).sum()), f"kshuf_{j}": int((s > e_shf).sum()),
                    f"top_rot_{j}": float(s[0] / e_rot), f"z_{j}": z, f"p_{j}": p})
    sig = bh([row[f"p_{w}>{r}"] for w, r in JOINTS])
    for (w, r), s_ in zip(JOINTS, sig):
        row[f"sig_{w}>{r}"] = int(s_)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1", default="runs/grok_ckpt/ckpt"); ap.add_argument("--phase1-epoch", type=int, default=72000)
    ap.add_argument("--p", type=int, default=113)
    ap.add_argument("--frac", type=float, default=0.3); ap.add_argument("--frac-b", type=float, default=0.25)
    ap.add_argument("--task-b", default="mul")
    ap.add_argument("--control", action="store_true", help="phase 2 trains on task A's own training split")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--dense", action="store_true", help="also measure at every step 0-40")
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--wd", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cs = sorted(c for c in glob.glob(os.path.join(a.phase1, "ep*.pt")) if not c.endswith(".opt"))
    eps = [int(os.path.basename(c)[2:8]) for c in cs]
    i = min(range(len(eps)), key=lambda j: abs(eps[j] - a.phase1_epoch))
    if eps[i] != a.phase1_epoch:
        sys.exit(f"asked for epoch {a.phase1_epoch}, nearest is {eps[i]} -- refusing (see HANDBOOK §11)")
    sd0 = torch.load(cs[i], map_location="cpu")
    pa, la, tra, tea = task_data(a.p, "add", a.frac, a.seed)
    pb, lb, trb, teb = task_data(a.p, a.task_b, a.frac_b, a.seed + 1)
    xb, yb = (pa[tra], la[tra]) if a.control else (pb[trb], lb[trb])
    m = Model(a.p, d=sd0["block.q.weight"].shape[0]); m.load_state_dict(sd0)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=a.wd)
    # --dense: every step to 40, to settle whether the first wires go before, with or after the
    # behaviour (at step 20 A is already ~0.54).
    marks = set(list(range(0, 401, 20)) + list(range(500, a.steps + 1, 250))
                + (list(range(0, 41)) if a.dense else []))
    rows = []
    for step in range(a.steps + 1):
        if step in marks:
            m.eval()
            with torch.no_grad():
                gate = toy_gate({k_: v_.detach().double() for k_, v_ in m.state_dict().items()}, seed=a.seed)
            # displacement ||W_t - W_0|| / ||W_0|| (Frobenius): the clock Weyl puts on every spectral measure
            disp = {f"disp_{n}": float((p.detach() - sd0[n]).norm() / sd0[n].norm())
                    for n, p in m.named_parameters() if p.dim() == 2}
            r = dict(step=step, acc_a=acc(m, pa, la, tea), acc_b=acc(m, pb, lb, teb), **measure(m.state_dict(), a.seed), **gate, **disp)
            rows.append(r); m.train()
            print(f"{step:>5} A={r['acc_a']:.3f} B={r['acc_b']:.3f}  "
                  + "  ".join(f"{w}>{rr}:{r[f'compx_{w}>{rr}']:.2f}|{r[f'alignx_{w}>{rr}']:.2f}/{r[f'krot_{w}>{rr}']}" for w, rr in JOINTS), flush=True)
        if step == a.steps:
            break
        loss = F.cross_entropy(m(xb), yb); opt.zero_grad(); loss.backward(); opt.step()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
