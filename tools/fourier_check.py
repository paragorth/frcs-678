"""Do the null-selected directions correspond to the circuit's key Fourier frequencies?

Everything measured so far says the above-edge directions are load-bearing (transplanting them
installs the function; transplanting the rest installs nothing) but not what they *are*. That is the
difference between representational geometry and interpretability. Nanda et al. (2023) show the
grokked modular-addition circuit computes trigonometric identities on a small number of key
frequencies, found by taking the discrete Fourier transform over the p token indices. Their
criterion needs to know the task's symmetry group; ours needs only the weight matrix.

So this asks whether the two agree. For the unembedding (p x d), the LEFT singular vectors live in
token space, so they can be transformed over the p indices directly. If the above-edge left
directions concentrate their power on a handful of frequencies -- and the below-edge ones do not --
then a shuffled-entry null has located the same object Nanda's task-specific analysis does, and the
subspace can be described rather than merely shown to matter.

Reported per above-edge direction: the fraction of its Fourier power in its single strongest
frequency, and how many frequencies are needed to reach 80% of its power. For comparison, the same
for below-edge directions and for random vectors.

    python tools/fourier_check.py --ckpt runs/grok_ckpt/ckpt --epoch 72000
"""
from __future__ import annotations
import argparse, glob, os, sys
import numpy as np
import torch

sys.path.insert(0, "/Users/pg/frcs-678")
from semataxy.structure import shuffled, spectrum  # noqa: E402


def split_sides(W, seed, n_shuffles=4, tol=1.02):
    """Left singular vectors, split by the permutation-null edge. W is (p x d) for the unembedding,
    so the left vectors are indexed by token and can be Fourier-transformed over p."""
    W64 = np.asarray(W, np.float64)
    U, s, Vt = np.linalg.svd(W64, full_matrices=False)
    lam = s ** 2 / max(W64.shape)
    rng = np.random.default_rng(seed)
    edge = max(spectrum(shuffled(W64, rng))[0] for _ in range(n_shuffles)) * tol
    keep = lam > edge
    return U[:, keep].T, U[:, ~keep].T, int(keep.sum())


def fourier_profile(v):
    """(share of power in the strongest frequency, frequencies needed for 80% of power)."""
    f = np.abs(np.fft.rfft(v - v.mean())) ** 2
    if f.sum() <= 0:
        return float("nan"), float("nan")
    p = np.sort(f)[::-1] / f.sum()
    return float(p[0]), int(np.searchsorted(np.cumsum(p), 0.80) + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/Users/pg/frcs-678/runs/grok_ckpt/ckpt")
    ap.add_argument("--epoch", type=int, default=72000)
    ap.add_argument("--matrix", default="unembed.weight")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    cs = sorted(c for c in glob.glob(os.path.join(a.ckpt, "ep*.pt")) if not c.endswith(".opt"))
    eps = [int(os.path.basename(c)[2:8]) for c in cs]
    i = min(range(len(eps)), key=lambda j: abs(eps[j] - a.epoch))
    sd = torch.load(cs[i], map_location="cpu")
    W = sd[a.matrix].numpy()
    print(f"{a.matrix} at epoch {eps[i]}, shape {W.shape}")

    above, below, k = split_sides(W, a.seed)
    p = W.shape[0]
    print(f"{k} above-edge directions of {min(W.shape)}; left vectors live in {p}-dim token space\n")

    rng = np.random.default_rng(a.seed)
    rand = rng.standard_normal((max(k, 8), p))
    rand /= np.linalg.norm(rand, axis=1, keepdims=True)

    def summarise(label, M, show=0):
        if len(M) == 0:
            return print(f"{label:16} (none)")
        tops, n80 = zip(*[fourier_profile(v) for v in M])
        print(f"{label:16} n={len(M):>3}  top-freq power {np.mean(tops):.3f} "
              f"(max {np.max(tops):.3f})   freqs for 80%: {np.mean(n80):.1f}")
        for j in range(min(show, len(M))):
            f = np.abs(np.fft.rfft(M[j] - M[j].mean())) ** 2
            order = np.argsort(f)[::-1][:3]
            print(f"    dir {j}: strongest frequencies {list(order)} "
                  f"carrying {f[order].sum()/f.sum():.3f} of its power")

    print(f"{'set':16}{'':>6}{'concentration in Fourier space':>34}")
    summarise("above-edge", above, show=6)
    summarise("below-edge", below)
    summarise("random", rand)

    ta = np.mean([fourier_profile(v)[0] for v in above]) if len(above) else float("nan")
    tb = np.mean([fourier_profile(v)[0] for v in below]) if len(below) else float("nan")
    tr = np.mean([fourier_profile(v)[0] for v in rand])
    print(f"\nabove-edge directions are {ta/tr:.1f}x more Fourier-concentrated than random, "
          f"and {ta/tb:.1f}x more than below-edge")
    print("-> the null-selected subspace IS the Fourier circuit" if ta > 3 * tr and ta > 2 * tb else
          "-> no clear correspondence: the null's subspace is not simply the key-frequency subspace")


if __name__ == "__main__":
    main()
