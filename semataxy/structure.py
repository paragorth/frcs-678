"""Structural Semataxy of a weight matrix: organisation across scales.

For a matrix W (n x m) form the correlation spectrum lambda_i = s_i^2 / n from
its singular values. The null is the *same entries shuffled*: a random matrix
with identical value distribution, whose spectrum is the Marchenko-Pastur bulk.
Everything above the shuffled spectrum's edge is relational structure that
permutation destroys and entropy cannot see.

Sigma_struct = number of consecutive populated octaves above the bulk edge
             = over how many doublings of scale the structure is continuous.

  noise    -> nothing above the edge           -> 0
  crystal  -> a few isolated spikes, a gap     -> 0 or 1
  trained  -> continuous power-law tail        -> several

Also reported: fraction of eigenvalues and of spectral mass above the edge, and
a Hill estimate of the tail exponent alpha (quality of the organisation).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class SpectrumStats:
    name: str
    shape: tuple
    n_eigs: int
    bulk_edge: float
    lam_max: float
    n_above: int
    frac_above: float
    mass_above: float
    octaves_total: float  # log2(lam_max / edge), 0 if nothing above
    sigma_struct: int  # consecutive populated octaves from the edge
    alpha: float  # Hill tail exponent on the top eigenvalues (nan if too few)
    # Shape of the organisation, as opposed to how much of it there is (20 Oct 2026). sigma_struct,
    # n_above and frac_above are all *width*: how far the above-edge region extends. Parag's point is
    # that semataxy is a claim about shape, and that within one space shape is all that varies -- so
    # width only ever adds information across differently sized spaces, where it is confounded with
    # size. These three are dimensionless and invariant to how many directions clear the edge.
    alpha_above: float  # Hill exponent with xmin = the null edge, over all above-edge eigenvalues
    entropy_above: float  # Shannon entropy of the above-edge mass / log(k): 1 = flat, 0 = one spike
    gini_above: float  # Gini of the above-edge eigenvalues: 0 = flat, 1 = one spike


def spectrum(W: np.ndarray) -> np.ndarray:
    W = np.asarray(W, dtype=np.float64)
    if W.shape[0] < W.shape[1]:
        W = W.T
    n = W.shape[0]
    s = np.linalg.svd(W, compute_uv=False)
    return np.sort(s**2 / n)[::-1]


def shuffled(W: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    flat = np.asarray(W).ravel().copy()
    rng.shuffle(flat)
    return flat.reshape(np.asarray(W).shape)


def hill_alpha(lam: np.ndarray, k: int) -> float:
    """Hill estimator of the power-law exponent of the top-k eigenvalues."""
    lam = np.sort(lam)[::-1]
    if k < 5 or len(lam) <= k:
        return float("nan")
    top, ref = lam[:k], lam[k]
    if ref <= 0:
        return float("nan")
    return float(1.0 + k / np.sum(np.log(top / ref)))


def alpha_above_edge(above: np.ndarray, edge: float) -> float:
    """Hill exponent of the above-edge eigenvalues with xmin fixed to the permutation-null edge.

    This is the null-native version of the tail exponent: `hill_alpha` picks its cut-off as
    max(5, min(n_above, n/10)) and references lam[k], so the fitted region is chosen partly by an
    arbitrary rule. Here the null chooses it -- xmin is the edge, and every eigenvalue above it is
    used. Lower alpha = heavier tail = mass spread further above the edge.
    """
    above = np.asarray(above, dtype=np.float64)
    if len(above) < 2 or edge <= 0:
        return float("nan")
    s = float(np.sum(np.log(above / edge)))
    if not s > 1e-12:  # every eigenvalue sits essentially on the edge
        return float("nan")
    return float(1.0 + len(above) / s)


def mass_shape(above: np.ndarray) -> tuple[float, float]:
    """(normalised entropy, Gini) of the above-edge mass: how that mass is spread over its own
    directions, independent of how many there are. Both are invariant under scaling the matrix and
    under duplicating the profile, so they separate shape from width."""
    above = np.asarray(above, dtype=np.float64)
    k = len(above)
    if k < 2 or above.sum() <= 0:
        return float("nan"), float("nan")
    p = above / above.sum()
    p = p[p > 0]
    ent = float(-np.sum(p * np.log(p)) / math.log(k))
    x = np.sort(above)
    gini = float((2.0 * np.sum((np.arange(1, k + 1)) * x) / (k * x.sum())) - (k + 1.0) / k)
    return ent, gini


def analyze_matrix(name: str, W: np.ndarray, rng: np.random.Generator, min_per_octave: int = 2,
                   n_shuffles: int = 1) -> SpectrumStats:
    lam = spectrum(W)
    edge = 0.0
    for _ in range(n_shuffles):
        edge = max(edge, spectrum(shuffled(W, rng))[0])
    edge *= 1.02  # tolerance for the finite-size fluctuation of the random edge
    above = lam[lam > edge]
    n_above = int(len(above))
    frac_above = n_above / len(lam)
    mass_above = float(above.sum() / lam.sum()) if lam.sum() > 0 else 0.0
    lam_max = float(lam[0])
    octaves_total = float(math.log2(lam_max / edge)) if n_above else 0.0

    # consecutive populated octaves starting at the edge
    sigma = 0
    if n_above:
        k = 0
        while True:
            lo, hi = edge * 2**k, edge * 2 ** (k + 1)
            count = int(np.sum((lam > lo) & (lam <= hi)))
            if count >= min_per_octave:
                sigma += 1
                k += 1
            else:
                break
    alpha = hill_alpha(lam, k=max(5, min(n_above, len(lam) // 10)))
    a_above = alpha_above_edge(above, float(edge))
    ent_above, gini_above = mass_shape(above)
    return SpectrumStats(name, tuple(np.asarray(W).shape), len(lam), float(edge), lam_max, n_above,
                         float(frac_above), mass_above, octaves_total, sigma, alpha,
                         a_above, ent_above, gini_above)


def synthetic(kind: str, shape=(896, 896), rng=None) -> np.ndarray:
    """Reference matrices for the noise / crystal regimes."""
    rng = rng or np.random.default_rng(0)
    n, m = shape
    if kind == "noise":
        return rng.normal(0, 0.02, size=shape)
    if kind == "crystal":  # periodic structure: a few spikes, nothing in between
        r = 3
        u = np.stack([np.sin(2 * np.pi * (i + 1) * np.arange(n) / n) for i in range(r)], 1)
        v = np.stack([np.cos(2 * np.pi * (i + 1) * np.arange(m) / m) for i in range(r)], 1)
        return (u @ v.T) * 0.5 + rng.normal(0, 0.02, size=shape)
    raise ValueError(kind)


def model_matrices(model_name: str, min_dim: int = 64):
    """Yield (name, 2-D weight) for every linear layer of a Hugging Face model."""
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    for name, p in model.named_parameters():
        if p.ndim == 2 and min(p.shape) >= min_dim and "embed" not in name and "lm_head" not in name:
            yield name, p.detach().cpu().numpy()


def analyze_model(model_name: str, conditions=("trained", "shuffled", "init"), seed: int = 0,
                  max_layers: int | None = None):
    rng = np.random.default_rng(seed)
    rows = []
    for i, (name, W) in enumerate(model_matrices(model_name)):
        if max_layers is not None and i >= max_layers:
            break
        variants = {}
        if "trained" in conditions:
            variants["trained"] = W
        if "shuffled" in conditions:
            variants["shuffled"] = shuffled(W, rng)
        if "init" in conditions:  # what descent started from: Gaussian with the trained std
            variants["init"] = rng.normal(0, W.std(), size=W.shape)
        for cond, M in variants.items():
            st = analyze_matrix(name, M, rng)
            rows.append({"condition": cond, **asdict(st)})
    return rows
