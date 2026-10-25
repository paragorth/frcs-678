"""Closed-form Haar null for the composition score, after Wang (2026, arXiv 2608.22007).

For a writer W (d x n, into the residual stream) and a reader R (m x d), the squared composition score
is C^2 = ||R W||_F^2 / (||R||_F^2 ||W||_F^2) = tr(G Q^T H Q) / (tr G tr H) with Q = I, where
G = R^T R and H = W W^T. Replace Q by a Haar-random rotation of the stream: each matrix keeps its own
spectrum and only the alignment is destroyed. In eigenbases, T = tr(G Q^T H Q) = sum_ij a_i b_j Q_ij^2
with a = eig(G) = s_R^2 and b = eig(H) = s_W^2, so its moments follow from those of squared Haar
orthogonal entries:
    E[Q_ij^2] = 1/d,  E[Q_ij^4] = 3/(d(d+2)),  E[Q_ij^2 Q_ik^2] = 1/(d(d+2))  (shared row or column),
    E[Q_ij^2 Q_lk^2] = (d+1)/(d(d-1)(d+2))  (i != l, j != k).
Hence E[C^2] = 1/d exactly (Wang, Prop. 5), and with A1 = sum a, A2 = sum a^2 (B likewise)
    E[T^2] = [3 A2 B2 + A2 (B1^2 - B2) + (A1^2 - A2) B2] / (d(d+2))
             + (A1^2 - A2)(B1^2 - B2)(d+1) / (d(d-1)(d+2)).
Checks: G = H = I gives E[T^2] = d^2 (T is constant); rank one gives 3/(d(d+2)) = E[X^2] for
X ~ Beta(1/2, (d-1)/2), Wang's Lemma 6. z = (C^2 - 1/d) / sd; p is the one-sided normal tail
(an approximation for small ranks, where the null is skewed); Benjamini-Hochberg across the joints
measured at one step.
"""
import math
import numpy as np


def haar_null(s_r, s_w, d):
    """Mean and variance of C^2 under a Haar rotation, from the singular values of R and W."""
    a = np.asarray(s_r, np.float64) ** 2; b = np.asarray(s_w, np.float64) ** 2
    A1, A2, B1, B2 = a.sum(), (a * a).sum(), b.sum(), (b * b).sum()
    ET2 = ((3 * A2 * B2 + A2 * (B1 ** 2 - B2) + (A1 ** 2 - A2) * B2) / (d * (d + 2))
           + (A1 ** 2 - A2) * (B1 ** 2 - B2) * (d + 1) / (d * (d - 1) * (d + 2)))
    mean = 1.0 / d
    var = ET2 / (A1 * B1) ** 2 - mean ** 2
    return mean, max(var, 0.0)


def z_and_p(c2, s_r, s_w, d):
    mean, var = haar_null(s_r, s_w, d)
    z = (c2 - mean) / math.sqrt(var) if var > 0 else float("nan")
    p = 0.5 * math.erfc(z / math.sqrt(2)) if z == z else float("nan")
    return z, p


def bh(pvals, q=0.05):
    """Benjamini-Hochberg: which of pvals are significant at false-discovery rate q."""
    p = np.asarray(pvals, np.float64); n = len(p)
    order = np.argsort(p); ranked = p[order]
    below = ranked <= q * (np.arange(1, n + 1) / n)
    keep = np.zeros(n, bool)
    if below.any():
        keep[order[:np.nonzero(below)[0].max() + 1]] = True
    return keep
