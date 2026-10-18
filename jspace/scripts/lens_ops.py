"""Lens directions and residual-stream interventions.

The library ships no steering helper, so these implement the constructions the
paper's data/experiments/README.md describes.

The Stage 3 intervention is the *lens-coordinate patch*, not a naive addition:

    h' = h - (h.u)u + a*u        u = unit lens direction, a = alpha * scale

which sets the coordinate along ``u`` to exactly ``a`` and leaves the
orthogonal remainder of the residual untouched. Note that alpha=0 is therefore
still an intervention (it clamps the concept coordinate to zero); the true
no-intervention run is recorded separately as the ``baseline`` condition.
"""
from __future__ import annotations

import contextlib

import torch


def layer_mean_residual_norm(model, prompts, layer: int) -> float:
    """Mean L2 norm of the residual stream at ``layer`` over ``prompts``.
    This is the unit ``alpha`` is expressed in."""
    from jlens.hooks import ActivationRecorder

    norms = []
    with torch.no_grad():
        for p in prompts:
            ids = model.encode(p, max_length=512)
            with ActivationRecorder(model.layers, at=[layer]) as rec:
                model.forward(ids)
            h = rec.activations[layer].float()
            norms.append(h.norm(dim=-1).mean().item())
    return float(sum(norms) / len(norms))


@contextlib.contextmanager
def intervene(model, layer: int, fn):
    """Apply ``fn(hidden) -> hidden`` to the output of block ``layer``."""

    def hook(module, inputs, output):
        is_tuple = not torch.is_tensor(output)
        tensor = output[0] if is_tuple else output
        new = fn(tensor)
        return (new,) + tuple(output[1:]) if is_tuple else new

    handle = model.layers[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


class CoordinatePatch:
    """Clamp the ``direction`` coordinate of the residual to ``amount``.

    Records the L2 norm of the applied delta so the random / orthogonal
    controls can be checked for norm matching after the fact rather than
    assumed.
    """

    def __init__(self, direction: torch.Tensor, amount: float, positions=None):
        self.u = direction / direction.norm()
        self.amount = float(amount)
        self.positions = positions
        self.delta_norm = None        # mean over positions
        self.delta_norms_pos = None   # [P] -- the reference for matched controls

    def __call__(self, h):
        u = self.u.to(h.device, h.dtype)
        out = h.clone()
        sl = slice(None) if self.positions is None else self.positions
        seg = out[:, sl, :]
        coeff = (seg.float() @ u.float()).unsqueeze(-1)      # [B, P, 1]
        delta = (self.amount - coeff) * u.float()            # [B, P, d]
        n = delta.norm(dim=-1)                               # [B, P]
        self.delta_norms_pos = n[0].detach().clone()
        self.delta_norm = float(n.mean())
        out[:, sl, :] = (seg.float() + delta).to(h.dtype)
        return out


class MatchedDeltaPatch:
    """Additive perturbation along ``direction`` whose per-position delta norm
    is matched exactly to a reference trial (the concept condition at the same
    prompt and alpha).

    The clamp form cannot match norms across directions, because the amount it
    has to move depends on how much of that direction the residual already
    carries -- and the orthogonal complement carries far more than any single
    tracked direction does. This control removes that confound: identical
    perturbation magnitude at every position, different direction.
    """

    def __init__(self, direction, ref_norms_pos, sign=1.0, positions=None):
        self.u = direction / direction.norm()
        self.ref = ref_norms_pos
        self.sign = 1.0 if sign >= 0 else -1.0
        self.positions = positions
        self.delta_norm = None
        self.delta_norms_pos = None

    def __call__(self, h):
        u = self.u.to(h.device, h.dtype)
        out = h.clone()
        sl = slice(None) if self.positions is None else self.positions
        seg = out[:, sl, :]
        ref = self.ref.to(seg.device).float()[: seg.shape[1]]
        delta = self.sign * ref[None, :, None] * u.float()
        self.delta_norms_pos = delta.norm(dim=-1)[0].detach().clone()
        self.delta_norm = float(delta.norm(dim=-1).mean())
        out[:, sl, :] = (seg.float() + delta).to(h.dtype)
        return out


def coordinate_rms(model, prompts, layer, direction, max_len=64) -> float:
    """RMS of the residual's coordinate along ``direction`` at ``layer``,
    over all non-BOS positions of ``prompts``.

    This is the unit alpha is expressed in. Using the layer's *total* residual
    norm instead makes alpha=1 mean "put the entire residual magnitude on one
    direction", which is ~33x the amount that direction normally carries and
    destroys the model in every condition, controls included -- so the sweep
    would measure only how the model breaks.
    """
    from jlens.hooks import ActivationRecorder

    u = (direction / direction.norm())
    sq, n = 0.0, 0
    with torch.no_grad():
        for p in prompts:
            ids = model.encode(p, max_length=max_len)
            with ActivationRecorder(model.layers, at=[layer]) as rec:
                model.forward(ids)
            c = (rec.activations[layer][0].float() @ u.to(
                rec.activations[layer].device).float())[1:]
            sq += float((c ** 2).sum()); n += c.numel()
    return (sq / max(n, 1)) ** 0.5


def make_swapper(d_from: torch.Tensor, d_to: torch.Tensor, positions=None):
    """Replace the component along ``d_from`` with the same coefficient along
    ``d_to`` -- the paper's "clamping a lens coordinate" swap. Used by Stage 2."""
    def fn(h):
        u = d_from.to(h.device, h.dtype)
        v = d_to.to(h.device, h.dtype)
        out = h.clone()
        sl = slice(None) if positions is None else positions
        seg = out[:, sl, :]
        coeff = (seg @ u).unsqueeze(-1)
        out[:, sl, :] = seg - coeff * u + coeff * v
        return out
    return fn


def random_direction(d_model: int, seed: int, device="cpu") -> torch.Tensor:
    """Fixed pseudo-random unit direction. Control 1: if a step appears here
    too, it is an artefact of injection magnitude, not ignition."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    v = torch.randn(d_model, generator=g)
    return (v / v.norm()).to(device)


def orthogonal_direction(lens, layer: int, seed: int, device="cpu"):
    """Unit direction orthogonal to the span of every tracked readout
    direction at ``layer``. Control 2: the readout should barely move.

    Note on what "orthogonal to J-space" can mean here: ``J_l`` is a square
    ``d_model x d_model`` matrix and generically full rank, so *nothing* is
    orthogonal to its row space -- that reading is vacuous. The meaningful
    complement is the one taken here: orthogonal to the subspace this
    experiment's measurement can actually see.

    Returns ``(direction, max_abs_cosine_against_tracked)``.
    """
    M = lens.matrix(layer).float()                  # [n_seed, d_model]
    Q, _ = torch.linalg.qr(M.T)                     # [d_model, n_seed] basis
    g = torch.Generator(device="cpu").manual_seed(seed)
    v = torch.randn(M.shape[1], generator=g)
    v = v - Q @ (Q.T @ v)
    v = v / v.norm()
    leak = (M @ v).abs().max().item()
    return v.to(device), leak
