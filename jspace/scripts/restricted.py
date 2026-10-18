"""Restricted cotangent seeding: the lens rows for a chosen token set.

The library's fit computes the full ``J_l`` (``d_model`` backward passes per
prompt: one per output dimension). Everything downstream here only ever uses
the per-token readout direction

    r_{l,t} = J_l^T (W_U[t] * g)        g = final-norm gain (RMSNorm treated
                                        as a fixed diagonal, the same
                                        linearisation the paper's
                                        transpose-row definition assumes)

so we seed the cotangent with ``W_U[t] * g`` directly instead of with
one-hots. This is not an approximation: seeding cotangent ``c`` at every valid
target position and averaging the gradient over source positions returns
exactly ``J_l^T c`` for the library's own estimator. It costs ``n_seed``
backward passes per prompt instead of ``d_model`` (~160 vs 1536, a ~10x
saving), and the accumulator is ``[n_seed, d_model]`` per layer.

Cotangents are unit-normalised and scaled by COTANGENT_SCALE. Both are
irrelevant to the result -- only the direction of each row is ever used -- but
the scale buys headroom against fp16 underflow in the backward pass.
"""
from __future__ import annotations

import math
import os
import time

import torch

from jlens.fitting import _atomic_save, valid_position_mask
from jlens.hooks import ActivationRecorder

COTANGENT_SCALE = 256.0


def cotangent_rows(model, token_ids) -> torch.Tensor:
    """``[n_seed, d_model]`` unit cotangents ``normalize(W_U[t] * g)``."""
    W_U = model._lm_head.weight
    g = getattr(model._final_norm, "weight", None)
    rows = W_U[torch.as_tensor(token_ids, device=W_U.device)].float()
    if g is not None:
        rows = rows * g.float()
    return rows / rows.norm(dim=-1, keepdim=True)


def rows_for_prompt(model, prompt, source_layers, cot, *, target_layer=None,
                    seed_batch=32, max_seq_len=256, skip_first=16):
    """``{layer: [n_seed, d_model]}`` for one prompt. Mirrors
    :func:`jlens.fitting.jacobian_for_prompt` with cotangents from ``cot``
    ([n_seed, d_model]) in place of one-hot output dimensions."""
    n_layers, d_model = model.n_layers, model.d_model
    target_layer = n_layers - 1 if target_layer is None else target_layer
    n_seed = cot.shape[0]

    input_ids = model.encode(prompt, max_length=max_seq_len)
    seq_len = input_ids.shape[1]
    position_mask = valid_position_mask(seq_len, skip_first=skip_first)
    n_valid = int(position_mask.sum())

    out = {l: torch.zeros(n_seed, d_model, dtype=torch.float32)
           for l in source_layers}
    n_passes = math.ceil(n_seed / seed_batch)

    with (
        ActivationRecorder(model.layers, at=[*source_layers, target_layer],
                           start_graph_at=min(source_layers)) as rec,
        torch.enable_grad(),
    ):
        replicated = input_ids.expand(seed_batch, -1)
        model.forward(replicated)
        target_act = rec.activations[target_layer]     # [seed_batch, S, d]
        source_acts = [rec.activations[l] for l in source_layers]

        valid = position_mask.nonzero(as_tuple=True)[0].to(target_act.device)
        cotangent = torch.zeros_like(target_act)
        cot_dev = (cot * COTANGENT_SCALE).to(target_act.device, target_act.dtype)

        for pass_idx, start in enumerate(range(0, n_seed, seed_batch)):
            n_this = min(seed_batch, n_seed - start)
            # Batch element b carries token (start+b)'s cotangent at every
            # valid target position; the rest of the batch is zeroed so its
            # gradient contribution vanishes.
            cotangent.zero_()
            cotangent[:n_this, valid, :] = cot_dev[start:start + n_this, None, :]
            grads = torch.autograd.grad(
                outputs=target_act, inputs=source_acts,
                grad_outputs=cotangent, retain_graph=(pass_idx < n_passes - 1))
            for layer, grad in zip(source_layers, grads, strict=True):
                pos = valid.to(grad.device, non_blocking=True)
                out[layer][start:start + n_this] = (
                    grad[:n_this, pos, :].float().mean(dim=1).cpu())
            del grads

    return out, seq_len, n_valid


class RestrictedLens:
    """Per-layer readout rows for a fixed token set.

    ``rows[layer][j]`` is ``J_l^T (W_U[t_j] * g)`` up to a positive scale.
    ``direction`` returns the unit vector, which is all downstream code uses.
    """

    def __init__(self, rows, seeds, *, n_prompts, d_model, meta=None):
        self.rows = {int(l): r.float() for l, r in rows.items()}
        self.seeds = seeds                      # [{word,id,group}, ...]
        self.n_prompts = n_prompts
        self.d_model = d_model
        self.meta = meta or {}
        self.source_layers = sorted(self.rows)
        self.index = {s["id"]: j for j, s in enumerate(seeds)}
        self.by_word = {s["word"]: j for j, s in enumerate(seeds)}

    def __repr__(self):
        return (f"RestrictedLens(d_model={self.d_model}, n_seed={len(self.seeds)}, "
                f"n_prompts={self.n_prompts}, layers="
                f"[{self.source_layers[0]}..{self.source_layers[-1]}])")

    def direction(self, layer: int, token_id: int) -> torch.Tensor:
        r = self.rows[layer][self.index[token_id]]
        return r / r.norm()

    def matrix(self, layer: int) -> torch.Tensor:
        """``[n_seed, d_model]`` of unit readout directions at ``layer``."""
        R = self.rows[layer]
        return R / R.norm(dim=-1, keepdim=True)

    def save(self, path):
        torch.save({"rows": {l: r.to(torch.float32) for l, r in self.rows.items()},
                    "seeds": self.seeds, "n_prompts": self.n_prompts,
                    "d_model": self.d_model, "meta": self.meta}, path)

    @classmethod
    def load(cls, path):
        c = torch.load(path, map_location="cpu", weights_only=False)
        return cls(c["rows"], c["seeds"], n_prompts=c["n_prompts"],
                   d_model=c["d_model"], meta=c.get("meta"))


def fit_restricted(model, prompts, seeds, *, source_layers, target_layer=None,
                   seed_batch=32, max_seq_len=256, skip_first=16,
                   checkpoint_path=None, checkpoint_every=1, resume=True,
                   log=print):
    """Accumulate the restricted rows as a running mean over ``prompts``.
    Checkpoints atomically every ``checkpoint_every`` prompts and resumes."""
    d_model = model.d_model
    token_ids = [s["id"] for s in seeds]
    cot = cotangent_rows(model, token_ids)
    n_seed = len(seeds)

    row_sum, n_done, next_idx, n_bad = None, 0, 0, 0
    if resume and checkpoint_path and os.path.exists(checkpoint_path):
        st = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if st["source_layers"] != list(source_layers) or st["token_ids"] != token_ids:
            raise ValueError(
                f"checkpoint {checkpoint_path} was fitted with a different layer "
                f"or token set; move it aside to refit")
        row_sum, n_done, next_idx = st["row_sum"], st["n_done"], st["next_idx"]
        log(f"  resuming from checkpoint: {next_idx}/{len(prompts)} prompts")
    if row_sum is None:
        row_sum = {l: torch.zeros(n_seed, d_model, dtype=torch.float32)
                   for l in source_layers}

    def write_ckpt():
        if checkpoint_path:
            _atomic_save({"row_sum": row_sum, "n_done": n_done,
                          "next_idx": next_idx, "source_layers": list(source_layers),
                          "token_ids": token_ids}, checkpoint_path)

    for i, prompt in enumerate(prompts):
        if i < next_idx:
            continue
        t0 = time.perf_counter()
        try:
            per, seq_len, n_valid = rows_for_prompt(
                model, prompt, source_layers, cot, target_layer=target_layer,
                seed_batch=seed_batch, max_seq_len=max_seq_len,
                skip_first=skip_first)
        except ValueError as exc:      # prompt too short
            log(f"  skipping prompt {i}: {exc}")
            next_idx = i + 1
            continue
        # Finiteness guard. A single prompt that produces a non-finite row
        # poisons the running mean for every prompt after it, and the result
        # is a lens of pure NaN that still saves and loads without complaint.
        # Skip the prompt, count it, and abort if it is not a rare accident.
        bad = {l: int((~torch.isfinite(per[l])).sum()) for l in source_layers}
        if any(bad.values()):
            n_bad += 1
            log(f"  !! prompt {i} produced non-finite rows {bad} -- skipped "
                f"({n_bad} bad so far)")
            next_idx = i + 1
            if n_bad >= 10 and n_bad > 0.25 * (n_done + n_bad):
                raise RuntimeError(
                    f"{n_bad} of {n_done + n_bad} prompts produced non-finite "
                    f"rows; this is systematic, not incidental -- stopping "
                    f"rather than fitting a lens on the remainder")
            continue
        # Convergence diagnostic: how much this prompt moves the running mean.
        if n_done > 0:
            rel = max(((per[l] - row_sum[l] / n_done).norm()
                       / ((n_done + 1) * (row_sum[l] / n_done).norm())).item()
                      for l in source_layers)
        else:
            rel = float("nan")
        for l in source_layers:
            row_sum[l] += per[l]
        n_done += 1
        next_idx = i + 1
        if i % 10 == 0 or i == len(prompts) - 1:
            log(f"  prompt {i+1}/{len(prompts)} seq={seq_len} valid={n_valid} "
                f"{time.perf_counter()-t0:.2f}s d_mean={rel:.2e}")
        if checkpoint_every and next_idx % checkpoint_every == 0:
            write_ckpt()

    write_ckpt()
    if n_done == 0:
        raise ValueError("no prompts were long enough to fit on")
    rows = {l: row_sum[l] / n_done for l in source_layers}
    still_bad = {l: int((~torch.isfinite(rows[l])).sum()) for l in source_layers}
    if any(still_bad.values()):
        raise RuntimeError(f"fitted lens is non-finite after guarding: {still_bad}")
    log(f"fit: {n_done} prompts used, {n_bad} skipped as non-finite")
    lens = RestrictedLens(rows, seeds, n_prompts=n_done, d_model=d_model)
    lens.meta = {"n_bad_prompts": n_bad}
    return lens
