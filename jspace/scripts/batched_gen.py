"""Batched greedy generation.

Every generation-based measure in this repo ran one prompt at a time. Job 2 on
the 14B was ~6700 sequential generations (39 min); the ramp test was 1200 (20
min). Batching is the single biggest wall-clock win available and costs nothing
in fidelity: greedy decoding is deterministic, so a batched result is identical
to the sequential one, given correct left-padding and attention masking.

Left-padding matters: with right-padding the pad tokens sit between the prompt
and the continuation and the model attends to them. Left-padding keeps every
sequence's last real token at position -1, which is also where the injection
hooks and the margin readout expect it.
"""
from __future__ import annotations
import torch


@torch.no_grad()
def generate_batch(model, hf, tok, prompts, max_new=48, max_len=64,
                   patch_factory=None, inject_layer=None, batch_size=16):
    """Greedy-generate continuations for many prompts.

    patch_factory: optional callable returning a fresh patch object per batch
    (for interventions). Applied via `intervene` on inject_layer.
    Returns a list of decoded continuations, aligned with `prompts`.
    """
    from lens_ops import intervene
    old_side = getattr(tok, "padding_side", "right")
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    out_texts = []
    try:
        for i in range(0, len(prompts), batch_size):
            chunk = prompts[i:i + batch_size]
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                      max_length=max_len)
            ids = enc.input_ids.to(model.input_device)
            am = enc.attention_mask.to(model.input_device)
            ctx = (intervene(model, inject_layer, patch_factory())
                   if patch_factory is not None else None)
            if ctx is not None:
                with ctx:
                    gen = hf.generate(ids, attention_mask=am,
                                      max_new_tokens=max_new, do_sample=False,
                                      pad_token_id=tok.pad_token_id)
            else:
                gen = hf.generate(ids, attention_mask=am, max_new_tokens=max_new,
                                  do_sample=False, pad_token_id=tok.pad_token_id)
            for j in range(len(chunk)):
                out_texts.append(tok.decode(gen[j][ids.shape[1]:],
                                            skip_special_tokens=True))
    finally:
        tok.padding_side = old_side
    return out_texts


@torch.no_grad()
def last_logits_batch(model, hf, tok, prompts, max_len=64,
                      patch_factory=None, inject_layer=None, batch_size=16):
    """Final-position logits for many prompts. Same padding contract."""
    from lens_ops import intervene
    old_side = getattr(tok, "padding_side", "right")
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    outs = []
    try:
        for i in range(0, len(prompts), batch_size):
            chunk = prompts[i:i + batch_size]
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                      max_length=max_len)
            ids = enc.input_ids.to(model.input_device)
            am = enc.attention_mask.to(model.input_device)
            ctx = (intervene(model, inject_layer, patch_factory())
                   if patch_factory is not None else None)
            if ctx is not None:
                with ctx:
                    lg = hf(ids, attention_mask=am).logits[:, -1, :].float()
            else:
                lg = hf(ids, attention_mask=am).logits[:, -1, :].float()
            outs.extend(list(lg.cpu()))
    finally:
        tok.padding_side = old_side
    return outs
