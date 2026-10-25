"""Weight access per architecture, for the induction tools (24 Oct, the Qwen2.5 ladder).

The induction tools were written for GPT-NeoX (Pythia), which keeps q, k and v in one fused matrix laid
out per head as [q | k | v]. Llama-style models (Qwen2, Llama, Mistral) keep separate q_proj / k_proj /
v_proj, and with grouped-query attention several query heads read with one shared key and value head:
query head h uses kv head h // (H / n_kv). Everything below returns the same objects for both, so a
Pythia run measures exactly what it measured before (same matrices, same SVDs, same null).

  head_qkv(sd, l, h)   the (hd, d) query, key and value rows query head h reads the stream with
  head_out(sd, l, h)   its (d, hd) columns of the output projection
  ln_attn(sd, l)       the gain of the norm in front of layer l's attention (folded into readers; ones in OLMo-1)
  embed(sd), unembed(sd)   the unembedding with the final norm's gain folded in
"""
from __future__ import annotations
import torch

from tools.pythia_joints import thin, parts_k, _dt


class Arch:
    def __init__(self, model):
        c = model.config
        self.L, self.H, self.d = c.num_hidden_layers, c.num_attention_heads, c.hidden_size
        self.hd = getattr(c, "head_dim", None) or self.d // self.H
        self.n_kv = getattr(c, "num_key_value_heads", None) or self.H
        mt = c.model_type
        if mt == "gpt_neox":
            self.family = "neox"
        elif mt in ("qwen2", "llama", "mistral"):
            self.family = "llama"          # RMSNorm gain is `weight` itself (not 1 + weight as in Gemma)
        elif mt == "olmo":
            self.family = "olmo"           # OLMo-1 (25 Oct): llama names, LayerNorm with no gain at all
        else:
            raise ValueError(f"no weight layout for model_type {mt!r}")

    def head_qkv(self, sd, l, h):
        hd = self.hd
        if self.family == "neox":
            qkv = sd[f"gpt_neox.layers.{l}.attention.query_key_value.weight"].reshape(self.H, 3, hd, self.d)[h]
            return qkv[0], qkv[1], qkv[2]
        pre, g = f"model.layers.{l}.self_attn.", h // (self.H // self.n_kv)
        return (sd[pre + "q_proj.weight"][h * hd:(h + 1) * hd],
                sd[pre + "k_proj.weight"][g * hd:(g + 1) * hd],
                sd[pre + "v_proj.weight"][g * hd:(g + 1) * hd])

    def head_out(self, sd, l, h):
        key = (f"gpt_neox.layers.{l}.attention.dense.weight" if self.family == "neox"
               else f"model.layers.{l}.self_attn.o_proj.weight")
        return sd[key][:, h * self.hd:(h + 1) * self.hd]

    def ln_attn(self, sd, l):
        if self.family == "olmo":
            return torch.ones(self.d, dtype=sd["model.embed_tokens.weight"].dtype)
        return sd[f"gpt_neox.layers.{l}.input_layernorm.weight" if self.family == "neox"
                  else f"model.layers.{l}.input_layernorm.weight"]

    def embed(self, sd):
        return sd["gpt_neox.embed_in.weight" if self.family == "neox" else "model.embed_tokens.weight"]

    def unembed(self, sd):
        if self.family == "neox":
            W = sd["lm_head.weight"] if "lm_head.weight" in sd else sd["embed_out.weight"]
            return W * sd["gpt_neox.final_layer_norm.weight"][None, :]
        W = sd["lm_head.weight"] if "lm_head.weight" in sd else sd["model.embed_tokens.weight"]   # tied in small Qwen
        return W if self.family == "olmo" else W * sd["model.norm.weight"][None, :]


def induction_parts(sd, A, heads, dev="cpu"):
    """The writers and readers the induction tool scores, and nothing else: embedding (E), unembedding
    (U), and for each (layer, head) its key read K{l}h{h} (norm gain folded in) and output write
    O{l}h{h}, with each head's own shuffle-null k. Same definitions as pythia_joints.stream_parts
    (per_head), without the all-layer parts that tool computes and this one never used."""
    g = lambda t: t.detach().to(dev, _dt(dev))
    out = {}
    U, s, Vh = thin(g(A.embed(sd))); out["E"] = ("w", Vh.T, s)
    U, s, Vh = thin(g(A.unembed(sd))); out["U"] = ("r", Vh.T, s)
    for l, h in dict.fromkeys(heads):
        Kh = g(A.head_qkv(sd, l, h)[1]) * g(A.ln_attn(sd, l))[None, :]
        Oh = g(A.head_out(sd, l, h))
        U, s, Vh = thin(Kh); out[f"K{l}h{h}"] = ("r", Vh.T, s); out[f"k_K{l}h{h}"] = parts_k(Kh, s)
        U, s, Vh = thin(Oh); out[f"O{l}h{h}"] = ("w", U, s); out[f"k_O{l}h{h}"] = parts_k(Oh, s)
    return out
