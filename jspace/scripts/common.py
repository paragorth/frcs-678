"""Shared model/lens setup for the ignition experiment.

Primary target is Qwen2.5-1.5B-Instruct in fp16 on CUDA. The 0.5B config
stays selectable (--model 0.5b) as a fallback only; it was a constraint of an
8GB laptop, not a choice, and a 24-layer model may not have a clean workspace
band at all.
"""
from __future__ import annotations

import argparse
import os
import resource
import time

import torch
import transformers

import jlens

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.environ.get("JLENS_OUT", os.path.join(ROOT, "out"))

# name, n_layers, injection layer (~25% depth), workspace band (middle third)
CONFIGS = {
    "14b": dict(name="Qwen/Qwen2.5-14B-Instruct", n_layers=48),
    "7b": dict(name="Qwen/Qwen2.5-7B-Instruct", n_layers=28),
    "1.5b": dict(name="Qwen/Qwen2.5-1.5B-Instruct", n_layers=28),
    "0.5b": dict(name="Qwen/Qwen2.5-0.5B-Instruct", n_layers=24),
}
DEFAULT_MODEL = os.environ.get("JLENS_MODEL", "1.5b")


def depth_plan(n_layers: int) -> dict:
    """Injection site at ~25% depth; readout band = the middle third."""
    inject = round(0.25 * n_layers)
    lo, hi = round(n_layers / 3), round(2 * n_layers / 3)
    band = list(range(lo, hi))
    return {"inject_layer": inject, "band": band,
            "source_layers": sorted({inject, *band})}


def depth_plan_at(inject: int, n_layers: int) -> dict:
    """Depth plan for an ARBITRARY injection layer, band defined relative to it.

    band = [inject+2, inject+12) clipped to the stack. Two layers above the
    injection site, up to ten wide. At inject=7 on 28 layers this reproduces
    `depth_plan`'s L9..L18 exactly, so the existing 7B fit is reproduced rather
    than approximated.

    Above inject=15 on a 28-layer model the band truncates against the top of
    the stack (widths 8, 5, 2 at inject 18, 21, 24). That is geometric and
    unavoidable; PREREG-depth-sweep.md excludes those depths from adjudication.
    Defined here once and imported -- never re-derived in a caller.
    """
    # The final layer is the lens's TARGET basis and cannot also be a source
    # (`source_layers must all be < target_layer`). Hence n_layers - 1.
    band = list(range(inject + 2, min(inject + 12, n_layers - 1)))
    return {"inject_layer": inject, "band": band,
            "source_layers": sorted({inject, *band})}


def add_model_arg(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--model", choices=sorted(CONFIGS), default=DEFAULT_MODEL)


def model_slug(key: str) -> str:
    return {"14b": "qwen14b", "7b": "qwen7b",
            "1.5b": "qwen15b", "0.5b": "qwen05b"}[key]


def lens_path(key: str, tag: str = "") -> str:
    return os.path.join(OUT, f"{model_slug(key)}{tag}_jacobian_lens.pt")


def pick_device_dtype():
    if torch.cuda.is_available():
        # bf16 over fp16: same speed on Ada, ~10^38 dynamic range instead of
        # 65504, so a backward pass on a pathological prompt overflows to inf
        # (and then NaN) far less readily. The fit accumulates in fp32 either way.
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    # CPU: fp16 matmul is poorly supported / very slow on ARM; fp32 is the
    # working choice. Reported honestly rather than silently substituted.
    return "cpu", torch.float32


def load_model(key: str = DEFAULT_MODEL, dtype=None, device=None):
    device_, dtype_ = pick_device_dtype()
    device = device or device_
    dtype = dtype or dtype_
    name = CONFIGS[key]["name"]
    hf = transformers.AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
    hf.to(device).eval()
    tok = transformers.AutoTokenizer.from_pretrained(name)
    model = jlens.from_hf(hf, tok)
    assert model.n_layers == CONFIGS[key]["n_layers"], (
        f"{name}: expected {CONFIGS[key]['n_layers']} layers, got {model.n_layers}")
    return model, hf, tok, device, dtype


def peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def peak_vram_gb() -> float:
    return torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0


class Timer:
    def __enter__(self):
        self.t = time.time()
        return self

    def __exit__(self, *exc):
        self.dt = time.time() - self.t
