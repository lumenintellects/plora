from __future__ import annotations

"""plora.compat - small helpers to smooth over device & dtype decisions.

The prototype is CPU-first but we retain optional support for Apple-Silicon
(MPS) and CUDA; detection is automatic. The helpers keep **all** downstream
modules free from direct `torch.cuda` checks.
"""

import os
from functools import lru_cache
from typing import Tuple

import torch


@lru_cache(maxsize=1)
def get_device() -> torch.device:
    """Pick the best available device, defaulting to CPU."""
    if os.getenv("PLORA_FORCE_CPU", "0") == "1":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@lru_cache(maxsize=1)
def get_dtype() -> torch.dtype:
    """Heuristic dtype based on device capabilities."""
    dev = get_device().type
    if dev == "cuda":
        return torch.float16
    if dev == "mps":
        return torch.bfloat16
    return torch.float32


def device_dtype() -> Tuple[torch.device, torch.dtype]:
    """Return *(device, dtype)* pair."""
    return get_device(), get_dtype()
