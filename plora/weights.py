from __future__ import annotations

"""Lightweight weight-space statistics for adapters.

For sim/dry-runs without loading a backbone, we approximate norms using the
artefact file size and a policy-provided reference distribution.
"""

from pathlib import Path
from typing import Tuple

from .manifest import Manifest


def weight_norm_proxy(adapter_dir: Path, manifest: Manifest) -> float:
    """Return a simple proxy for adapter "magnitude".

    If adapter file exists, use its on-disk size in bytes as the proxy.
    """
    artefact = adapter_dir / manifest.artifacts.filename
    try:
        return float(artefact.stat().st_size)
    except Exception:
        return float(manifest.artifacts.size_bytes)


def weight_norm_zscore(adapter_dir: Path, manifest: Manifest, ref_mean: float, ref_std: float) -> float:
    """Compute a z-score of the proxy norm against a reference distribution.

    ref_std must be > 0; caller is responsible for sensible values.
    """
    x = weight_norm_proxy(adapter_dir, manifest)
    if ref_std <= 0:
        return 0.0
    return (x - ref_mean) / ref_std


def weight_norms_from_safetensors(adapter_dir: Path) -> Tuple[float, float]:
    """Return (frobenius_norm_sum, max_tensor_norm) over LoRA A/B tensors.

    If safetensors is unavailable or file missing, returns (0.0, 0.0).
    """
    try:
        from safetensors.torch import load_file  # type: ignore
        import torch
    except Exception:
        return 0.0, 0.0

    # Attempt to load adapter weights
    ckpt = adapter_dir / "adapter_model.safetensors"
    if not ckpt.exists():
        return 0.0, 0.0
    try:
        tensors = load_file(str(ckpt))
    except Exception:
        return 0.0, 0.0

    fro_sum = 0.0
    max_norm = 0.0
    for name, t in tensors.items():
        if "lora_A" in name or "lora_B" in name:
            n = float(t.norm().item())
            fro_sum += n
            if n > max_norm:
                max_norm = n
    return fro_sum, max_norm


