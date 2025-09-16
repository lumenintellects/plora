from __future__ import annotations

"""Lightweight weight-space statistics for adapters.

For sim/dry-runs without loading a backbone, we approximate norms using the
artefact file size and a policy-provided reference distribution.
"""

from pathlib import Path
from typing import Tuple, List

from .manifest import Manifest
import json
import math
import numpy as np


def weight_norm_proxy(adapter_dir: Path, manifest: Manifest) -> float:
    """Return a simple proxy for adapter "magnitude".

    If adapter file exists, use its on-disk size in bytes as the proxy.
    """
    artefact = adapter_dir / manifest.artifacts.filename
    try:
        return float(artefact.stat().st_size)
    except Exception:
        return float(manifest.artifacts.size_bytes)


def weight_norm_zscore(
    adapter_dir: Path, manifest: Manifest, ref_mean: float, ref_std: float
) -> float:
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


def tensor_norm_anomaly_z(adapter_dir: Path) -> Tuple[float, int]:
    """Return (max_z, count) where z is robust z-score of per-tensor norms.

    Uses median and MAD for robustness. Returns (0.0, 0) if unavailable.
    """
    try:
        from safetensors.torch import load_file
        import torch
    except Exception:
        return 0.0, 0

    ckpt = adapter_dir / "adapter_model.safetensors"
    if not ckpt.exists():
        return 0.0, 0
    try:
        tensors = load_file(str(ckpt))
    except Exception:
        return 0.0, 0

    norms: List[float] = []
    for name, t in tensors.items():
        if "lora_A" in name or "lora_B" in name:
            # Use Frobenius norm of 2D tensor; ensure float conversion
            try:
                norms.append(float(t.to(dtype=t.dtype).norm().item()))
            except Exception:
                norms.append(float(t.float().norm().item()))
    if len(norms) < 2:
        return 0.0, len(norms)
    import numpy as np

    arr = np.array(norms, dtype=float)
    med = float(np.median(arr))
    dev = np.abs(arr - med)
    mad = float(np.median(dev))
    if mad > 0:
        z = dev / (1.4826 * mad)
    else:
        # If MAD==0 but there are deviations (ties except one outlier), mark deviations as strong outliers
        z = np.where(dev > 0, 10.0, 0.0)
    return float(np.abs(z).max()), len(norms)


# ---------------------------------------------------------------------------
# Optional lightweight detectors using on-disk artifacts written by probes
# ---------------------------------------------------------------------------


def activation_mahalanobis(adapter_dir: Path) -> float:
    """Return Mahalanobis distance of observed activation vs reference.

    Expects optional files:
      - activation_ref.json: {"mean": [..], "cov": [[..]]}
      - activation_obs.json: {"vec": [..]}
    Returns 0.0 if files missing or invalid.
    """
    try:
        ref_p = adapter_dir / "activation_ref.json"
        obs_p = adapter_dir / "activation_obs.json"
        if not (ref_p.exists() and obs_p.exists()):
            return 0.0
        ref = json.loads(ref_p.read_text())
        obs = json.loads(obs_p.read_text())
        mu = np.array(ref.get("mean", []), dtype=float)
        cov = np.array(ref.get("cov", []), dtype=float)
        x = np.array(obs.get("vec", []), dtype=float)
        if mu.size == 0 or cov.size == 0 or x.size == 0:
            return 0.0
        if (
            x.shape[0] != mu.shape[0]
            or cov.shape[0] != cov.shape[1]
            or cov.shape[0] != mu.shape[0]
        ):
            return 0.0
        # regularize covariance for stability
        cov_reg = cov + 1e-6 * np.eye(cov.shape[0])
        inv = np.linalg.inv(cov_reg)
        d = x - mu
        m2 = float(d.T @ inv @ d)
        return float(math.sqrt(max(0.0, m2)))
    except Exception:
        return 0.0


def gradient_spike_z(adapter_dir: Path) -> float:
    """Return robust z-score for a single observed grad norm vs reference.

    Expects optional files:
      - grad_ref.json: {"median": m, "mad": s}
      - grad_obs.json: {"norm": v}
    Returns 0.0 if files missing or invalid.
    """
    try:
        ref_p = adapter_dir / "grad_ref.json"
        obs_p = adapter_dir / "grad_obs.json"
        if not (ref_p.exists() and obs_p.exists()):
            return 0.0
        ref = json.loads(ref_p.read_text())
        obs = json.loads(obs_p.read_text())
        med = float(ref.get("median", 0.0))
        mad = float(ref.get("mad", 0.0))
        val = float(obs.get("norm", 0.0))
        if mad <= 0:
            return 0.0
        z = abs((val - med) / (1.4826 * mad))
        return float(z)
    except Exception:
        return 0.0
