from __future__ import annotations

"""Linear algebra helpers for LoRA A/B analysis.

Utilities include:
- Loading LoRA A/B tensors from an adapter directory
- Subspace principal angles between two matrices
- Overlap metrics and effective rank estimators
"""

from pathlib import Path
from typing import Dict, Tuple

import torch


def load_lora_AB_from_safetensors(
    adapter_dir: Path,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """Return mapping of module key -> (A, B) LoRA tensors from safetensors.

    If file is missing or unreadable, returns an empty dict.
    """
    try:
        from safetensors.torch import load_file
    except Exception:
        return {}

    ckpt = adapter_dir / "adapter_model.safetensors"
    if not ckpt.exists():
        return {}
    try:
        tensors = load_file(str(ckpt))
    except Exception:
        return {}

    grouped: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    A_map: Dict[str, torch.Tensor] = {}
    B_map: Dict[str, torch.Tensor] = {}
    for name, t in tensors.items():
        if not isinstance(t, torch.Tensor):
            continue
        if "lora_A" in name:
            key = name.split(".lora_A")[0]
            A_map[key] = t.float()
        elif "lora_B" in name:
            key = name.split(".lora_B")[0]
            B_map[key] = t.float()

    for key, A in A_map.items():
        B = B_map.get(key)
        if B is not None:
            grouped[key] = (A, B)
    return grouped


def orthonormal_basis(X: torch.Tensor) -> torch.Tensor:
    """Return an orthonormal basis for the column space of X via QR.

    Handles empty or rank-0 matrices by returning an empty basis with correct shape.
    """
    if X.numel() == 0:
        return X
    # Move to CPU for numerical stability in QR if needed
    Xc = X.detach().to(torch.float32)
    try:
        Q, _ = torch.linalg.qr(Xc, mode="reduced")
    except RuntimeError:
        # Fallback: SVD-based basis
        U, S, _ = torch.linalg.svd(Xc, full_matrices=False)
        tol = (
            max(Xc.shape)
            * torch.finfo(S.dtype).eps
            * (S.max() if S.numel() else torch.tensor(0.0))
        )
        r = int((S > tol).sum().item()) if S.numel() > 0 else 0
        Q = U[:, :r]
    return Q


def principal_angles(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Compute principal angles (radians) between column spaces of A and B.

    Returns angles sorted descending in [0, pi/2]. If one space is rank-0,
    returns a zero-length tensor.
    """
    QA = orthonormal_basis(A)
    QB = orthonormal_basis(B)
    if QA.numel() == 0 or QB.numel() == 0:
        return torch.empty(0)
    # Compute singular values of cross-Gram matrix
    M = QA.T @ QB
    s = torch.linalg.svdvals(M)
    s = torch.clamp(s, 0.0, 1.0)
    # Convert to angles; sort descending
    ang = torch.acos(s)
    ang, _ = torch.sort(ang, descending=True)
    return ang


def subspace_overlap_cos2(A: torch.Tensor, B: torch.Tensor) -> float:
    """Return mean cos^2 of principal angles between subspaces of A and B.

    1.0 means identical subspaces (up to rotation), 0.0 means orthogonal.
    """
    ang = principal_angles(A, B)
    if ang.numel() == 0:
        return 0.0
    c2 = torch.cos(ang) ** 2
    return float(c2.mean().item())


def effective_rank(X: torch.Tensor, *, tol: float | None = None) -> int:
    """Estimate numerical rank via SVD thresholding.

    If tol is None, use heuristic tol = max(m, n) * eps * max(S).
    """
    if X.numel() == 0:
        return 0
    U, S, Vh = torch.linalg.svd(X.detach().to(torch.float32), full_matrices=False)
    if S.numel() == 0:
        return 0
    if tol is None:
        tol = max(X.shape) * torch.finfo(S.dtype).eps * float(S.max().item())
    return int((S > tol).sum().item())
