from __future__ import annotations

"""Theoretical predictors for diffusion on graphs.

We include:
- spectral-gap based heuristic for rounds to diffuse under push–pull gossip:
  t ≈ C log(n) / λ2
- Cheeger bounds via conductance Φ: 1/(2Φ) ≤ mixing_time ≤ 1/Φ^2 up to logs
- Epidemic threshold proxy for SIS-like spreading: β_c ≈ 1/λ1(A)
"""

import math
from typing import Sequence
import torch


def predicted_rounds_spectral(n: int, lambda2: float, *, C: float = 2.0) -> int:
    """Return predicted rounds to diffuse: ceil(C * log(n) / max(lambda2, eps)).

    This is a heuristic; for disconnected graphs (lambda2≈0) we return a large
    sentinel value proportional to n.
    """
    if n <= 1:
        return 0
    eps = 1e-9
    # Use an effective gap bounded by 1.0 to approximate normalized scaling
    # and avoid decreasing predictions with growing n on dense graphs.
    lam = max(min(lambda2, 1.0), eps)
    if lambda2 <= eps:
        return int(max(1, math.ceil(C * math.log(n) * n)))
    return int(max(1, math.ceil(C * math.log(n) / lam)))


def _laplacian(neighbours: Sequence[Sequence[int]]) -> torch.Tensor:
    n = len(neighbours)
    A = torch.zeros((n, n), dtype=torch.float64)
    for i, nbrs in enumerate(neighbours):
        for j in nbrs:
            if 0 <= j < n and j != i:
                A[i, j] = 1.0
                A[j, i] = 1.0
    D = torch.diag(A.sum(dim=1))
    return D - A


def cheeger_bounds(neighbours: Sequence[Sequence[int]]) -> tuple[float, float]:
    """Return (lower, upper) bounds on mixing time via conductance proxy.

    We approximate conductance Φ using the Fiedler vector induced cut.
    Lower bound ~ 1/(2Φ), upper bound ~ 1/Φ^2 (ignoring log-factors).
    """
    n = len(neighbours)
    if n == 0:
        return (0.0, 0.0)
    L = _laplacian(neighbours)
    try:
        evals, evecs = torch.linalg.eigh(L)
    except RuntimeError:
        L = L.to(torch.float32)
        evals, evecs = torch.linalg.eigh(L)
    idx = torch.argsort(evals)
    if evals.numel() < 2:
        return (0.0, 0.0)
    fiedler = evecs[:, idx[1]]
    # Partition by sign of Fiedler vector
    S = set(torch.nonzero(fiedler >= 0, as_tuple=False).view(-1).tolist())
    Sc = set(range(n)) - S
    # Conductance estimate
    deg = [len(neighbours[i]) for i in range(n)]
    volS = sum(deg[i] for i in S)
    volSc = sum(deg[i] for i in Sc)
    cut = 0
    for u in S:
        for v in neighbours[u]:
            if v in Sc:
                cut += 1
    denom = max(1, min(volS, volSc))
    phi = cut / denom
    if phi <= 0:
        return (0.0, float("inf"))
    return (float(1.0 / (2 * phi)), float(1.0 / (phi * phi)))


def epidemic_threshold(neighbours: Sequence[Sequence[int]]) -> float:
    """Return β_c ≈ 1/λ1(A) where λ1 is spectral radius of adjacency A."""
    n = len(neighbours)
    if n == 0:
        return 0.0
    A = torch.zeros((n, n), dtype=torch.float64)
    for i, nbrs in enumerate(neighbours):
        for j in nbrs:
            if 0 <= j < n and j != i:
                A[i, j] = 1.0
                A[j, i] = 1.0
    try:
        evals = torch.linalg.eigvalsh(A)
    except RuntimeError:
        A = A.to(torch.float32)
        evals = torch.linalg.eigvalsh(A)
    lam1 = float(torch.max(evals).item()) if evals.numel() else 0.0
    if lam1 <= 0:
        return float("inf")
    return float(1.0 / lam1)


# Convenience re-export: conductance helper (kept here for theory cohesion)
def conductance_estimate(
    neighbours: Sequence[Sequence[int]], trials: int = 64, seed: int = 42
) -> float:
    """Approximate conductance Φ(G) via random cuts.

    Delegates to the implementation in swarm.metrics to keep a single source of truth
    while allowing imports from swarm.theory for all diffusion-related predictors.
    """
    from .metrics import conductance_estimate as _conductance_estimate  # lazy import

    return _conductance_estimate(neighbours, trials=trials, seed=seed)
