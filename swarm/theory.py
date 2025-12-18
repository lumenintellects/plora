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


def predicted_rounds_spectral(
    n: int, 
    lambda2: float, 
    *, 
    C: float = 0.7,  # Empirically calibrated: 2.0 * 0.35 (observed efficiency ratio)
    normalized: bool = False,
    initial_informed_fraction: float | None = None,
    safety_margin: float = 0.0
) -> int:
    """Return predicted rounds to diffuse: ceil(C * log((1-p)⁻¹ · n) / max(lambda2, eps)).
    
    For multi-source diffusion, the time scales with the uninformed population remaining.
    If initial_informed_fraction p is provided, uses log((1-p)⁻¹ · n) instead of log(n).
    This accounts for the fact that when information starts from multiple sources,
    diffusion time scales with the remaining uninformed fraction, not total population.
    
    Safety measures:
    - Uses ceil() to round up (conservative)
    - Minimum of 1 round (ensures non-zero)
    - Optional safety_margin multiplier for additional conservatism
    
    Args:
        n: Number of nodes
        lambda2: Spectral gap (λ₂)
        C: Empirical constant (default 2.0)
        normalized: If True, lambda2 is from normalized Laplacian (range [0, 2]).
                    If False, lambda2 is from unnormalized Laplacian.
        initial_informed_fraction: Fraction p of nodes initially informed (default None).
                                   If None, uses standard log(n). If provided (0 < p < 1),
                                   uses log((1-p)⁻¹ · n) = log(n / (1-p)).
                                   For 3-domain setup with p=1/3, this becomes log(3/2 · n).
        safety_margin: Optional multiplier for additional safety (default 0.0 = no extra margin).
                       If > 0, multiplies result by (1 + safety_margin) before rounding.
                       Example: safety_margin=0.1 adds 10% safety buffer.
    
    Returns:
        Predicted number of rounds to achieve full diffusion (always ≥ 1).
        Predictions err on the side of caution (slight over-prediction preferred over under-prediction).
    
    References:
        Rumor spreading time scales with uninformed population remaining.
        See: people.cs.georgetown.edu for multi-source diffusion theory.
    """
    if n <= 1:
        return 0
    
    eps = 1e-9
    
    if normalized:
        # For normalized Laplacian, λ₂ is in [0, 2], no clamping needed
        lam = max(lambda2, eps)
    else:
        # For unnormalized Laplacian, clamp to [eps, 1.0] to avoid issues
        lam = max(min(lambda2, 1.0), eps)
    
    if lambda2 <= eps:
        return int(max(1, math.ceil(C * math.log(n) * n)))
    
    # Adjust for multi-source diffusion if initial informed fraction is provided
    if initial_informed_fraction is not None and 0 < initial_informed_fraction < 1:
        # Use log((1-p)⁻¹ · n) = log(n / (1-p)) instead of log(n)
        # This scales with uninformed population remaining
        # For p=1/3: log(3/2 · n) = log(3/2) + log(n) ≈ 0.405 + log(n)
        # The constant term (~log(3) ≈ 1.1 for p=1/3) improves typical-case accuracy
        # while maintaining worst-case scaling (still O(log N))
        uninformed_fraction = 1.0 - initial_informed_fraction
        log_term = math.log(n / uninformed_fraction)
    else:
        # Standard single-source formula: log(n)
        log_term = math.log(n)
    
    # Base prediction
    t_base = C * log_term / lam
    
    # Apply safety margin if specified (multiplies by (1 + margin))
    if safety_margin > 0:
        t_base = t_base * (1.0 + safety_margin)
    
    # Round up (ceil) and ensure minimum of 1 round
    # This ensures predictions err on the side of caution
    return int(max(1, math.ceil(t_base)))


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


def cheeger_bounds(
    neighbours: Sequence[Sequence[int]], 
    normalized: bool = True
) -> tuple[float, float]:
    """Return (lower, upper) bounds on λ₂ via conductance proxy.

    We approximate conductance Φ using the Fiedler vector induced cut.
    For normalized Laplacian: φ²/2 ≤ λ₂ ≤ 2φ (Cheeger inequality).
    For unnormalized Laplacian: returns mixing time bounds 1/(2Φ) and 1/Φ².
    
    Args:
        neighbours: Adjacency list
        normalized: If True, use normalized Laplacian (default, correct for Cheeger bounds).
    
    Returns:
        (lower_bound, upper_bound) on λ₂ for normalized Laplacian, or mixing time bounds for unnormalized.
    """
    n = len(neighbours)
    if n == 0:
        return (0.0, 0.0)
    
    if normalized:
        # Build normalized Laplacian for Cheeger bounds
        A = torch.zeros((n, n), dtype=torch.float64)
        for i, nbrs in enumerate(neighbours):
            for j in nbrs:
                if 0 <= j < n and j != i:
                    A[i, j] = 1.0
                    A[j, i] = 1.0
        
        deg = A.sum(dim=1)
        deg = torch.clamp(deg, min=1.0)  # Avoid division by zero
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(deg))
        L_norm = torch.eye(n, dtype=torch.float64) - D_inv_sqrt @ A @ D_inv_sqrt
        
        try:
            evals, evecs = torch.linalg.eigh(L_norm)
        except RuntimeError:
            L_norm = L_norm.to(torch.float32)
            evals, evecs = torch.linalg.eigh(L_norm)
    else:
        # Fallback to unnormalized (may not satisfy Cheeger bounds)
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
    
    if normalized:
        # For normalized Laplacian, Cheeger inequality is: φ²/2 ≤ λ₂ ≤ 2φ
        # Return bounds on λ₂ itself
        return (float(phi * phi / 2.0), float(2.0 * phi))
    else:
        # For unnormalized Laplacian, return mixing time bounds
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
    from .metrics import conductance_estimate as _conductance_estimate

    return _conductance_estimate(neighbours, trials=trials, seed=seed)
