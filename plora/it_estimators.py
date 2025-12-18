from __future__ import annotations

"""Information-theoretic estimators.

Implements a Kraskov-Stögbauer-Grassberger (KSG) kNN estimator for mutual
information between two continuous random vectors X and Y.

References:
  - Kraskov, A., Stögbauer, H. & Grassberger, P., 2004. 
  Estimating mutual information. Physical Review E, 69(6), p.066138. 
  Available at: http://dx.doi.org/10.1103/PhysRevE.69.066138

Defaults are chosen for small-N CPU runs in CI.
"""

import math

import numpy as np


def _digamma_scalar(x: float) -> float:
    """Approximate digamma ψ(x) with SciPy fallback; scalar input only."""
    try:
        from scipy.special import digamma as _sp_digamma

        return float(_sp_digamma(x))
    except Exception:
        # Recurrence to shift x to sufficiently large value
        result = 0.0
        xv = float(x)
        while xv < 8.0:
            result -= 1.0 / xv
            xv += 1.0
        # Asymptotic expansion
        inv = 1.0 / xv
        inv2 = inv * inv
        return (
            result
            + math.log(xv)
            - 0.5 * inv
            - inv2 * (1.0 / 12.0)
            + inv2 * inv2 * (1.0 / 120.0)
        )


def _digamma(x):
    """Vectorized digamma over numpy arrays or scalars."""
    import numpy as _np

    if _np.isscalar(x):
        return _digamma_scalar(float(x))
    arr = _np.asarray(x, dtype=float)
    vec = _np.vectorize(_digamma_scalar, otypes=[float])
    return vec(arr)


def mi_knn(
    X: np.ndarray,
    Y: np.ndarray,
    k: int = 5,
    *,
    epsilon: float | None = None,
) -> float:
    """Estimate mutual information I(X;Y) using KSG kNN estimator (KSG1).

    Parameters
    ----------
    X, Y : np.ndarray
        Arrays of shape (N, dx) and (N, dy). Will be converted to float64.
    k : int
        k-nearest neighbor parameter (default 5).
    epsilon : float | None
        Optional small jitter to break ties.

    Returns
    -------
    float
        Estimated MI in nats.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    if Y.ndim == 1:
        Y = Y[:, None]
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of samples")
    N = X.shape[0]
    if N <= k:
        # For tiny inputs, return zero MI to keep downstream metrics stable
        return 0.0

    # Small jitter to avoid duplicate distances in Chebyshev metric
    if epsilon is None:
        epsilon = 1e-10
    rng = np.random.default_rng(42)
    Xj = X + epsilon * rng.standard_normal(size=X.shape)
    Yj = Y + epsilon * rng.standard_normal(size=Y.shape)

    Z = np.concatenate([Xj, Yj], axis=1)

    # Chebyshev distances (L_infty) for KSG1
    # Compute pairwise max-norm distances in joint space efficiently for small N
    def cheb_dist(A: np.ndarray) -> np.ndarray:
        # returns matrix D where D[i,j] = ||A[i]-A[j]||_inf
        # vectorized via broadcasting; O(N^2 * d) acceptable for small N
        diff = np.abs(A[:, None, :] - A[None, :, :])
        return diff.max(axis=2)

    Dz = cheb_dist(Z)
    # For each i find epsilon_i = distance to k-th neighbor (exclude self)
    # argsort along axis=1, take k+1 due to self at 0
    idx = np.argpartition(Dz, kth=k, axis=1)
    eps_i = Dz[np.arange(N), idx[:, k]]

    # Count neighbors strictly within eps_i in marginals
    Dx = cheb_dist(Xj)
    Dy = cheb_dist(Yj)
    nx = (Dx < eps_i[:, None]).sum(axis=1) - 1  # exclude self
    ny = (Dy < eps_i[:, None]).sum(axis=1) - 1

    psi_k = _digamma(k)
    psi_N = _digamma(N)
    # KSG1 estimator (nats)
    est = psi_k + psi_N - float(np.mean(_digamma(nx + 1) + _digamma(ny + 1)))
    return float(est)
