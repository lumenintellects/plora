from __future__ import annotations

"""Simple bootstrap confidence intervals for metrics."""

from typing import List, Tuple
import random


def bootstrap_ci_mean(
    values: List[float], *, n_resamples: int = 1000, ci: float = 0.95, seed: int = 42
) -> Tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(((1 - ci) / 2) * n_resamples)]
    hi = means[int((1 - (1 - ci) / 2) * n_resamples) - 1]
    return (float(lo), float(hi))


def bh_fdr(pvals: List[float], alpha: float = 0.05) -> Tuple[float, List[bool]]:
    """Benjamini–Hochberg FDR control.

    Returns (threshold, rejected_mask) where rejected_mask[i] indicates p_i <= threshold
    under the BH procedure. Stable for small lists; ties resolved conservatively.
    """
    m = len(pvals)
    if m == 0:
        return 0.0, []
    # sort with original indices
    pairs = sorted((p, i) for i, p in enumerate(pvals))
    thresh = 0.0
    k_star = 0
    for rank, (p, _) in enumerate(pairs, start=1):
        crit = alpha * rank / m
        if p <= crit:
            k_star = rank
            thresh = p
    rejected = [False] * m
    if k_star > 0:
        # set rejected for those with p <= thresh
        for p, i in pairs:
            if p <= thresh:
                rejected[i] = True
            else:
                break
    return float(thresh), rejected
