from __future__ import annotations

import numpy as np

from plora.te import transfer_entropy_discrete, transfer_entropy_continuous


def _ar_process(n=1000, rho=0.8, noise=0.1, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(n)
    b = np.zeros(n)
    for t in range(1, n):
        b[t] = rho * a[t - 1] + (1 - rho) * b[t - 1] + noise * rng.standard_normal()
    return a, b


def test_te_directionality_ar():
    a, b = _ar_process(n=800)
    te_ab_d = transfer_entropy_discrete(a, b, k=1, bins=10)
    te_ba_d = transfer_entropy_discrete(b, a, k=1, bins=10)
    assert te_ab_d > te_ba_d

    te_ab_c = transfer_entropy_continuous(a, b, k=1, knn_k=5)
    te_ba_c = transfer_entropy_continuous(b, a, k=1, knn_k=5)
    assert te_ab_c > te_ba_c
