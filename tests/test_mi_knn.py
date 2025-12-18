from __future__ import annotations

import numpy as np
from plora.it_estimators import mi_knn


def test_mi_knn_sanity():
    rng = np.random.default_rng(0)
    N = 400
    # X ~ N(0,1), Y = rho X + sqrt(1-rho^2) E
    rho_low = 0.1
    rho_high = 0.9
    X = rng.standard_normal((N, 1))
    E = rng.standard_normal((N, 1))
    Y_low = rho_low * X + np.sqrt(1 - rho_low**2) * E
    Y_high = rho_high * X + np.sqrt(1 - rho_high**2) * E

    mi_low = mi_knn(X, Y_low, k=5)
    mi_high = mi_knn(X, Y_high, k=5)

    assert mi_high > mi_low
    # MI(X;X) should be higher still
    mi_self = mi_knn(X, X, k=5)
    assert mi_self > mi_high
