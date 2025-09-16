from __future__ import annotations

import numpy as np
import torch

from plora.mine import mine_estimate, MineConfig


def test_mine_gaussian_recovers_mi_within_tolerance():
    rng = np.random.default_rng(0)
    n = 1500
    rho = 0.7
    X = rng.standard_normal((n, 2))
    E = rng.standard_normal((n, 2))
    Y = rho * X + np.sqrt(1 - rho**2) * E
    X_t = torch.from_numpy(X).float()
    Y_t = torch.from_numpy(Y).float()
    mi, _ = mine_estimate(X_t, Y_t, cfg=MineConfig(epochs=150, batch_size=256))
    # True MI for Gaussian: -0.5 * log(1 - rho^2) per dimension; here dims=2
    true_mi = -0.5 * np.log(1 - rho**2) * 2
    assert abs(mi - true_mi) < 0.4
