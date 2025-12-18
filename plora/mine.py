from __future__ import annotations

"""Variational Mutual Information Neural Estimator (MINE).

CPU-friendly reference with small networks and moving-average bias correction.

References: 
    - Belghazi, M.I., Baratin, A., Rajeswar, S., Ozair, S., Bengio, Y., Courville, A. & Hjelm, R.D., 2021. 
    MINE: Mutual Information Neural Estimation. arXiv preprint arXiv:1801.04062. 
    Available at: https://arxiv.org/abs/1801.04062.
"""

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


class MineNet(nn.Module):
    def __init__(self, dim_x: int, dim_y: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_x + dim_y, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        z = torch.cat([x, y], dim=1)
        return self.net(z)


@dataclass
class MineConfig:
    hidden: int = 64
    lr: float = 5e-4
    ema_rate: float = 0.02
    epochs: int = 300
    batch_size: int = 256
    device: str = "cpu"


def mine_estimate(
    X: torch.Tensor, Y: torch.Tensor, *, cfg: MineConfig | None = None
) -> Tuple[float, MineNet]:
    """Estimate I(X;Y) with MINE (returns nats and the trained net).

    X, Y: tensors of shape [N, dx] and [N, dy].
    """
    if cfg is None:
        cfg = MineConfig()
    device = torch.device(cfg.device)
    X = X.to(device)
    Y = Y.to(device)
    # Standardize inputs for stable training
    with torch.no_grad():
        X = (X - X.mean(dim=0, keepdim=True)) / (X.std(dim=0, keepdim=True) + 1e-6)
        Y = (Y - Y.mean(dim=0, keepdim=True)) / (Y.std(dim=0, keepdim=True) + 1e-6)
    N, dx = X.shape
    dy = Y.shape[1]
    net = MineNet(dx, dy, hidden=cfg.hidden).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)

    # Exponential moving averages for stability and final estimate smoothing
    ma_et = None
    mi_ma: float | None = None
    for _ in range(cfg.epochs):
        perm = torch.randperm(N, device=device)
        for s in range(0, N, cfg.batch_size):
            idx = perm[s : s + cfg.batch_size]
            x = X[idx]
            y = Y[idx]
            # shuffled pairing for product of marginals (fresh perm per batch)
            perm_b = torch.randperm(N, device=device)
            y_sh = Y[perm_b][idx]
            t = net(x, y)  # [B,1]
            s_marg = net(x, y_sh)  # [B,1]
            # Stable log-mean-exp for the marginal term
            lme = torch.logsumexp(s_marg, dim=0) - torch.log(
                torch.tensor(s_marg.shape[0], device=device, dtype=s_marg.dtype)
            )
            et = torch.exp(s_marg)
            # moving average for stability
            with torch.no_grad():
                if ma_et is None:
                    ma_et = et.mean().detach()
                else:
                    ma_et = (
                        1 - cfg.ema_rate
                    ) * ma_et + cfg.ema_rate * et.mean().detach()
            # MINE objective: E[T] - log E[exp(T)] (batch log-mean-exp)
            mi_step = t.mean() - lme.squeeze()
            loss = -mi_step
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            opt.step()
            # Smooth running MI estimate to reduce final upward bias
            with torch.no_grad():
                if mi_ma is None:
                    mi_ma = float(mi_step.item())
                else:
                    mi_ma = float(
                        (1 - cfg.ema_rate) * mi_ma + cfg.ema_rate * mi_step.item()
                    )

    # Final estimate: prefer smoothed EMA if available, else one-shot eval
    with torch.no_grad():
        if mi_ma is not None:
            mi = float(mi_ma)
        else:
            t = net(X, Y).mean()
            mi = float(t.item() - torch.log(ma_et + 1e-8).item())
    return float(mi), net
