from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from plora.mine import mine_estimate, MineConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rho", type=float, default=0.8)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=Path("results/mine_calib.json"))
    ns = ap.parse_args()

    rng = np.random.default_rng(0)
    X = rng.standard_normal((ns.n, 2))
    E = rng.standard_normal((ns.n, 2))
    Y = ns.rho * X + np.sqrt(1 - ns.rho**2) * E
    X_t = torch.from_numpy(X).float()
    Y_t = torch.from_numpy(Y).float()

    mi, _ = mine_estimate(X_t, Y_t, cfg=MineConfig(epochs=200, batch_size=256))
    ns.out.parent.mkdir(parents=True, exist_ok=True)
    ns.out.write_text(json.dumps({"rho": ns.rho, "mi": mi}, indent=2))
    print(f"MINE estimate: {mi:.3f} nats saved to {ns.out}")


if __name__ == "__main__":
    main()
