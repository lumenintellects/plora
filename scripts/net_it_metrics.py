from __future__ import annotations

"""Compute network-level MI/TE across agents and rounds with CIs and BH-FDR.

Assumes a JSON history of knowledge per round (agent_id -> list[str] of domains).
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from plora.stats import bh_fdr
from plora.it_estimators import mi_knn
from plora.te import transfer_entropy_discrete


def _adaptive_k(n_samples: int) -> int:
    # Heuristic: k ≈ sqrt(N) clamped to [3, 10]
    if n_samples <= 5:
        return 3
    k = int(max(3, min(10, int(n_samples**0.5))))
    return k


def _adaptive_bins(x: np.ndarray) -> int:
    # Freedman–Diaconis with conservative caps; fallback to Sturges
    n = x.shape[0]
    if n <= 1:
        return 4
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    if iqr <= 0:
        # Sturges
        return int(max(4, min(16, np.ceil(np.log2(max(2, n))) + 1)))
    width = 2 * iqr * (n ** (-1 / 3))
    if width <= 0:
        return 8
    bins = int(np.ceil((x.max() - x.min()) / width))
    return int(max(4, min(16, bins)))


def _perm_test_mi(
    x: np.ndarray, y: np.ndarray, k: int, n_perm: int = 64, seed: int = 42
) -> float:
    rng = np.random.default_rng(seed)
    obs = mi_knn(x[:, None], y[:, None], k=k)
    cnt = 0
    for _ in range(n_perm):
        yp = rng.permutation(y)
        val = mi_knn(x[:, None], yp[:, None], k=k)
        if val >= obs:
            cnt += 1
    p = (cnt + 1) / (n_perm + 1)
    return float(p)


def _perm_test_te(
    a: np.ndarray, b: np.ndarray, k: int, bins: int, n_perm: int = 64, seed: int = 42
) -> float:
    rng = np.random.default_rng(seed)
    obs = transfer_entropy_discrete(a, b, k=k, bins=bins)
    cnt = 0
    for _ in range(n_perm):
        bp = rng.permutation(b)
        val = transfer_entropy_discrete(a, bp, k=k, bins=bins)
        if val >= obs:
            cnt += 1
    p = (cnt + 1) / (n_perm + 1)
    return float(p)


def compute_net_it(
    history: List[Dict[int, List[str]]], *, n_boot: int = 200, n_perm: int = 64
) -> dict:
    # Convert to domain count time series per agent
    agents = sorted(history[0].keys()) if history else []
    T = len(history)
    counts = np.array(
        [[len(history[t][a]) for t in range(T)] for a in agents], dtype=float
    )
    # Adaptive parameters
    k = _adaptive_k(T)
    # Pairwise MI and TE
    N = counts.shape[0]
    mi_mat = np.zeros((N, N), dtype=float)
    mi_ci: Dict[str, Tuple[float, float]] = {}
    mi_pvals: List[float] = []
    te_mat = np.zeros((N, N), dtype=float)
    te_pvals: List[float] = []
    bins = _adaptive_bins(counts.flatten()) if counts.size else 8

    # Bootstrap helper for MI per pair (resample time indices)
    rng = np.random.default_rng(123)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            x = counts[i]
            y = counts[j]
            mi_val = mi_knn(x[:, None], y[:, None], k=k)
            mi_mat[i, j] = mi_val
            # bootstrap CI
            if T > 1:
                means = []
                for _ in range(n_boot):
                    idx = rng.integers(0, T, size=T)
                    means.append(mi_knn(x[idx, None], y[idx, None], k=k))
                means.sort()
                lo = means[int(0.025 * n_boot)]
                hi = means[int(0.975 * n_boot) - 1]
            else:
                lo = hi = mi_val
            mi_ci[f"{agents[i]}-{agents[j]}"] = (float(lo), float(hi))
            # permutation p-value
            mi_pvals.append(_perm_test_mi(x, y, k=k, n_perm=n_perm))
            # TE A->B
            te_val = transfer_entropy_discrete(x, y, k=1, bins=bins)
            te_mat[i, j] = te_val
            te_pvals.append(_perm_test_te(x, y, k=1, bins=bins, n_perm=n_perm))

    # Apply BH-FDR separately to MI and TE p-values
    mi_thresh, mi_rej = bh_fdr(mi_pvals, alpha=0.05)
    te_thresh, te_rej = bh_fdr(te_pvals, alpha=0.05)

    return {
        "agents": agents,
        "adaptive": {"k": k, "bins": int(bins)},
        "mi_matrix": mi_mat.tolist(),
        "mi_ci": {k: [float(v[0]), float(v[1])] for k, v in mi_ci.items()},
        "mi_pvals": mi_pvals,
        "mi_bh": {"threshold": mi_thresh, "rejected": mi_rej},
        "te_matrix": te_mat.tolist(),
        "te_pvals": te_pvals,
        "te_bh": {"threshold": te_thresh, "rejected": te_rej},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--history",
        type=Path,
        required=True,
        help="JSON file with list of knowledge snapshots",
    )
    ap.add_argument("--out", type=Path, default=Path("results/net_it_metrics.json"))
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--perm", type=int, default=64)
    ns = ap.parse_args()

    hist = json.loads(ns.history.read_text())
    result = compute_net_it(
        [{int(k): list(v) for k, v in round_map.items()} for round_map in hist],
        n_boot=ns.boot,
        n_perm=ns.perm,
    )
    ns.out.parent.mkdir(parents=True, exist_ok=True)
    ns.out.write_text(json.dumps(result, indent=2))
    print(f"Wrote network IT metrics to {ns.out}")


if __name__ == "__main__":
    main()
