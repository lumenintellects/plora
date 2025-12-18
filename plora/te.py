from __future__ import annotations

"""Transfer Entropy estimators.

Implements:
- Discrete TE via histogram binning and conditional entropies
"""

from typing import Tuple
import numpy as np


def _hist_counts(*cols: np.ndarray, bins: int) -> Tuple[np.ndarray, Tuple]:
    data = np.stack(cols, axis=1)
    hist, edges = np.histogramdd(data, bins=bins)
    return hist, edges


def _entropy_from_counts(counts: np.ndarray) -> float:
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def transfer_entropy_discrete(
    series_a: np.ndarray,
    series_b: np.ndarray,
    *,
    k: int = 1,
    bins: int = 8,
) -> float:
    """Estimate TE A→B for discrete (binned) series.

    TE(A→B) = H(B_t | B_{t-1..t-k}) - H(B_t | B_{t-1..t-k}, A_{t-1..t-k}).
    """
    a = np.asarray(series_a, dtype=float)
    b = np.asarray(series_b, dtype=float)
    n = min(a.shape[0], b.shape[0])
    if n <= k:
        return 0.0
    a = a[:n]
    b = b[:n]
    # Build lag matrices
    bt = b[k:]
    b_past = np.stack([b[k - i - 1 : n - i - 1] for i in range(k)], axis=1)
    a_past = np.stack([a[k - i - 1 : n - i - 1] for i in range(k)], axis=1)

    # Discretize by binning
    bt_d = np.digitize(bt, np.linspace(bt.min(), bt.max(), bins + 1)[1:-1])
    bp_d = np.stack(
        [
            np.digitize(
                b_past[:, i],
                np.linspace(b_past[:, i].min(), b_past[:, i].max(), bins + 1)[1:-1],
            )
            for i in range(k)
        ],
        axis=1,
    )
    ap_d = np.stack(
        [
            np.digitize(
                a_past[:, i],
                np.linspace(a_past[:, i].min(), a_past[:, i].max(), bins + 1)[1:-1],
            )
            for i in range(k)
        ],
        axis=1,
    )

    # H(B_t | B_past)
    counts_joint, _ = _hist_counts(bt_d, *[bp_d[:, i] for i in range(k)], bins=bins)
    counts_bp, _ = _hist_counts(*[bp_d[:, i] for i in range(k)], bins=bins)
    H_bt_bp = _entropy_from_counts(counts_joint) - _entropy_from_counts(counts_bp)

    # H(B_t | B_past, A_past)
    counts_joint2, _ = _hist_counts(
        bt_d,
        *[bp_d[:, i] for i in range(k)],
        *[ap_d[:, i] for i in range(k)],
        bins=bins,
    )
    counts_cond2, _ = _hist_counts(
        *[bp_d[:, i] for i in range(k)], *[ap_d[:, i] for i in range(k)], bins=bins
    )
    H_bt_bp_ap = _entropy_from_counts(counts_joint2) - _entropy_from_counts(
        counts_cond2
    )

    return float(H_bt_bp - H_bt_bp_ap)
