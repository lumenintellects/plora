from plora.metrics import paired_wilcoxon


def _simulate_deltas(delta_mean: float, n: int = 40):
    """Return synthetic per-example deltas centred at *delta_mean*."""
    return [delta_mean for _ in range(n)]


def test_r8_beats_r2():
    """Sanity check: larger negative delta (better) should be more significant."""

    deltas_r2 = _simulate_deltas(-0.1)
    deltas_r8 = _simulate_deltas(-0.3)

    p_r2 = paired_wilcoxon(deltas_r2)["p"]
    p_r8 = paired_wilcoxon(deltas_r8)["p"]

    assert p_r8 < p_r2, "Higher-capacity LoRA should have stronger significance"
