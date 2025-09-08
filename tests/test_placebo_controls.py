from plora.metrics import paired_wilcoxon


def test_placebo_not_significant():
    """Synthetic sanity check: placebo deltas centred on 0 should not be significant."""

    baseline = [1.0] * 50
    placebo = [1.0 + (-1) ** i * 0.01 for i in range(50)]  # tiny noise around baseline
    deltas = [p - b for p, b in zip(placebo, baseline)]
    p = paired_wilcoxon(deltas)["p"]
    assert p > 0.05, "Placebo should not beat baseline significantly"
