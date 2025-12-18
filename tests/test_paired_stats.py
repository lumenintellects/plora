import random

from plora.metrics import paired_wilcoxon, bootstrap_ci


def test_wilcoxon_significance():
    # Construct synthetic deltas where treatment always better (negative NLL)
    deltas = [-1.0 for _ in range(30)]
    stats = paired_wilcoxon(deltas)
    assert stats["p"] < 0.001


def test_bootstrap_ci_contains_true_delta():
    random.seed(42)
    xs = [random.random() for _ in range(100)]
    ys = [x - 0.5 for x in xs]  # true mean delta = -0.5
    low, high = bootstrap_ci(xs, ys, n_resamples=500)
    assert low < -0.4 and high < -0.1
