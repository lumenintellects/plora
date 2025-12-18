from __future__ import annotations

from plora.stats import bootstrap_ci_mean


def test_bootstrap_ci_shrinks_with_sample_size():
    small = [1, 2, 3, 4, 5]
    large = small * 10
    lo_s, hi_s = bootstrap_ci_mean(small, n_resamples=500, seed=0)
    lo_l, hi_l = bootstrap_ci_mean(large, n_resamples=500, seed=0)
    width_s = hi_s - lo_s
    width_l = hi_l - lo_l
    assert width_l <= width_s
