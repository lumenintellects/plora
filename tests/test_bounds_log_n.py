from __future__ import annotations

from swarm.graph_v2 import erdos_renyi_graph
from swarm.metrics import spectral_gap
from swarm.theory import predicted_rounds_spectral


def test_predicted_rounds_scale_with_log_n():
    # Build two ER graphs of sizes n and 2n with same p; predicted rounds for 2n
    # should be at least as large, and roughly grow with log(2n)/log(n).
    p = 0.2
    n1 = 50
    n2 = 100
    g1 = erdos_renyi_graph(n1, p, seed=123)
    g2 = erdos_renyi_graph(n2, p, seed=456)
    lam1 = spectral_gap(g1)
    lam2 = spectral_gap(g2)
    t1 = predicted_rounds_spectral(n1, lam1)
    t2 = predicted_rounds_spectral(n2, lam2)
    assert t2 >= t1
    # sanity: growth factor near log(2n)/log(n) (~1.26 for n=50)
    ratio = t2 / max(1, t1)
    assert ratio <= 3.0  # very loose upper bound for noisy gaps
