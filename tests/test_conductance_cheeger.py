from __future__ import annotations

from swarm.graph_v2 import erdos_renyi_graph
from swarm.metrics import spectral_gap, conductance_estimate
from swarm.theory import cheeger_bounds, epidemic_threshold


def test_conductance_correlates_with_lambda2():
    g1 = erdos_renyi_graph(40, 0.15, seed=1)
    g2 = erdos_renyi_graph(40, 0.35, seed=2)
    phi1 = conductance_estimate(g1, trials=64)
    phi2 = conductance_estimate(g2, trials=64)
    lam1 = spectral_gap(g1)
    lam2 = spectral_gap(g2)
    # denser graph should tend to higher conductance and spectral gap
    assert phi2 >= phi1
    assert lam2 >= lam1


def test_cheeger_bounds_and_epidemic_threshold_smoke():
    n = 8
    # ring graph
    ring = {i: [(i - 1) % n, (i + 1) % n] for i in range(n)}
    lower, upper = cheeger_bounds([ring[i] for i in range(n)])
    assert lower >= 0 and upper >= lower
    beta_c = epidemic_threshold([ring[i] for i in range(n)])
    assert beta_c > 0
