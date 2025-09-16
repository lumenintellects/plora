import math
from swarm.metrics import (
    coverage,
    entropy_avg,
    mutual_information,
    spectral_gap,
    mi_series,
    mi_deltas,
)
from scripts.net_it_metrics import compute_net_it


def _make_knowledge_unique_domains(N: int):
    # agent i owns domain_i only
    return {i: {f"d{i}"} for i in range(N)}


def _make_knowledge_full(N: int, D: int):
    doms = [f"d{i}" for i in range(D)]
    return {i: set(doms) for i in range(N)}


def test_entropy_and_mi_initial():
    N = 5
    know = _make_knowledge_unique_domains(N)
    cov = coverage(know)
    # Each domain present in exactly 1/5 agents
    assert all(math.isclose(p, 1 / N) for p in cov.values())

    H = entropy_avg(coverage_map=cov)
    # Check >0
    assert H > 0

    I0 = mutual_information(know)
    assert math.isclose(I0, math.log2(N), rel_tol=1e-6)


def test_entropy_and_mi_full_diffusion():
    N = 5
    D = 5
    know = _make_knowledge_full(N, D)
    cov = coverage(know)
    # p_d ==1
    assert all(p == 1.0 for p in cov.values())
    H = entropy_avg(coverage_map=cov)
    assert H == 0.0
    I = mutual_information(know)
    assert I == 0.0


def test_spectral_gap_positive_on_connected_graph():
    # Simple path graph of 5 nodes has positive algebraic connectivity
    n = 5
    adj = {i: [j for j in (i - 1, i + 1) if 0 <= j < n] for i in range(n)}
    lam2 = spectral_gap([adj[i] for i in range(n)])
    assert lam2 > 0.0


def test_mi_series_and_deltas():
    # Start with unique domains per agent (high MI), then converge to full diffusion (MI=0)
    N = 4
    history = []
    # round 0: unique
    history.append({i: {f"d{i}"} for i in range(N)})
    # round 1: two agents share
    history.append({0: {"d0", "d1"}, 1: {"d1"}, 2: {"d2"}, 3: {"d3"}})
    # round 2: all share everything
    doms = {f"d{i}" for i in range(N)}
    history.append({i: set(doms) for i in range(N)})
    series = mi_series(history)
    deltas = mi_deltas(history)
    assert len(series) == 3 and len(deltas) == 3
    assert series[-1] == 0.0
    # Initial MI > later MI
    assert series[0] > series[1] > series[2]


def test_compute_net_it_smoke():
    # Build tiny history with 3 agents over 3 rounds
    history = [
        {0: ["a"], 1: ["b"], 2: ["c"]},
        {0: ["a", "b"], 1: ["b"], 2: ["c"]},
        {0: ["a", "b", "c"], 1: ["a", "b", "c"], 2: ["a", "b", "c"]},
    ]
    res = compute_net_it(history, n_boot=50, n_perm=16)
    assert "agents" in res and "mi_matrix" in res and "te_matrix" in res
    assert len(res["agents"]) == 3
