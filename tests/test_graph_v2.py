from __future__ import annotations

from collections import deque

from swarm.graph_v2 import (
    erdos_renyi_graph,
    watts_strogatz_graph,
    barabasi_albert_graph,
    weighted_from_unweighted,
    apply_temporal_dropout,
)


def _is_connected(adj):
    n = len(adj)
    seen = [False] * n
    q = deque([0])
    seen[0] = True
    while q:
        u = q.popleft()
        for v in adj[u]:
            if not seen[v]:
                seen[v] = True
                q.append(v)
    return all(seen)


def test_erdos_renyi_connectivity_and_symmetry():
    n = 20
    p = 0.25
    adj = erdos_renyi_graph(n, p, seed=123)

    assert len(adj) == n
    # symmetry and no self-loops
    for i in range(n):
        assert i not in adj[i]
        for j in adj[i]:
            assert i in adj[j]

    # our generator enforces connectivity via link-to-previous fallback
    assert _is_connected(adj)


def test_watts_strogatz_basic_properties():
    n = 30
    k = 4
    beta = 0.2
    adj = watts_strogatz_graph(n, k, beta, seed=321)
    # no self loops and symmetry
    for i in range(n):
        assert i not in adj[i]
        for j in adj[i]:
            assert i in adj[j]
    # approximate average degree ≈ k
    avg_deg = sum(len(adj[i]) for i in range(n)) / n
    assert 0.5 * k <= avg_deg <= 2.0 * k


def test_barabasi_albert_basic_properties():
    n = 40
    m = 2
    adj = barabasi_albert_graph(n, m, seed=999)
    # symmetry
    for i in range(n):
        assert i not in adj[i]
        for j in adj[i]:
            assert i in adj[j]
    # min degree at least m (except initial conditions can enforce this exactly)
    mind = min(len(adj[i]) for i in range(n))
    assert mind >= m


def test_weighted_and_temporal_helpers():
    n = 6
    adj = erdos_renyi_graph(n, 0.5, seed=1)
    wadj = weighted_from_unweighted(adj, weight=2.0)
    # weights applied and shape consistent
    for i in range(n):
        assert len(wadj[i]) == len(adj[i])
        for j, w in wadj[i]:
            assert isinstance(j, int) and isinstance(w, float)
    # temporal dropout reduces edges on odd t
    adj2 = apply_temporal_dropout(adj, t=1)
    assert sum(len(x) for x in adj2) <= sum(len(x) for x in adj)
