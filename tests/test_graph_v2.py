from __future__ import annotations

from collections import deque

from swarm.graph_v2 import erdos_renyi_graph


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
