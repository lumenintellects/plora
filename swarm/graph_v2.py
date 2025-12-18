from __future__ import annotations

"""Graph overlays for Swarm Sim v2.

Includes:
- Erdos–Renyi G(n,p) with connectivity fixups
- Watts–Strogatz small-world graph
- Barabási–Albert preferential attachment graph
"""

import random
from typing import List, Tuple


def erdos_renyi_graph(n: int, p: float, seed: int) -> List[List[int]]:
    rnd = random.Random(seed)
    nbrs: List[List[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if rnd.random() < p:
                nbrs[i].append(j)
                nbrs[j].append(i)
    # Ensure no isolated vertices by linking to previous
    for i in range(1, n):
        if not nbrs[i]:
            nbrs[i].append(i - 1)
            nbrs[i - 1].append(i)
    # Ensure connectivity: link component representatives into a chain
    seen = [False] * n
    comps: List[int] = []  # representative indices

    def dfs(start: int):
        stack = [start]
        seen[start] = True
        while stack:
            u = stack.pop()
            for v in nbrs[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)

    for i in range(n):
        if not seen[i]:
            comps.append(i)
            dfs(i)
    # Connect components sequentially if more than one
    for i in range(1, len(comps)):
        a = comps[i - 1]
        b = comps[i]
        if b not in nbrs[a]:
            nbrs[a].append(b)
            nbrs[b].append(a)
    return nbrs


def watts_strogatz_graph(n: int, k: int, beta: float, seed: int) -> List[List[int]]:
    """Return WS small-world graph adjacency (undirected) with n nodes.

    Start from ring lattice where each node connects to k/2 neighbors on each side.
    Rewire each edge (u->v) with probability beta to a random node, avoiding self-loops
    and duplicates. Ensures symmetry in the resulting adjacency lists.
    """
    rnd = random.Random(seed)
    if k % 2 == 1:
        k = k + 1  # enforce even k
    k = max(2, min(k, n - 1))
    nbrs: List[List[int]] = [[] for _ in range(n)]
    # ring lattice
    half = k // 2
    for i in range(n):
        for d in range(1, half + 1):
            j = (i + d) % n
            nbrs[i].append(j)
            nbrs[j].append(i)
    # rewire clockwise edges
    for i in range(n):
        for d in range(1, half + 1):
            j = (i + d) % n
            if rnd.random() < beta:
                # remove existing (i,j)
                if j in nbrs[i]:
                    nbrs[i].remove(j)
                if i in nbrs[j]:
                    nbrs[j].remove(i)
                # pick new target
                candidates = set(range(n)) - {i} - set(nbrs[i])
                if not candidates:
                    # restore original edge if no candidates
                    nbrs[i].append(j)
                    nbrs[j].append(i)
                    continue
                new_j = rnd.choice(list(candidates))
                nbrs[i].append(new_j)
                nbrs[new_j].append(i)
    return nbrs


def barabasi_albert_graph(n: int, m: int, seed: int) -> List[List[int]]:
    """Return BA preferential attachment adjacency (undirected).

    Start with a small connected core of size m+1 as a clique, then add nodes
    preferentially attaching to existing nodes with probability proportional to degree.
    """
    rnd = random.Random(seed)
    m = max(1, min(m, n - 1))
    nbrs: List[List[int]] = [[] for _ in range(n)]
    # initial clique of size m+1
    init = m + 1
    for i in range(init):
        for j in range(i + 1, init):
            nbrs[i].append(j)
            nbrs[j].append(i)
    # degree list for attachment
    degrees = [len(nbrs[i]) for i in range(init)] + [0] * (n - init)
    total_deg = sum(degrees)

    def pick_target() -> int:
        x = rnd.random() * total_deg
        acc = 0.0
        for idx, d in enumerate(degrees):
            acc += d
            if x <= acc:
                return idx
        return len(degrees) - 1

    for v in range(init, n):
        targets = set()
        while len(targets) < m:
            t = pick_target()
            if t != v and t not in targets:
                targets.add(t)
        for u in targets:
            nbrs[v].append(u)
            nbrs[u].append(v)
        degrees[v] = len(nbrs[v])
        total_deg += degrees[v]
        for u in targets:
            degrees[u] += 1
            total_deg += 1
    return nbrs


def weighted_from_unweighted(
    adj: List[List[int]], weight: float = 1.0
) -> List[List[Tuple[int, float]]]:
    """Convert adjacency list to weighted adjacency with constant weight."""
    return [[(j, float(weight)) for j in nbrs] for nbrs in adj]


def apply_temporal_dropout(
    adj: List[List[int]], t: int, period: int = 2
) -> List[List[int]]:
    """Return a time-varying adjacency by dropping every other edge set by period.

    For demonstration: on even t, keep original; on odd t, drop edges with odd indices.
    """
    if period <= 0:
        return adj
    if t % period == 0:
        return adj
    out: List[List[int]] = []
    for nbrs in adj:
        out.append([j for idx, j in enumerate(nbrs) if idx % 2 == 0])
    return out
