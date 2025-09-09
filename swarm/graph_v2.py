from __future__ import annotations

"""Erdos-Renyi overlay utilities for Swarm Sim v2.

We avoid external deps and provide a small helper to generate an undirected
G(n, p) graph with a quick connectivity pass to avoid isolated nodes.
"""

import random
from typing import List


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
