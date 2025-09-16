"""Metrics for Swarm Sim diffusion dynamics.

Implements:
* coverage per domain p_d(t)
* average entropy H_avg(t)
* mutual information I_t between (Agent,Domain) matrix
* rounds-to-diffuse t_d and t_all

All functions are pure and rely only on built-in modules.  They accept the
minimal state representation - a mapping agent_id -> set[str] of domains held.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, Iterable, List, Mapping, Sequence, Set
import torch

__all__ = [
    "coverage",
    "entropy_avg",
    "mutual_information",
    "rounds_to_diffuse",
    "spectral_gap",
    "mi_series",
    "mi_deltas",
    "cooccurrence_excess",
    "pid_lite_summary",
    "agent_agent_mi",
    "network_flow_rate",
    "conductance_estimate",
]


def _domains_from_knowledge(knowledge: Mapping[int, Set[str]]) -> List[str]:
    seen: set[str] = set()
    for doms in knowledge.values():
        seen.update(doms)
    return sorted(seen)


def coverage(
    knowledge: Mapping[int, Set[str]],
    domains: Sequence[str] | None = None,
) -> Dict[str, float]:
    """Return p_d = |agents possessing d| / N for each domain."""
    N = len(knowledge)
    if N == 0:
        raise ValueError("Knowledge must contain at least one agent.")
    if domains is None:
        domains = _domains_from_knowledge(knowledge)
    cov: Dict[str, float] = {d: 0.0 for d in domains}
    for doms in knowledge.values():
        for d in doms:
            if d in cov:
                cov[d] += 1.0
    for d in domains:
        cov[d] /= N
    return cov


def entropy_avg(
    coverage_map: Mapping[str, float] | None = None,
    *,
    knowledge: Mapping[int, Set[str]] | None = None,
    domains: Sequence[str] | None = None,
) -> float:
    """Compute mean binary entropy over domains.

    You can pass either ``coverage_map`` directly or raw ``knowledge``.
    """
    if coverage_map is None:
        if knowledge is None:
            raise ValueError("Either coverage_map or knowledge must be provided.")
        coverage_map = coverage(knowledge, domains)
    entropies: list[float] = []
    for p in coverage_map.values():
        if p in (0.0, 1.0):
            entropies.append(0.0)
        else:
            entropies.append(-(p * math.log2(p) + (1 - p) * math.log2(1 - p)))
    return sum(entropies) / len(entropies) if entropies else 0.0


def mutual_information(
    knowledge: Mapping[int, Set[str]],
    domains: Sequence[str] | None = None,
) -> float:
    """Mutual information I(A;D) between Agent and Domain presence.

    Definition in prompt:
        p(a, d) = I[a has d] / sum_{a,d} I[a has d]
        p(a)     = 1 / N
        p(d)     = k_d / N   (k_d - agents possessing d)
    """
    if not knowledge:
        raise ValueError("Knowledge cannot be empty.")
    N = len(knowledge)
    if domains is None:
        domains = _domains_from_knowledge(knowledge)
    # Count k_d
    k_d: Counter[str] = Counter()
    total_pos = 0
    for doms in knowledge.values():
        for d in doms:
            k_d[d] += 1
            total_pos += 1
    if total_pos == 0:
        return 0.0

    # If every agent has every domain, MI is 0 by definition (variables independent)
    if all(k == N for k in k_d.values()):
        return 0.0

    mi = 0.0
    inv_total = 1.0 / total_pos
    inv_N = 1.0 / N
    for a, doms in knowledge.items():
        for d in doms:
            p_ad = inv_total  # 1/total_pos
            p_a = inv_N
            p_d = k_d[d] / N
            mi += p_ad * math.log2(p_ad / (p_a * p_d))
    return mi


def rounds_to_diffuse(
    history: Sequence[Mapping[int, Set[str]]],
    domains: Sequence[str] | None = None,
) -> Dict[str, int | None]:
    """Given per-round knowledge snapshots, return first round each domain fully diffuses.

    ``history`` is a list where index is round number and value is knowledge
    at that round.
    Returns mapping domain -> t_d (round index) or None if never reached.
    """
    if not history:
        raise ValueError("History cannot be empty.")
    N = len(next(iter(history[0].values()))) if history else 0
    if domains is None:
        domains = _domains_from_knowledge(history[0])
    result: Dict[str, int | None] = {d: None for d in domains}
    for t, know in enumerate(history):
        cov = coverage(know, domains)
        for d, p in cov.items():
            if p == 1.0 and result[d] is None:
                result[d] = t
    return result


def spectral_gap(neighbours: Sequence[Sequence[int]]) -> float:
    """Return algebraic connectivity (λ2) of the graph Laplacian.

    neighbours: adjacency list (undirected). For disconnected graphs, returns 0.0.
    """
    n = len(neighbours)
    if n == 0:
        return 0.0
    # Build adjacency and degree
    A = torch.zeros((n, n), dtype=torch.float64)
    for i, nbrs in enumerate(neighbours):
        for j in nbrs:
            if 0 <= j < n and j != i:
                A[i, j] = 1.0
                A[j, i] = 1.0
    deg = torch.diag(A.sum(dim=1))
    L = deg - A
    try:
        evals = torch.linalg.eigvalsh(L)
    except RuntimeError:
        # Fallback to float32 if float64 fails
        L = L.to(torch.float32)
        evals = torch.linalg.eigvalsh(L)
    evals, _ = torch.sort(evals)
    if evals.numel() < 2:
        return 0.0
    lam2 = float(evals[1].item())
    if lam2 < 0 and lam2 > -1e-9:
        lam2 = 0.0
    return lam2


def mi_series(
    history: Sequence[Mapping[int, Set[str]]], domains: Sequence[str] | None = None
) -> List[float]:
    """Return mutual information per round for a knowledge history."""
    return [mutual_information(know, domains) for know in history]


def mi_deltas(
    history: Sequence[Mapping[int, Set[str]]], domains: Sequence[str] | None = None
) -> List[float]:
    """Return per-round information gain ΔI_t = I_t - I_{t-1}. First round has 0."""
    series = mi_series(history, domains)
    if not series:
        return []
    deltas: List[float] = [0.0]
    for i in range(1, len(series)):
        deltas.append(series[i] - series[i - 1])
    return deltas


def cooccurrence_excess(knowledge: Mapping[int, Set[str]], d1: str, d2: str) -> float:
    """Return p(d1 & d2) - p(d1) p(d2) across agents (positive => synergy).

    Uses domain presence across agents as Bernoulli variables.
    """
    N = len(knowledge)
    if N == 0:
        return 0.0
    k1 = sum(1 for doms in knowledge.values() if d1 in doms)
    k2 = sum(1 for doms in knowledge.values() if d2 in doms)
    k12 = sum(1 for doms in knowledge.values() if (d1 in doms and d2 in doms))
    p1 = k1 / N
    p2 = k2 / N
    p12 = k12 / N
    return float(p12 - p1 * p2)


def pid_lite_summary(
    knowledge: Mapping[int, Set[str]], domains: Sequence[str] | None = None
) -> dict:
    """Compute simple synergy/redundancy proxies over domain pairs.

    For each pair (d1,d2):
      synergy ≈ max(0, p12 - p1 p2)
      redundancy ≈ max(0, min(p1, p2) - p12)
    Returns aggregate means across pairs.
    """
    if domains is None:
        domains = _domains_from_knowledge(knowledge)
    if len(domains) < 2:
        return {"synergy_mean": 0.0, "redundancy_mean": 0.0}
    Npairs = 0
    syn_sum = 0.0
    red_sum = 0.0
    N = len(knowledge)
    # precompute coverages
    cov = coverage(knowledge, domains)
    for i in range(len(domains)):
        for j in range(i + 1, len(domains)):
            d1, d2 = domains[i], domains[j]
            p1 = cov[d1]
            p2 = cov[d2]
            k12 = sum(1 for doms in knowledge.values() if (d1 in doms and d2 in doms))
            p12 = k12 / N
            syn = max(0.0, p12 - p1 * p2)
            red = max(0.0, min(p1, p2) - p12)
            syn_sum += syn
            red_sum += red
            Npairs += 1
    if Npairs == 0:
        return {"synergy_mean": 0.0, "redundancy_mean": 0.0}
    return {"synergy_mean": syn_sum / Npairs, "redundancy_mean": red_sum / Npairs}


def agent_agent_mi(knowledge: Mapping[int, Set[str]]) -> float:
    """MI between Agent and Agent's domain set cardinality as a simple proxy.

    Treat variable A as agent id (uniform) and Dcount as number of domains held.
    """
    N = len(knowledge)
    if N == 0:
        return 0.0
    from collections import Counter

    counts = Counter(len(doms) for doms in knowledge.values())
    total = sum(counts.values())
    p_d = {k: v / total for k, v in counts.items()}
    import math

    # p(a,d) uniform over agents; approximate with 1/N for each agent
    mi = 0.0
    for dcount, pd in p_d.items():
        p_ad = pd / N
        p_a = 1.0 / N
        mi += N * p_ad * math.log2((p_ad) / (p_a * pd))
    return mi


def network_flow_rate(history: Sequence[Mapping[int, Set[str]]]) -> float:
    """Average per-round increase in mutual information over history."""
    if not history:
        return 0.0
    series = mi_series(history)
    if len(series) < 2:
        return 0.0
    deltas = [series[i] - series[i - 1] for i in range(1, len(series))]
    return float(sum(deltas) / len(deltas))


def conductance_estimate(
    neighbours: Sequence[Sequence[int]], trials: int = 64, seed: int = 42
) -> float:
    """Approximate conductance Φ(G) via random cuts.

    Φ(S) = cut(S, \bar S) / min(vol(S), vol(\bar S)). We sample random indicator vectors
    to form S and take the minimum Φ over trials.
    """
    import random

    n = len(neighbours)
    if n == 0:
        return 0.0
    deg = [len(neighbours[i]) for i in range(n)]
    vol_total = sum(deg)
    rng = random.Random(seed)
    best_phi = float("inf")
    for _ in range(trials):
        S = {i for i in range(n) if rng.random() < 0.5}
        if not S or len(S) == n:
            continue
        Sc = set(range(n)) - S
        cut = 0
        volS = 0
        for u in S:
            volS += deg[u]
            for v in neighbours[u]:
                if v in Sc:
                    cut += 1
        volSc = vol_total - volS
        denom = max(1, min(volS, volSc))
        phi = cut / denom
        if phi < best_phi:
            best_phi = phi
    if best_phi == float("inf"):
        return 0.0
    return float(best_phi)
