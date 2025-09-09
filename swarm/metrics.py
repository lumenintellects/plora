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

__all__ = [
    "coverage",
    "entropy_avg",
    "mutual_information",
    "rounds_to_diffuse",
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
