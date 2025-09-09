from __future__ import annotations

"""Push-pull gossip driver for Swarm Sim v2 (edge-driven rounds).

This module operates directly over in-process Agent objects for simplicity.
It complements the v1 socket-based GossipNode for experiments that focus on
graph-level diffusion rather than transport mechanics.
"""

import asyncio
from typing import List

from swarm.graph_v2 import erdos_renyi_graph


async def push_pull_round(
    agents, neighbours: List[List[int]], *, max_inflight: int | None = None
):
    sem = asyncio.Semaphore(max_inflight) if max_inflight and max_inflight > 0 else None

    async def talk(u: int, v: int):
        dom_u, ad_u = agents[u].best_offer_for(agents[v])
        dom_v, ad_v = agents[v].best_offer_for(agents[u])
        if ad_u is not None and dom_u is not None:
            await agents[v].accept(ad_u, dom_u)
        if ad_v is not None and dom_v is not None:
            await agents[u].accept(ad_v, dom_v)

    tasks = []
    seen = set()
    for u in range(len(agents)):
        for v in neighbours[u]:
            if (v, u) in seen:
                continue
            seen.add((u, v))
            if sem is None:
                tasks.append(asyncio.create_task(talk(u, v)))
            else:

                async def guarded(u=u, v=v):
                    async with sem:
                        await talk(u, v)

                tasks.append(asyncio.create_task(guarded()))
    if tasks:
        await asyncio.gather(*tasks)


async def run_gossip(
    agents,
    rounds: int,
    *,
    p: float = 0.25,
    seed: int = 42,
    max_inflight: int | None = None,
):
    nbrs = erdos_renyi_graph(len(agents), p=p, seed=seed)
    for _ in range(rounds):
        await push_pull_round(agents, nbrs, max_inflight=max_inflight)
