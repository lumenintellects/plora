from __future__ import annotations

"""Push-pull gossip driver for Swarm Sim v2 (edge-driven rounds).

"""

import asyncio
from typing import Callable, List, Optional

from swarm.graph_v2 import erdos_renyi_graph, apply_temporal_dropout


async def push_pull_round(
    agents,
    neighbours: List[List[int]],
    *,
    max_inflight: int | None = None,
    on_accept: Optional[Callable[[int, int, str], None]] = None,
):
    sem = asyncio.Semaphore(max_inflight) if max_inflight and max_inflight > 0 else None

    async def talk(u: int, v: int):
        dom_u, ad_u = agents[u].best_offer_for(agents[v])
        dom_v, ad_v = agents[v].best_offer_for(agents[u])
        if ad_u is not None and dom_u is not None:
            ok = await agents[v].accept(ad_u, dom_u)
            if ok and on_accept is not None:
                on_accept(u, v, dom_u)
        if ad_v is not None and dom_v is not None:
            ok = await agents[u].accept(ad_v, dom_v)
            if ok and on_accept is not None:
                on_accept(v, u, dom_v)

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
    neighbours: Optional[List[List[int]]] = None,
    on_round: Optional[Callable[[int, List[tuple[int, int, str]]], None]] = None,
):
    nbrs = (
        neighbours
        if neighbours is not None
        else erdos_renyi_graph(len(agents), p=p, seed=seed)
    )
    for t in range(rounds):
        # Simple temporal variation: apply dropout pattern over time
        nbrs_t = apply_temporal_dropout(nbrs, t)
        accepted_events: List[tuple[int, int, str]] = []

        def _ev(u: int, v: int, dom: str) -> None:
            accepted_events.append((u, v, dom))

        await push_pull_round(agents, nbrs_t, max_inflight=max_inflight, on_accept=_ev)
        if on_round is not None:
            try:
                on_round(t, accepted_events)
            except Exception:
                pass
