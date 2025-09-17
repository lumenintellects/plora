from __future__ import annotations

"""Calibrate diffusion constant C in t ≈ C log(n) / λ2 for push–pull gossip.

Runs Swarm v2 in-process for multiple sizes and topologies, estimates C per run,
and writes a compact JSONL/JSON summary of results.
"""

import argparse
import json
import random
from pathlib import Path
import hashlib
from typing import List

from plora.agent import Agent, make_dummy_adapter
from swarm.swarm_v2 import run_gossip
from swarm.graph_v2 import (
    erdos_renyi_graph,
    watts_strogatz_graph,
    barabasi_albert_graph,
)
from swarm.metrics import spectral_gap
from plora.config import get as cfg

_DOMAINS_DEFAULT = ["arithmetic", "legal", "medical"]


def build_graph(topology: str, n: int, p: float, k: int, m: int, seed: int):
    if topology == "er":
        return erdos_renyi_graph(n, p, seed)
    if topology == "ws":
        return watts_strogatz_graph(n, k, p, seed)
    if topology == "ba":
        return barabasi_albert_graph(n, m, seed)
    raise ValueError("unknown topology")


def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topology", choices=["er", "ws", "ba"], required=True)
    ap.add_argument(
        "--ns", type=lambda s: [int(x) for x in s.split(",")], required=True
    )
    ap.add_argument("--p", type=float, default=cfg("graph.p", 0.25), help="ER/WS probability")
    ap.add_argument("--k", type=int, default=cfg("graph.ws_k", 4), help="WS k")
    ap.add_argument("--m", type=int, default=cfg("graph.ba_m", 2), help="BA m")
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, required=True)
    ns = ap.parse_args(argv)

    rng = random.Random(ns.seed)
    records: List[dict] = []
    for n in ns.ns:
        nbrs = build_graph(ns.topology, n, ns.p, ns.k, ns.m, seed=ns.seed + n)
        lam2 = spectral_gap(nbrs)
        # Build agents
        agents: List[Agent] = []
        for i in range(n):
            dom = _DOMAINS_DEFAULT[i % len(_DOMAINS_DEFAULT)]
            root = Path(".calib_tmp") / f"agent_{ns.topology}_{n}_{i}"
            ad = make_dummy_adapter(dom, root)
            ag = Agent(i, dom, ad, root_dir=root)
            agents.append(ag)

        history: List[dict[int, set[str]]] = []

        def _on_round(t: int, accepted_events: list[tuple[int, int, str]]):
            know = {ag.agent_id: set(ag.knowledge) for ag in agents}
            history.append(know)

        # Run
        import asyncio

        asyncio.run(
            run_gossip(
                agents,
                rounds=ns.rounds,
                p=ns.p,
                seed=ns.seed,
                neighbours=nbrs,
                on_round=_on_round,
            )
        )

        # Determine observed t_all (first round with full coverage across domains)
        t_all = None
        from swarm.metrics import coverage

        if history:
            for t, know in enumerate(history):
                cov = coverage(know)
                if all(pv == 1.0 for pv in cov.values()):
                    t_all = t
                    break
        # Estimate C
        import math

        C_hat = (
            (t_all * lam2 / math.log(max(2, n)))
            if (t_all is not None and lam2 > 0)
            else None
        )
        records.append(
            {
                "topology": ns.topology,
                "n": n,
                "lambda2": lam2,
                "t_all": t_all,
                "C_hat": C_hat,
                "params": {"p": ns.p, "k": ns.k, "m": ns.m},
            }
        )

    ns.out.parent.mkdir(parents=True, exist_ok=True)
    ns.out.write_text(json.dumps(records, indent=2))
    print(f"Wrote {len(records)} calibration records to {ns.out}")


if __name__ == "__main__":
    main()
