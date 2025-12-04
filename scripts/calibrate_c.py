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
    ap.add_argument(
        "--p", type=float, default=cfg("graph.p", 0.25), help="ER/WS probability"
    )
    ap.add_argument("--k", type=int, default=cfg("graph.ws_k", 4), help="WS k")
    ap.add_argument("--m", type=int, default=cfg("graph.ba_m", 2), help="BA m")
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, required=True)
    ns = ap.parse_args(argv)

    rng = random.Random(ns.seed)
    records: List[dict] = []
    base_n = min(ns.ns)
    for n in ns.ns:
        nbrs = build_graph(ns.topology, n, ns.p, ns.k, ns.m, seed=ns.seed + n)
        lam2 = spectral_gap(nbrs, normalized=True)
        # Build agents
        agents: List[Agent] = []
        for i in range(n):
            dom = _DOMAINS_DEFAULT[i % len(_DOMAINS_DEFAULT)]
            root = Path(".calib_tmp") / f"agent_{ns.topology}_{n}_{i}"
            ad = make_dummy_adapter(dom, root)
            ag = Agent(i, dom, ad, root_dir=root)
            agents.append(ag)

        history: List[dict[int, set[str]]] = []
        # Capture initial knowledge before any gossip rounds so t=0 represents the
        # pre-diffusion state (otherwise instantaneous diffusion would yield t_all=0).
        history.append({ag.agent_id: set(ag.knowledge) for ag in agents})

        def _on_round(t: int, accepted_events: list[tuple[int, int, str]]):
            know = {ag.agent_id: set(ag.knowledge) for ag in agents}
            history.append(know)

        # Scale rounds for larger graphs (ensure enough horizon for diffusion)
        rounds_for_n = max(int(ns.rounds * max(1, n / base_n)), ns.rounds)

        # Run
        import asyncio

        asyncio.run(
            run_gossip(
                agents,
                rounds=rounds_for_n,
                p=ns.p,
                seed=ns.seed,
                neighbours=nbrs,
                on_round=_on_round,
            )
        )

        # Determine observed t_all (first round with ≥90% coverage across domains, as per specification)
        t_all = None
        from swarm.metrics import coverage

        if history:
            for t, know in enumerate(history):
                cov = coverage(know)
                # Use 90% coverage threshold as per specification (≥90% coverage)
                if all(pv >= 0.9 for pv in cov.values()):
                    t_all = t
                    break
        # Estimate C using multi-source diffusion formula
        # For 3-domain setup: p = 1/3, so use log((1-p)⁻¹ · n) = log(3/2 · n)
        import math
        
        # Multi-source adjustment: each domain starts in ~n/3 agents (p=1/3)
        initial_informed_fraction = 1.0 / len(_DOMAINS_DEFAULT)  # p = 1/3 for 3 domains
        uninformed_fraction = 1.0 - initial_informed_fraction  # 1-p = 2/3
        log_term = math.log(max(2, n) / uninformed_fraction)  # log(n / (1-p)) = log(3/2 · n)
        
        C_hat = (
            (t_all * lam2 / log_term)
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
                "rounds": rounds_for_n,
                "params": {"p": ns.p, "k": ns.k, "m": ns.m},
            }
        )

    ns.out.parent.mkdir(parents=True, exist_ok=True)
    ns.out.write_text(json.dumps(records, indent=2))
    print(f"Wrote {len(records)} calibration records to {ns.out}")


if __name__ == "__main__":
    main()
