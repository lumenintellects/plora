"""CLI entry point for Swarm Sim v1.

Usage examples:
    python -m swarm.sim_entry --topology line --agents 5 --mode sim --max_rounds 50 --seed 42
    python -m swarm.sim_entry --topology mesh --agents 5 --mode sim --max_rounds 20 --seed 1
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import random
from pathlib import Path

from swarm.graph_engine import GraphEngine, build_topology
from swarm.gossip_node import GossipNode

from plora.agent import Agent, AdapterInfo
from plora.manifest import Manifest

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("swarm.sim_entry")

_DOMAINS_DEFAULT = ["arithmetic", "legal", "medical"]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Swarm Sim graph study")
    p.add_argument("--topology", choices=["line", "mesh"], required=True)
    p.add_argument("--agents", type=int, default=5)
    p.add_argument("--mode", choices=["sim", "full"], default="sim")
    p.add_argument("--max_rounds", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--results_dir", default="results", type=Path)
    return p.parse_args()


async def _main_async(ns: argparse.Namespace) -> None:
    rng = random.Random(ns.seed)
    topo = build_topology(ns.topology, ns.agents)

    nodes: list[GossipNode] = []
    # Create dummy agents (sim-only mode populates knowledge directly)
    for i in range(ns.agents):
        domain = _DOMAINS_DEFAULT[i % len(_DOMAINS_DEFAULT)]
        # Create a minimal dummy manifest object (not used further in sim-only)
        dummy_manifest = Manifest(
            schema_version=0,
            plasmid_id=f"dummy-{i}",
            domain=domain,
            base_model="dummy/base",
            peft_format="lora",
            lora={"r": 1, "alpha": 1, "dropout": 0.0, "target_modules": []},
            artifacts={
                "filename": "adapter_model.bin",
                "sha256": "0" * 64,
                "size_bytes": 1,
            },
            train_meta={
                "seed": 0,
                "epochs": 0,
                "dataset_id": "none",
                "sample_count": 0,
                "timestamp_unix": 0,
            },
            metrics={
                "val_ppl_before": 0.0,
                "val_ppl_after": 0.0,
                "delta_ppl": 0.0,
                "val_em": None,
                "val_chrf": None,
            },
            safety={"licence": "CC0", "poisoning_score": 0.0},
            signer={
                "algo": "none",
                "pubkey_fingerprint": "none",
                "signature_b64": "",
            },
            compatibility={"peft_min": "0", "transformers": "0"},
        )

        dummy_adapter = AdapterInfo(Path("/dev/null"), dummy_manifest, 0)
        agent = Agent(agent_id=i, domain=domain, adapter=dummy_adapter)
        agent.knowledge.add(domain)
        node = GossipNode(
            agent=agent,
            agent_id=i,
            neighbours=topo[i],
            mode=ns.mode,
            rand=random.Random(rng.randint(0, 2 ** 32 - 1)),
        )
        nodes.append(node)

    engine = GraphEngine(
        nodes,
        topology_kind=ns.topology,
        domains=_DOMAINS_DEFAULT,
        max_rounds=ns.max_rounds,
        seed=ns.seed,
        report_dir=ns.results_dir,
    )
    await engine.run()


def main() -> None:
    ns = _parse_args()
    asyncio.run(_main_async(ns))


if __name__ == "__main__":
    main()
