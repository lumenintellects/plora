"""Graph engine orchestrating Swarm Sim rounds.

Responsibilities:
* Build topology (line | mesh) for N agents.
* Spawn one ``GossipNode`` per agent and ensure each starts its server.
* Run synchronous rounds: await ``node.tick(rnd)`` for all nodes in parallel.
* After each round compute metrics and append to history.
* Stop when diffusion complete or max_rounds reached.
* Persist report to JSON.
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Sequence, Set

from swarm.gossip_node import GossipNode
from swarm.metrics import coverage, entropy_avg, mutual_information, rounds_to_diffuse

logger = logging.getLogger(__name__)


def build_topology(kind: str, n: int) -> Dict[int, List[int]]:
    """Return adjacency list mapping node_id -> list(neighbour_ids)."""
    if kind == "line":
        return {
            i: [j for j in (i - 1, i + 1) if 0 <= j < n] for i in range(n)
        }
    if kind == "mesh":
        return {i: [j for j in range(n) if j != i] for i in range(n)}
    raise ValueError(f"Unknown topology kind: {kind}")


class GraphEngine:
    """Coordinator that drives multiple ``GossipNode`` instances."""

    def __init__(
        self,
        nodes: List[GossipNode],
        *,
        topology_kind: str,
        domains: Sequence[str],
        max_rounds: int,
        seed: int,
        report_dir: Path,
    ) -> None:
        self.nodes = nodes
        self.topology_kind = topology_kind
        self.domains = list(domains)
        self.max_rounds = max_rounds
        self.seed = seed
        self.report_dir = report_dir
        self.history: List[Mapping[int, Set[str]]] = []
        self.round_logs: List[dict] = []

    # ------------------------------------------------------------------
    async def run(self) -> Path:
        await asyncio.gather(*(n.start() for n in self.nodes))
        try:
            prev_accepted = 0
            prev_bytes = 0
            for t in range(self.max_rounds + 1):
                # Snapshot knowledge at *start* of round t
                know = {n.agent_id: set(n.agent.knowledge) for n in self.nodes}
                self.history.append(know)
                cov = coverage(know, self.domains)
                H_avg = entropy_avg(coverage_map=cov)
                I_t = mutual_information(know, self.domains)
                accepted_total = sum(n.accepted_offers for n in self.nodes)
                accepted_delta = max(0, accepted_total - prev_accepted)
                prev_accepted = accepted_total
                bytes_total = sum(n.bytes_sent for n in self.nodes)
                bytes_delta = max(0, bytes_total - prev_bytes)
                prev_bytes = bytes_total
                offers_this_round = len(self.nodes) if t > 0 else 0  # v1 tick: one offer per node per round
                acceptance_rate = (accepted_delta / offers_this_round) if offers_this_round > 0 else 0.0
                self.round_logs.append(
                    {
                        "t": t,
                        "coverage": cov,
                        "H_avg": H_avg,
                        "I": I_t,
                        # acceptance/rejection cumulative counters (from Agents)
                        "accepted_total": sum(n.agent.accepted for n in self.nodes),
                        "accepted_clean_total": sum(getattr(n.agent, "accepted_clean", 0) for n in self.nodes),
                        "accepted_trojan_total": sum(getattr(n.agent, "accepted_trojan", 0) for n in self.nodes),
                        "rejected_hash_total": sum(n.agent.rejected_hash for n in self.nodes),
                        "rejected_safety_total": sum(n.agent.rejected_safety for n in self.nodes),
                        "rejected_clean_total": sum(getattr(n.agent, "rejected_clean", 0) for n in self.nodes),
                        "rejected_trojan_total": sum(getattr(n.agent, "rejected_trojan", 0) for n in self.nodes),
                        # per-round deltas
                        "accepted_delta": accepted_delta,
                        "bytes_sent_delta": bytes_delta,
                        "acceptance_rate": acceptance_rate,
                        "bytes_sent_total": bytes_total,
                    }
                )
                # Check stopping condition, full diffusion
                if all(p == 1.0 for p in cov.values()):
                    break
                # Skip tick after last snapshot if we already reached cap
                if t == self.max_rounds:
                    break
                await asyncio.gather(*(n.tick(t) for n in self.nodes))
        finally:
            await asyncio.gather(*(n.close() for n in self.nodes))

        # Compile final metrics
        t_d = rounds_to_diffuse(self.history, self.domains)
        t_all = max(v or self.max_rounds for v in t_d.values())
        bytes_on_wire = sum(n.bytes_sent for n in self.nodes)
        accepted = sum(n.accepted_offers for n in self.nodes)
        MI_drop = self.round_logs[0]["I"] - self.round_logs[-1]["I"]

        # Aggregate rejection reasons from agents
        reasons_hist: MutableMapping[str, int] = defaultdict(int)
        for n in self.nodes:
            for reason, count in getattr(n.agent, "rejection_reasons", {}).items():
                reasons_hist[reason] += count

        report = {
            "meta": {
                "topology": self.topology_kind,
                "N": len(self.nodes),
                "domains": self.domains,
                "seed": self.seed,
                "timestamp": datetime.utcnow().isoformat(),
            },
            "rounds": self.round_logs,
            "final": {
                "t_d": t_d,
                "t_all": t_all,
                "MI_drop": MI_drop,
                "bytes_on_wire": bytes_on_wire,
                "accepted_offers": accepted,
                "gate": {
                    "rejected_hash_total": sum(n.agent.rejected_hash for n in self.nodes),
                    "rejected_safety_total": sum(n.agent.rejected_safety for n in self.nodes),
                    "accepted_clean_total": sum(getattr(n.agent, "accepted_clean", 0) for n in self.nodes),
                    "accepted_trojan_total": sum(getattr(n.agent, "accepted_trojan", 0) for n in self.nodes),
                    "rejected_clean_total": sum(getattr(n.agent, "rejected_clean", 0) for n in self.nodes),
                    "rejected_trojan_total": sum(getattr(n.agent, "rejected_trojan", 0) for n in self.nodes),
                    # FN: trojan accepted; FP: clean rejected
                    "false_negatives": sum(getattr(n.agent, "accepted_trojan", 0) for n in self.nodes),
                    "false_positives": sum(getattr(n.agent, "rejected_clean", 0) for n in self.nodes),
                    "rejection_reasons": dict(reasons_hist),
                },
            },
        }
        if self.report_dir is not None:
            self.report_dir.mkdir(parents=True, exist_ok=True)
            path = self.report_dir / f"swarm_graph_report_{self.topology_kind}_{self.seed}.json"
            path.write_text(json.dumps(report, indent=2))
            logger.info("Saved report to %s", path)
            return path

        # If report_dir is None (e.g. during unit tests), skip writing.
        return Path("/dev/null")
