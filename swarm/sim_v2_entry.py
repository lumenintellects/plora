from __future__ import annotations

"""CLI entry for Swarm Sim v2 (push-pull, in-process agents).

Example:
    python -m swarm.sim_v2_entry --agents 6 --rounds 5 --graph_p 0.25 --seed 42
"""

import argparse
import asyncio
import hashlib
import os
import random
import tempfile
from pathlib import Path
import json

from plora.agent import Agent, AdapterInfo
from plora.manifest import Manifest
from swarm.swarm_v2 import run_gossip
from plora.gate import Policy
from plora.targets import ATTENTION_SUFFIXES
from plora.backdoor import mark_trojan
from swarm.graph_v2 import (
    erdos_renyi_graph,
    watts_strogatz_graph,
    barabasi_albert_graph,
)
from swarm.metrics import (
    coverage as cov_fn,
    entropy_avg,
    mutual_information as mi_fn,
    rounds_to_diffuse as rtd_fn,
    spectral_gap as spectral_gap_fn,
)
from swarm.theory import predicted_rounds_spectral
from swarm.consensus import ConsensusEngine


_DOMAINS_DEFAULT = ["arithmetic", "legal", "medical"]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Swarm Sim v2 push–pull gossip")
    p.add_argument("--agents", type=int, default=6)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--graph_p", type=float, default=0.25)
    p.add_argument("--graph", choices=["er", "ws", "ba"], default="er")
    p.add_argument("--ws_k", type=int, default=4, help="WS: degree parameter k")
    p.add_argument("--ws_beta", type=float, default=0.2, help="WS: rewiring prob")
    p.add_argument("--ba_m", type=int, default=2, help="BA: new edges per node m")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--max_inflight",
        type=int,
        default=0,
        help="Cap concurrent edge talks per round (0=unbounded)",
    )
    p.add_argument(
        "--report_dir",
        type=Path,
        default=Path("results"),
        help="Directory to write a v2 JSON report",
    )
    p.add_argument("--security", choices=["on", "off"], default="off")
    p.add_argument(
        "--consensus",
        choices=["on", "off"],
        default="off",
        help="Enable in-process quorum consensus gating",
    )
    p.add_argument(
        "--quorum",
        type=int,
        default=2,
        help="Consensus quorum size when --consensus=on",
    )
    p.add_argument(
        "--allowed_targets",
        type=str,
        default="attention",
        help="One of attention|all; whitelist for security policy",
    )
    p.add_argument(
        "--allowed_targets_file",
        type=Path,
        default=None,
        help="Optional path to newline-separated allowed target suffixes",
    )
    p.add_argument(
        "--allowed_ranks",
        type=str,
        default="4,8,16",
        help="Comma-separated allowed ranks, e.g. 4,8,16",
    )
    p.add_argument(
        "--trojan_rate",
        type=float,
        default=0.0,
        help="Fraction [0,1] of agents to mark as trojan at init",
    )
    p.add_argument(
        "--signatures",
        choices=["on", "off"],
        default="off",
        help="Enable RSA signature verification in policy",
    )
    p.add_argument(
        "--trusted_pubkeys",
        type=str,
        default="",
        help="Comma-separated paths to trusted public key PEM files",
    )
    p.add_argument(
        "--policy_file",
        type=Path,
        default=None,
        help="Optional JSON policy file to load",
    )
    p.add_argument("--tau_trigger", type=float, default=None)
    p.add_argument("--tau_norm_z", type=float, default=None)
    p.add_argument("--tau_clean_delta", type=float, default=None)
    return p.parse_args()


def _make_dummy_adapter(domain: str, root: Path) -> AdapterInfo:
    dom_dir = root / domain
    dom_dir.mkdir(parents=True, exist_ok=True)
    model_path = dom_dir / "adapter_model.safetensors"
    payload = f"dummy-{domain}".encode()
    model_path.write_bytes(payload)
    (dom_dir / "adapter_config.json").write_text("{}")
    sha = hashlib.sha256(payload).hexdigest()
    man = Manifest(
        schema_version=0,
        plasmid_id=f"dummy-{sha[:8]}",
        domain=domain,
        base_model="dummy/base",
        peft_format="lora",
        lora={"r": 1, "alpha": 1, "dropout": 0.0, "target_modules": []},
        artifacts={
            "filename": model_path.name,
            "sha256": sha,
            "size_bytes": len(payload),
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
        signer={"algo": "none", "pubkey_fingerprint": "none", "signature_b64": ""},
        compatibility={"peft_min": "0", "transformers": "0"},
    )
    man.dump(dom_dir / "plora.yml")
    return AdapterInfo(model_path, man, len(payload))


async def _main_async(ns: argparse.Namespace) -> None:
    rng = random.Random(ns.seed)

    agents: list[Agent] = []
    # Build policy if enabled
    policy = None
    if ns.security == "on":
        # Resolve allowed targets
        targets = None
        if ns.allowed_targets_file is not None:
            if ns.allowed_targets_file.exists():
                targets = [
                    line.strip()
                    for line in ns.allowed_targets_file.read_text().splitlines()
                    if line.strip()
                ]
        elif ns.allowed_targets == "attention":
            targets = ATTENTION_SUFFIXES
        ranks = tuple(int(x) for x in ns.allowed_ranks.split(",") if x)
        trusted = [Path(p) for p in ns.trusted_pubkeys.split(",") if p]
        if ns.policy_file is not None and ns.policy_file.exists():
            policy = Policy.from_file(ns.policy_file)
        else:
            policy = Policy(
                base_model="dummy/base",
                allowed_ranks=ranks,
                allowed_targets=targets,
                signatures_enabled=(ns.signatures == "on"),
                trusted_public_keys=trusted if trusted else None,
            )
        if ns.consensus == "on":
            policy.consensus_enabled = True
        # override thresholds if provided
        if ns.tau_trigger is not None:
            policy.tau_trigger = ns.tau_trigger
        if ns.tau_norm_z is not None:
            policy.tau_norm_z = ns.tau_norm_z
        if ns.tau_clean_delta is not None:
            policy.tau_clean_delta = ns.tau_clean_delta
    # Build minimal agents with dummy manifests; v2 runs in-process and reuses
    # Agent.accept() for copying semantics even in sim mode.
    tmp_root = Path(tempfile.mkdtemp(prefix="swarm_v2_"))
    for i in range(ns.agents):
        dom = _DOMAINS_DEFAULT[i % len(_DOMAINS_DEFAULT)]
        agent_root = tmp_root / f"agent_{i}"
        adapter = _make_dummy_adapter(dom, agent_root)
        # Probabilistically mark as trojan for evaluation
        if ns.trojan_rate > 0.0 and rng.random() < ns.trojan_rate:
            mark_trojan(adapter.path.parent)
        ag = Agent(
            agent_id=i,
            domain=dom,
            adapter=adapter,
            root_dir=agent_root,
            security_policy=policy,
        )
        agents.append(ag)

    # Wire a shared in-process consensus engine if enabled
    engine = None
    if (
        policy is not None
        and getattr(policy, "consensus_enabled", False)
        and ns.consensus == "on"
    ):
        engine = ConsensusEngine(quorum=max(1, int(ns.quorum)))
        for ag in agents:
            ag.consensus_engine = engine

    # Build neighbours per selected topology
    if ns.graph == "er":
        nbrs = erdos_renyi_graph(len(agents), p=ns.graph_p, seed=ns.seed)
        topo_name = "erdos_renyi"
    elif ns.graph == "ws":
        nbrs = watts_strogatz_graph(
            len(agents), k=ns.ws_k, beta=ns.ws_beta, seed=ns.seed
        )
        topo_name = "watts_strogatz"
    else:
        nbrs = barabasi_albert_graph(len(agents), m=ns.ba_m, seed=ns.seed)
        topo_name = "barabasi_albert"

    # Prepare per-round metrics capture
    domains = list({ag.domain for ag in agents})
    lam2 = spectral_gap_fn(nbrs)
    round_logs: list[dict] = []
    history: list[dict[int, set[str]]] = []
    prev_I: float | None = None

    def _on_round(t: int, accepted_events: list[tuple[int, int, str]]) -> None:
        know = {ag.agent_id: set(ag.knowledge) for ag in agents}
        history.append(know)
        cov = cov_fn(know, domains)
        H_avg = entropy_avg(coverage_map=cov)
        I_t = mi_fn(know, domains)
        mi_delta = (I_t - prev_I) if (prev_I is not None) else 0.0
        prev_I = I_t
        round_logs.append(
            {
                "t": t,
                "coverage": cov,
                "entropy_avg": H_avg,
                "mutual_information": I_t,
                "accepted": [(int(u), int(v), str(d)) for (u, v, d) in accepted_events],
                "mi_delta": mi_delta,
            }
        )

    await run_gossip(
        agents,
        ns.rounds,
        p=ns.graph_p,
        seed=ns.seed,
        max_inflight=(ns.max_inflight or None),
        neighbours=nbrs,
        on_round=_on_round,
    )

    # Build simple v2 report (unified fields where possible)
    domains = list({ag.domain for ag in agents})
    coverage = {
        d: sum(1.0 for ag in agents if d in ag.knowledge) / float(len(agents))
        for d in domains
    }
    reasons = {}
    for ag in agents:
        for k, v in getattr(ag, "rejection_reasons", {}).items():
            reasons[k] = reasons.get(k, 0) + v
    report = {
        "meta": {
            "topology": topo_name,
            "N": len(agents),
            "domains": domains,
            "seed": ns.seed,
            "p": ns.graph_p,
            "lambda2": lam2,
        },
        "rounds": round_logs,
        "final": {
            "coverage": coverage,
            "bytes_on_wire": 0,
            "accepted_offers": sum(getattr(ag, "accepted", 0) for ag in agents),
            "gate": {
                "rejected_hash_total": sum(
                    getattr(ag, "rejected_hash", 0) for ag in agents
                ),
                "rejected_safety_total": sum(
                    getattr(ag, "rejected_safety", 0) for ag in agents
                ),
                "accepted_clean_total": sum(
                    getattr(ag, "accepted_clean", 0) for ag in agents
                ),
                "accepted_trojan_total": sum(
                    getattr(ag, "accepted_trojan", 0) for ag in agents
                ),
                "rejected_clean_total": sum(
                    getattr(ag, "rejected_clean", 0) for ag in agents
                ),
                "rejected_trojan_total": sum(
                    getattr(ag, "rejected_trojan", 0) for ag in agents
                ),
                "false_negatives": sum(
                    getattr(ag, "accepted_trojan", 0) for ag in agents
                ),
                "false_positives": sum(
                    getattr(ag, "rejected_clean", 0) for ag in agents
                ),
                "rejection_reasons": reasons,
            },
        },
    }
    if engine is not None:
        try:
            # expose committed artefacts per slot in report for smoke validation
            report["final"]["consensus"] = {
                "committed": {
                    str(k): v for k, v in getattr(engine, "_commit", {}).items()
                }
            }
        except Exception:
            pass
    try:
        # Observed rounds to diffuse per domain
        rtd = rtd_fn(history, domains) if history else {d: None for d in domains}
        report["final"]["rounds_to_diffuse"] = rtd
        report["final"]["observed_t_all"] = (
            max([t for t in rtd.values() if t is not None])
            if any(v is not None for v in rtd.values())
            else None
        )
        report["final"]["predicted_t_all"] = predicted_rounds_spectral(
            len(agents), lam2
        )
    except Exception:
        pass
    try:
        ns.report_dir.mkdir(parents=True, exist_ok=True)
        out = ns.report_dir / f"swarm_v2_report_seed{ns.seed}.json"
        out.write_text(json.dumps(report, indent=2))
        print(f"Saved v2 report to {out}")
    except Exception:
        pass


def main() -> None:
    ns = _parse_args()
    asyncio.run(_main_async(ns))


if __name__ == "__main__":
    main()
