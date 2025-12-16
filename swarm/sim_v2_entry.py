from __future__ import annotations

"""CLI entry for Swarm Sim v2 (push-pull, in-process agents).

Example:
    python -m swarm.sim_v2_entry --agents 6 --rounds 5 --graph_p 0.25 --seed 42
"""

import argparse
import asyncio
import random
import tempfile
from pathlib import Path
import json
import math

from plora.agent import Agent, make_dummy_adapter, load_real_adapter
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
from plora.config import get as cfg


_DOMAINS_DEFAULT = ["arithmetic", "legal", "medical"]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Swarm Sim v2 push–pull gossip")
    p.add_argument("--agents", type=int, default=6)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--graph_p", type=float, default=cfg("graph.p", 0.25))
    p.add_argument("--graph", choices=["er", "ws", "ba"], default="er")
    p.add_argument(
        "--ws_k", type=int, default=cfg("graph.ws_k", 4), help="WS: degree parameter k"
    )
    p.add_argument(
        "--ws_beta",
        type=float,
        default=cfg("graph.ws_beta", 0.2),
        help="WS: rewiring prob",
    )
    p.add_argument(
        "--ba_m",
        type=int,
        default=cfg("graph.ba_m", 2),
        help="BA: new edges per node m",
    )
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
    p.add_argument(
        "--history-alias",
        type=Path,
        default=None,
        help="Optional path to copy the serialized per-round history JSON.",
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
        default=cfg("swarm.quorum", 2),
        help="Consensus quorum size when --consensus=on",
    )
    p.add_argument(
        "--allowed_targets",
        type=str,
        default=None,
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
        default=None,
        help="Comma-separated allowed ranks, e.g. 4,8,16",
    )
    p.add_argument(
        "--trojan_rate",
        type=float,
        default=cfg("swarm.trojan_rate", 0.0),
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
    p.add_argument(
        "--probes_calib",
        type=Path,
        default=Path("results/probes_calib.json"),
        help="Optional path to probes calibration JSON (tau_* thresholds)",
    )
    p.add_argument("--tau_trigger", type=float, default=None)
    p.add_argument("--tau_norm_z", type=float, default=None)
    p.add_argument("--tau_clean_delta", type=float, default=None)
    # Optionally estimate bytes transferred on each accepted offer
    p.add_argument(
        "--estimate_size",
        choices=["on", "off"],
        default="off",
        help="Estimate bytes_on_wire by summing adapter artifact sizes for each accepted offer",
    )
    # Use real trained adapters instead of dummy ones
    p.add_argument(
        "--adapters_dir",
        type=Path,
        default=None,
        help="Directory containing real trained adapters (e.g., 'out'). Falls back to dummy if adapter not found.",
    )
    return p.parse_args()


def _make_dummy_adapter(domain: str, root: Path):
    dom_dir = root / domain
    return make_dummy_adapter(domain, dom_dir)


async def _main_async(ns: argparse.Namespace) -> None:
    rng = random.Random(ns.seed)

    agents: list[Agent] = []
    # Build policy if enabled
    policy = None
    if ns.security == "on":
        # Resolve allowed targets
        targets = None
        if ns.allowed_targets_file is not None and ns.allowed_targets_file.exists():
            targets = [
                line.strip()
                for line in ns.allowed_targets_file.read_text().splitlines()
                if line.strip()
            ]
        else:
            if ns.allowed_targets in {"attention", "all"}:
                targets = (
                    ATTENTION_SUFFIXES if ns.allowed_targets == "attention" else None
                )
            else:
                tcfg = cfg("allowed_targets")
                if tcfg == "attention":
                    targets = ATTENTION_SUFFIXES
                elif tcfg == "all":
                    targets = None
                elif isinstance(tcfg, list):
                    targets = tcfg
        if ns.allowed_ranks:
            ranks = tuple(int(x) for x in ns.allowed_ranks.split(",") if x)
        else:
            r = cfg("allowed_ranks", [4, 8, 16])
            ranks = tuple(int(x) for x in r)
        trusted = [Path(p) for p in ns.trusted_pubkeys.split(",") if p]
        if ns.policy_file is not None and ns.policy_file.exists():
            policy = Policy.from_file(ns.policy_file)
        else:
            # Use configured base_model to match trained adapters (fixes base_model_mismatch rejections)
            configured_base_model = cfg("base_model", "google/gemma-3-1b-it")
            policy = Policy(
                base_model=configured_base_model,
                allowed_ranks=ranks,
                allowed_targets=targets,
                signatures_enabled=(ns.signatures == "on"),
                trusted_public_keys=trusted if trusted else None,
            )
        # Load probes calibration thresholds if present and not overridden below
        try:
            if ns.probes_calib is not None and ns.probes_calib.exists():
                import json as _json

                _pc = _json.loads(ns.probes_calib.read_text())
                if policy is not None:
                    if (
                        policy.tau_trigger is None
                        and _pc.get("tau_trigger") is not None
                    ):
                        policy.tau_trigger = float(_pc["tau_trigger"])
                    if (
                        policy.tau_clean_delta is None
                        and _pc.get("tau_clean_delta") is not None
                    ):
                        policy.tau_clean_delta = float(_pc["tau_clean_delta"])
                    if (
                        policy.tau_tensor_z is None
                        and _pc.get("tau_tensor_z") is not None
                    ):
                        policy.tau_tensor_z = float(_pc["tau_tensor_z"])
                    if (
                        policy.tau_norm_z is None
                        and _pc.get("tau_tensor_z") is not None
                    ):
                        # fall back to tensor_z if norm_z not present
                        try:
                            policy.tau_norm_z = float(
                                _pc.get("tau_norm_z", _pc["tau_tensor_z"])
                            )
                        except Exception:
                            pass
        except Exception:
            pass
        if ns.consensus == "on":
            policy.consensus_enabled = True
        # override thresholds if provided
        if ns.tau_trigger is not None:
            policy.tau_trigger = ns.tau_trigger
        if ns.tau_norm_z is not None:
            policy.tau_norm_z = ns.tau_norm_z
        if ns.tau_clean_delta is not None:
            policy.tau_clean_delta = ns.tau_clean_delta
    # Build agents with real adapters (if --adapters_dir provided) or dummy manifests.
    # v2 runs in-process and reuses Agent.accept() for copying semantics even in sim mode.
    tmp_root = Path(tempfile.mkdtemp(prefix="swarm_v2_"))
    real_adapter_count = 0
    adapters_dir = getattr(ns, "adapters_dir", None)
    for i in range(ns.agents):
        dom = _DOMAINS_DEFAULT[i % len(_DOMAINS_DEFAULT)]
        agent_root = tmp_root / f"agent_{i}"

        # Try to load real adapter if adapters_dir is specified
        adapter = None
        if adapters_dir is not None:
            real_adapter_path = adapters_dir / dom
            adapter = load_real_adapter(real_adapter_path)
            if adapter is not None:
                real_adapter_count += 1
                # Copy adapter to agent's root for proper isolation
                import shutil
                dest_dir = agent_root / dom
                dest_dir.mkdir(parents=True, exist_ok=True)
                for f in real_adapter_path.iterdir():
                    if f.is_file():
                        shutil.copy2(f, dest_dir / f.name)
                adapter = load_real_adapter(dest_dir)

        # Fall back to dummy adapter
        if adapter is None:
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

    if adapters_dir is not None:
        print(f"[sim_v2] Loaded {real_adapter_count}/{ns.agents} real adapters from {adapters_dir}")

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
    lam2 = spectral_gap_fn(nbrs, normalized=True)
    round_logs: list[dict] = []
    history: list[dict[int, set[str]]] = []
    prev_I: float | None = None
    cum_abs_mi_change: float = 0.0
    # Track total bytes transferred if enabled
    bytes_on_wire: int = 0
    # Robust to tests constructing Namespace without estimate_size
    estimate_enabled = getattr(ns, "estimate_size", "off") == "on"

    def _on_round(t: int, accepted_events: list[tuple[int, int, str]]) -> None:
        nonlocal prev_I, cum_abs_mi_change, bytes_on_wire
        know = {ag.agent_id: set(ag.knowledge) for ag in agents}
        history.append(know)
        cov = cov_fn(know, domains)
        H_avg = entropy_avg(coverage_map=cov)
        I_t = mi_fn(know, domains)
        mi_delta = (I_t - prev_I) if (prev_I is not None) else 0.0
        if prev_I is not None:
            cum_abs_mi_change += abs(mi_delta)
        prev_I = I_t
        mi_loss = -mi_delta if mi_delta < 0 else 0.0
        # If enabled, accumulate bytes for each accepted offer event
        if estimate_enabled and accepted_events:
            for (u, v, dom) in accepted_events:
                try:
                    ad_map = agents[u].shareable_adapters()
                    if dom in ad_map:
                        # Prefer manifest.artifacts.size_bytes if present
                        size = (
                            getattr(ad_map[dom].manifest.artifacts, "size_bytes", None)
                            or ad_map[dom].size_bytes
                        )
                        # guard against non-int
                        size_int = int(size) if isinstance(size, (int, float)) else 0
                        bytes_on_wire += max(0, size_int)
                except Exception:
                    # Ignore size estimation errors (keep simulation robust)
                    pass
        # Normalized MI (specialization index) relative to log2(N * D)
        N_agents = len(agents)
        D = len(domains) if domains else 1
        denom = math.log2(max(2, N_agents * D))  # guard: at least log2(2)=1
        mi_norm = (I_t / denom) if denom > 0 else 0.0
        round_logs.append(
            {
                "t": t,
                "coverage": cov,
                "entropy_avg": H_avg,
                "mutual_information": I_t,
                "mi_delta": mi_delta,
                "mi_loss": mi_loss,
                "mi_cum_abs": cum_abs_mi_change,
                "mi_norm": mi_norm,
                "accepted": [(int(u), int(v), str(d)) for (u, v, d) in accepted_events],
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
            # Bytes transferred (sum of adapter sizes) if estimation enabled
            "bytes_on_wire": bytes_on_wire if estimate_enabled else 0,
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
    # Derive bytes_per_offer metric if possible
    try:
        acc_total = report["final"]["accepted_offers"]
        if acc_total and report["final"]["bytes_on_wire"]:
            report["final"]["bytes_per_offer"] = report["final"]["bytes_on_wire"] / acc_total
        else:
            report["final"]["bytes_per_offer"] = 0
    except Exception:
        report["final"]["bytes_per_offer"] = 0
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
            len(agents), lam2, normalized=True
        )
    except Exception:
        pass
    try:
        ns.report_dir.mkdir(parents=True, exist_ok=True)
        out = ns.report_dir / f"swarm_v2_report_seed{ns.seed}.json"
        out.write_text(json.dumps(report, indent=2))
        print(f"Saved v2 report to {out}")
        # Persist per-round knowledge history for downstream metrics
        hist_payload = [
            {
                str(agent_id): sorted(list(knowledge))
                for agent_id, knowledge in agent_state.items()
            }
            for agent_state in history
        ]
        hist_text = json.dumps(hist_payload, indent=2)
        hist_out = ns.report_dir / f"history_seed{ns.seed}.json"
        hist_out.write_text(hist_text)
        print(f"Saved history to {hist_out}")
        if getattr(ns, "history_alias", None):
            alias_path = ns.history_alias
            alias_path.parent.mkdir(parents=True, exist_ok=True)
            alias_path.write_text(hist_text)
            print(f"Aliased history to {alias_path}")
    except Exception:
        pass


def main() -> None:
    ns = _parse_args()
    asyncio.run(_main_async(ns))


if __name__ == "__main__":
    main()
