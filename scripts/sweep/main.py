from __future__ import annotations

"""
Thesis-scale sweep runner for Swarm Sim v2.

This module powers ``poetry run python -m scripts.sweep.main`` and writes a
JSONL file summarising how different network topologies, sizes and gate
settings behave.
"""

import argparse
import asyncio
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Sequence

from plora.agent import Agent, make_dummy_adapter
from plora.backdoor import mark_trojan
from plora.config import get as cfg
from plora.gate import Policy
from plora.targets import ATTENTION_SUFFIXES
from swarm.graph_v2 import barabasi_albert_graph, erdos_renyi_graph, watts_strogatz_graph
from swarm.metrics import coverage as cov_fn
from swarm.metrics import mutual_information as mi_fn
from swarm.metrics import spectral_gap
from swarm.swarm_v2 import run_gossip
from swarm.theory import predicted_rounds_spectral

TMP_ROOT = Path(".sweep_tmp")
DOMAINS = ["arithmetic", "legal", "medical"]


def _load_calibrated_c(topology: str, default_c: float = 2.0) -> float:
    """Load calibrated C value from calibration file.
    
    Args:
        topology: Graph topology (er, ws, ba)
        default_c: Default C value if calibration file not found
        
    Returns:
        Calibrated C value, or default_c if not available
    """
    # Try topology-specific calibration file first
    calib_path = Path(f"results/c_calib_{topology}.json")
    if not calib_path.exists():
        # Fallback to ER calibration (most common)
        calib_path = Path("results/c_calib_er.json")
    
    if calib_path.exists():
        try:
            with calib_path.open() as f:
                calib_data = json.load(f)
            c_values = [c["C_hat"] for c in calib_data if c.get("C_hat") is not None]
            if c_values:
                return float(sum(c_values) / len(c_values))
        except (json.JSONDecodeError, KeyError, ValueError):
            pass
    
    return default_c


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Swarm v2 parameter sweep")
    parser.add_argument("--topos", type=str, default="er,ws,ba")
    parser.add_argument("--ns", type=str, default="20,40")
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(val) for val in cfg("value_add.seeds", [41, 42])),
    )
    parser.add_argument("--p", type=float, default=cfg("graph.p", 0.25))
    parser.add_argument("--ws_k", type=int, default=cfg("graph.ws_k", 4))
    parser.add_argument("--ws_beta", type=float, default=cfg("graph.ws_beta", 0.2))
    parser.add_argument("--ba_m", type=int, default=cfg("graph.ba_m", 2))
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--trojan_rates", type=str, default="0.0,0.3")
    parser.add_argument("--tau_trigger", type=str, default=str(cfg("gate.tau_trigger", 0.2)))
    parser.add_argument("--tau_norm_z", type=str, default=str(cfg("gate.tau_norm_z", 3.0)))
    parser.add_argument(
        "--tau_clean_delta",
        type=str,
        default=str(cfg("gate.tau_clean_delta", -0.05)),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/thesis_sweep.jsonl"),
        help="Destination JSONL file",
    )
    return parser.parse_args(argv)


def _build_graph(topology: str, n: int, p: float, k: int, m: int, seed: int):
    if topology == "er":
        return erdos_renyi_graph(n, p, seed)
    if topology == "ws":
        return watts_strogatz_graph(n, k, p, seed)
    if topology == "ba":
        return barabasi_albert_graph(n, m, seed)
    raise ValueError(f"Unknown topology '{topology}'")


def _env_manifest() -> Dict[str, object]:
    info: Dict[str, object] = {}
    try:
        import platform

        info["platform"] = {
            "python": platform.python_version(),
            "system": platform.system(),
            "release": platform.release(),
        }
    except Exception:
        pass
    for pkg in ("torch", "transformers", "peft", "numpy"):
        try:
            mod = __import__(pkg)
            info[pkg] = {"version": getattr(mod, "__version__", "")}
        except Exception:
            continue
    return info


def _reset_tmp_root(root: Path) -> Path:
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _cleanup_tmp_root(root: Path) -> None:
    shutil.rmtree(root, ignore_errors=True)


def _build_agents(
    n: int,
    policy: Policy,
    trojan_rate: float,
    seed: int,
    tmp_root: Path,
) -> List[Agent]:
    rng = random.Random(seed)
    agents: List[Agent] = []
    for idx in range(n):
        domain = DOMAINS[idx % len(DOMAINS)]
        agent_root = tmp_root / f"agent_{idx}"
        if agent_root.exists():
            shutil.rmtree(agent_root, ignore_errors=True)
        agent_root.mkdir(parents=True, exist_ok=True)
        adapter = make_dummy_adapter(domain, agent_root)
        if trojan_rate > 0.0 and rng.random() < trojan_rate:
            mark_trojan(adapter.path.parent)
        agents.append(
            Agent(
                agent_id=idx,
                domain=domain,
                adapter=adapter,
                root_dir=agent_root,
                security_policy=policy,
            )
        )
    return agents


def _teardown_agents(agents: Sequence[Agent]) -> None:
    for agent in agents:
        root = getattr(agent, "root_dir", None)
        if isinstance(root, Path):
            shutil.rmtree(root, ignore_errors=True)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    env_path = args.out.parent / "env_manifest.json"
    env_path.write_text(json.dumps(_env_manifest(), indent=2))

    topologies = [item.strip() for item in args.topos.split(",") if item.strip()]
    sizes = [int(item) for item in args.ns.split(",") if item]
    seeds = [int(item) for item in args.seeds.split(",") if item]
    trojan_rates = [float(item) for item in args.trojan_rates.split(",") if item]
    tau_trigger_grid = [float(item) for item in args.tau_trigger.split(",")]
    tau_norm_z_grid = [float(item) for item in args.tau_norm_z.split(",")]
    tau_clean_delta_grid = [float(item) for item in args.tau_clean_delta.split(",")]

    tmp_root = _reset_tmp_root(TMP_ROOT)
    try:
        with args.out.open("w") as outf:
            for topo in topologies:
                # Load calibrated C for this topology
                C_calibrated = _load_calibrated_c(topo, default_c=2.0)
                for n in sizes:
                    for seed in seeds:
                        nbrs = _build_graph(topo, n, args.p, args.ws_k, args.ba_m, seed)
                        lam2 = spectral_gap(nbrs, normalized=True)
                        # Multi-source diffusion: each domain starts in ~N/3 agents (p=1/3)
                        # Use log((1-p)⁻¹ · N) = log(3/2 · N) instead of log(N)
                        # This accounts for uninformed population remaining
                        initial_informed_fraction = 1.0 / len(DOMAINS)  # p = 1/3 for 3 domains
                        t_pred = predicted_rounds_spectral(
                            n, lam2, normalized=True, C=C_calibrated,
                            initial_informed_fraction=initial_informed_fraction
                        )
                        for trojan_rate in trojan_rates:
                            for tau_trig in tau_trigger_grid:
                                for tau_nz in tau_norm_z_grid:
                                    for tau_cd in tau_clean_delta_grid:
                                        policy = Policy(
                                            base_model="dummy/base",
                                            allowed_ranks=(1, 4, 8, 16),
                                            allowed_targets=ATTENTION_SUFFIXES,
                                            signatures_enabled=False,
                                            tau_trigger=tau_trig,
                                            tau_norm_z=tau_nz,
                                            tau_clean_delta=tau_cd,
                                        )
                                        agents = _build_agents(n, policy, trojan_rate, seed, tmp_root)

                                        round_logs: List[Dict[str, object]] = []
                                        prev_I = None

                                        def _on_round(t: int, events: List[tuple[int, int, str]]) -> None:
                                            nonlocal prev_I
                                            knowledge = {agent.agent_id: set(agent.knowledge) for agent in agents}
                                            coverage = cov_fn(knowledge)
                                            mi_val = mi_fn(knowledge)
                                            delta = (mi_val - prev_I) if prev_I is not None else 0.0
                                            prev_I = mi_val
                                            round_logs.append(
                                                {
                                                    "t": t,
                                                    "coverage": coverage,
                                                    "mutual_information": mi_val,
                                                    "mi_delta": delta,
                                                }
                                            )

                                        asyncio.run(
                                            run_gossip(
                                                agents,
                                                rounds=args.rounds,
                                                p=args.p,
                                                seed=seed,
                                                neighbours=nbrs,
                                                on_round=_on_round,
                                            )
                                        )

                                        t_obs = None
                                        for entry in round_logs:
                                            coverage = entry.get("coverage") or {}
                                            # Use 90% coverage threshold as per specification (≥90% coverage)
                                            if coverage and all(val >= 0.9 for val in coverage.values()):
                                                t_obs = entry["t"]
                                                break

                                        rejected_clean = sum(getattr(agent, "rejected_clean", 0) for agent in agents)
                                        accepted_trojan = sum(
                                            getattr(agent, "accepted_trojan", 0) for agent in agents
                                        )

                                        record = {
                                            "topology": topo,
                                            "N": n,
                                            "seed": seed,
                                            "params": {
                                                "p": args.p,
                                                "ws_k": args.ws_k,
                                                "ws_beta": args.ws_beta,
                                                "ba_m": args.ba_m,
                                            },
                                            "lambda2": lam2,
                                            "t_pred": t_pred,
                                            "t_obs": t_obs,
                                            "trojan_rate": trojan_rate,
                                            "thresholds": {
                                                "tau_trigger": tau_trig,
                                                "tau_norm_z": tau_nz,
                                                "tau_clean_delta": tau_cd,
                                            },
                                            "mi_series": [entry["mutual_information"] for entry in round_logs],
                                            "gate": {
                                                "rejected_clean_total": rejected_clean,
                                                "accepted_trojan_total": accepted_trojan,
                                            },
                                        }
                                        outf.write(json.dumps(record) + "\n")
                                        _teardown_agents(agents)
    finally:
        _cleanup_tmp_root(tmp_root)


if __name__ == "__main__":
    main()
