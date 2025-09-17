from __future__ import annotations

"""Sweep runner for Swarm Sim v2.

Runs across topologies (ER/WS/BA), sizes, seeds, and grids for trojan_rate and
gate thresholds. Logs λ2, predicted vs observed rounds-to-diffuse, MI
trajectories, and gate FN/FP. Writes consolidated JSONL for plotting.
"""

import argparse
import json
import random
from pathlib import Path
import hashlib
from typing import Dict, List, Sequence

from plora.agent import Agent, make_dummy_adapter
from plora.gate import Policy
from plora.backdoor import mark_trojan
from swarm.swarm_v2 import run_gossip
from swarm.graph_v2 import (
    erdos_renyi_graph,
    watts_strogatz_graph,
    barabasi_albert_graph,
)
from swarm.metrics import coverage as cov_fn, mutual_information as mi_fn, spectral_gap
from swarm.theory import predicted_rounds_spectral
from plora.config import get as cfg


_DOMAINS_DEFAULT = ["arithmetic", "legal", "medical"]


def _mk_dummy_adapter(domain: str, root: Path):
    return make_dummy_adapter(domain, root)


def _build_graph(topology: str, n: int, p: float, k: int, m: int, seed: int):
    if topology == "er":
        return erdos_renyi_graph(n, p, seed)
    if topology == "ws":
        return watts_strogatz_graph(n, k, p, seed)
    if topology == "ba":
        return barabasi_albert_graph(n, m, seed)
    raise ValueError("unknown topology")


def _env_manifest() -> dict:
    info = {}
    try:
        import platform

        info["platform"] = {
            "python": platform.python_version(),
            "system": platform.system(),
            "release": platform.release(),
        }
    except Exception:
        pass
    try:
        import torch

        info["torch"] = {
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "mps_available": bool(
                getattr(torch.backends, "mps", None)
                and torch.backends.mps.is_available()
            ),
        }
    except Exception:
        pass
    for pkg in ("transformers", "peft", "numpy"):
        try:
            mod = __import__(pkg)
            info[pkg] = getattr(mod, "__version__", "")
        except Exception:
            pass
    return info


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topos", type=str, default="er,ws,ba")
    ap.add_argument("--ns", type=str, default="20,40")
    ap.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(x) for x in cfg("value_add.seeds", [41, 42])),
    )
    ap.add_argument("--p", type=float, default=cfg("graph.p", 0.25))
    ap.add_argument("--ws_k", type=int, default=cfg("graph.ws_k", 4))
    ap.add_argument("--ws_beta", type=float, default=cfg("graph.ws_beta", 0.2))
    ap.add_argument("--ba_m", type=int, default=cfg("graph.ba_m", 2))
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--trojan_rates", type=str, default="0.0,0.3")
    ap.add_argument(
        "--tau_trigger", type=str, default=str(cfg("gate.tau_trigger", 0.2))
    )
    ap.add_argument("--tau_norm_z", type=str, default=str(cfg("gate.tau_norm_z", 3.0)))
    ap.add_argument(
        "--tau_clean_delta", type=str, default=str(cfg("gate.tau_clean_delta", -0.05))
    )
    ap.add_argument("--out", type=Path, default=Path("results/thesis_sweep.jsonl"))
    ns = ap.parse_args(argv)

    topologies = [t.strip() for t in ns.topos.split(",") if t.strip()]
    sizes = [int(x) for x in ns.ns.split(",") if x]
    seeds = [int(x) for x in ns.seeds.split(",") if x]
    trojans = [float(x) for x in ns.trojan_rates.split(",") if x]
    tau_trigger_grid = [float(x) for x in ns.tau_trigger.split(",")]
    tau_norm_z_grid = [float(x) for x in ns.tau_norm_z.split(",")]
    tau_clean_delta_grid = [float(x) for x in ns.tau_clean_delta.split(",")]

    ns.out.parent.mkdir(parents=True, exist_ok=True)
    outf = ns.out.open("w")

    # Write environment manifest once
    (ns.out.parent / "env_manifest.json").write_text(
        json.dumps(_env_manifest(), indent=2)
    )

    for topo in topologies:
        for n in sizes:
            for seed in seeds:
                # Build graph & spectral gap
                if topo == "er":
                    nbrs = _build_graph(topo, n, ns.p, ns.ws_k, ns.ba_m, seed)
                elif topo == "ws":
                    nbrs = _build_graph(topo, n, ns.ws_beta, ns.ws_k, ns.ba_m, seed)
                else:
                    nbrs = _build_graph(topo, n, ns.p, ns.ws_k, ns.ba_m, seed)
                lam2 = spectral_gap(nbrs)
                t_pred = predicted_rounds_spectral(n, lam2)
                for tr in trojans:
                    for tau_trig in tau_trigger_grid:
                        for tau_nz in tau_norm_z_grid:
                            for tau_cd in tau_clean_delta_grid:
                                rng = random.Random(seed)
                                # Build agents & policy per thresholds
                                agents: List[Agent] = []
                                policy = Policy(
                                    base_model="dummy/base",
                                    allowed_ranks=(1, 4, 8, 16),
                                    allowed_targets=None,
                                    signatures_enabled=False,
                                    tau_trigger=tau_trig,
                                    tau_norm_z=tau_nz,
                                    tau_clean_delta=tau_cd,
                                )
                                for i in range(n):
                                    dom = _DOMAINS_DEFAULT[i % len(_DOMAINS_DEFAULT)]
                                    root = (
                                        Path(".sweep_tmp")
                                        / f"agent_{topo}_{n}_{seed}_{i}"
                                    )
                                    ad = _mk_dummy_adapter(dom, root)
                                    if tr > 0.0 and rng.random() < tr:
                                        mark_trojan(ad.path.parent)
                                    ag = Agent(
                                        i,
                                        dom,
                                        ad,
                                        root_dir=root,
                                        security_policy=policy,
                                    )
                                    agents.append(ag)

                                # Logs
                                round_logs: List[Dict] = []
                                prev_I = None

                                def _on_round(
                                    t: int, accepted_events: List[tuple[int, int, str]]
                                ):
                                    nonlocal prev_I
                                    know = {
                                        ag.agent_id: set(ag.knowledge) for ag in agents
                                    }
                                    cov = cov_fn(know)
                                    I_t = mi_fn(know)
                                    mi_delta = (
                                        (I_t - prev_I) if (prev_I is not None) else 0.0
                                    )
                                    prev_I = I_t
                                    round_logs.append(
                                        {
                                            "t": t,
                                            "coverage": cov,
                                            "mutual_information": I_t,
                                            "mi_delta": mi_delta,
                                        }
                                    )

                                # Run gossip
                                import asyncio

                                asyncio.run(
                                    run_gossip(
                                        agents,
                                        rounds=ns.rounds,
                                        p=ns.p if topo == "er" else 0.25,
                                        seed=seed,
                                        neighbours=nbrs,
                                        on_round=_on_round,
                                    )
                                )

                                # Observed t_all
                                t_obs = None
                                from swarm.metrics import coverage

                                for t, _ in enumerate(round_logs):
                                    know = {
                                        ag.agent_id: set(ag.knowledge) for ag in agents
                                    }
                                    cov = coverage(know)
                                    if all(pv == 1.0 for pv in cov.values()):
                                        t_obs = t
                                        break

                                # Gate stats
                                rejected_clean_total = sum(
                                    getattr(ag, "rejected_clean", 0) for ag in agents
                                )
                                accepted_trojan_total = sum(
                                    getattr(ag, "accepted_trojan", 0) for ag in agents
                                )

                                rec = {
                                    "topology": topo,
                                    "N": n,
                                    "seed": seed,
                                    "params": {
                                        "p": ns.p,
                                        "ws_k": ns.ws_k,
                                        "ws_beta": ns.ws_beta,
                                        "ba_m": ns.ba_m,
                                    },
                                    "lambda2": lam2,
                                    "t_pred": t_pred,
                                    "t_obs": t_obs,
                                    "trojan_rate": tr,
                                    "thresholds": {
                                        "tau_trigger": tau_trig,
                                        "tau_norm_z": tau_nz,
                                        "tau_clean_delta": tau_cd,
                                    },
                                    "mi_series": [
                                        r["mutual_information"] for r in round_logs
                                    ],
                                    "gate": {
                                        "rejected_clean_total": rejected_clean_total,
                                        "accepted_trojan_total": accepted_trojan_total,
                                    },
                                }
                                outf.write(json.dumps(rec) + "\n")
    outf.close()


if __name__ == "__main__":
    main()
