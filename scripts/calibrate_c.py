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

from plora.agent import Agent, AdapterInfo
from plora.manifest import Manifest
from swarm.swarm_v2 import run_gossip
from swarm.graph_v2 import (
    erdos_renyi_graph,
    watts_strogatz_graph,
    barabasi_albert_graph,
)
from swarm.metrics import spectral_gap


_DOMAINS_DEFAULT = ["arithmetic", "legal", "medical"]


def _mk_dummy_adapter(domain: str, root: Path) -> AdapterInfo:
    root.mkdir(parents=True, exist_ok=True)
    model_path = root / "adapter_model.safetensors"
    payload = f"dummy-{domain}".encode()
    model_path.write_bytes(payload)
    (root / "adapter_config.json").write_text("{}")
    sha_hex = hashlib.sha256(payload).hexdigest()
    man = Manifest(
        schema_version=0,
        plasmid_id=f"dummy-{domain}",
        domain=domain,
        base_model="dummy/base",
        peft_format="lora",
        lora={"r": 1, "alpha": 1, "dropout": 0.0, "target_modules": []},
        artifacts={
            "filename": model_path.name,
            "sha256": sha_hex,
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
    man.dump(root / "plora.yml")
    return AdapterInfo(model_path, man, len(payload))


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
    ap.add_argument("--p", type=float, default=0.25, help="ER/WS probability")
    ap.add_argument("--k", type=int, default=4, help="WS k")
    ap.add_argument("--m", type=int, default=2, help="BA m")
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
            ad = _mk_dummy_adapter(dom, root)
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
                p=ns.p if ns.topology == "er" else 0.25,
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
