

import asyncio
import hashlib
import tempfile
from pathlib import Path

from plora.agent import AdapterInfo, Agent
from plora.manifest import Manifest
from swarm.graph_engine import GraphEngine, build_topology
from swarm.gossip_node import GossipNode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dummy_adapter(domain: str, root: Path) -> AdapterInfo:
    """Create a minimal on-disk adapter artefact accepted by `Agent`.

    The layout mimics a real LoRA checkpoint so that the normal validation &
    copy code paths are exercised during the test.
    """

    dom_dir = root / domain
    dom_dir.mkdir(parents=True, exist_ok=True)

    # Fake LoRA weights (a few bytes is enough for the test)
    model_path = dom_dir / "adapter_model.safetensors"
    model_bytes = f"dummy-payload-{domain}".encode()
    model_path.write_bytes(model_bytes)

    # Minimal mandatory files
    (dom_dir / "adapter_config.json").write_text("{}")

    sha = hashlib.sha256(model_bytes).hexdigest()

    manifest = Manifest(
        schema_version=0,
        plasmid_id=f"dummy-{sha[:8]}",
        domain=domain,
        base_model="dummy/base",
        peft_format="lora",
        lora={"r": 1, "alpha": 1, "dropout": 0.0, "target_modules": []},
        artifacts={
            "filename": model_path.name,
            "sha256": sha,
            "size_bytes": len(model_bytes),
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

    manifest.dump(dom_dir / "plora.yml")

    return AdapterInfo(model_path, manifest, len(model_bytes))


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


def test_swarm_smoke_diffusion():
    """End-to-end smoke test: verify that knowledge fully diffuses."""

    N = 3  # number of agents
    max_rounds = 10

    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)

        # Build topology & nodes
        domains = [f"d{i}" for i in range(N)]
        topology = build_topology("mesh", N)

        nodes = []
        for i in range(N):
            agent_root = tmp_root / f"agent_{i}"
            adapter = _make_dummy_adapter(domains[i], agent_root)
            agent = Agent(agent_id=i, domain=domains[i], adapter=adapter, root_dir=agent_root)
            node = GossipNode(agent=agent, agent_id=i, neighbours=topology[i], mode="sim")
            nodes.append(node)

        engine = GraphEngine(
            nodes=nodes,
            topology_kind="mesh",
            domains=domains,
            max_rounds=max_rounds,
            seed=42,
            report_dir=tmp_root,
        )

        # Run the simulation
        asyncio.run(engine.run())

        # Assert every agent learned every domain
        expected_knowledge = set(domains)
        for n in nodes:
            assert n.agent.knowledge == expected_knowledge, (
                f"Agent {n.agent_id} missing domains: {expected_knowledge - n.agent.knowledge}"
            )
