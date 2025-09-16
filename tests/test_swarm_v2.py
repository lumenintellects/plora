from __future__ import annotations

import asyncio
from pathlib import Path

from plora.agent import Agent, AdapterInfo
from plora.manifest import Manifest
from swarm.swarm_v2 import run_gossip
import json


def _mk_dummy(tmp: Path, dom: str) -> AdapterInfo:
    d = tmp / dom
    d.mkdir(parents=True, exist_ok=True)
    (d / "adapter_config.json").write_text("{}")
    (d / "adapter_model.safetensors").write_bytes(b"x")
    man = Manifest(
        schema_version=0,
        plasmid_id=f"m-{dom}",
        domain=dom,
        base_model="dummy/base",
        peft_format="lora",
        lora={"r": 1, "alpha": 1, "dropout": 0.0, "target_modules": []},
        artifacts={
            "filename": "adapter_model.safetensors",
            "sha256": "2d711642b726b04401627ca9fbac32f5da7d3b731f6d1e6f0b6d2d0b2fcd7cde",
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
        signer={"algo": "none", "pubkey_fingerprint": "none", "signature_b64": ""},
        compatibility={"peft_min": "0", "transformers": "0"},
    )
    man.dump(d / "plora.yml")
    return AdapterInfo(d / "adapter_model.safetensors", man, 1)


def test_push_pull_diffusion(tmp_path: Path):
    n = 6
    agents = []
    domains = [f"d{i}" for i in range(3)]
    for i in range(n):
        dom = domains[i % len(domains)]
        ad = _mk_dummy(tmp_path / f"a{i}", dom)
        ag = Agent(i, dom, ad, root_dir=(tmp_path / f"a{i}"))
        agents.append(ag)

    asyncio.run(run_gossip(agents, rounds=5, p=0.25, seed=42))

    expected = set(domains)
    for ag in agents:
        assert ag.knowledge == expected


def test_v2_consensus_commits(tmp_path: Path):
    # Use sim_v2_entry to produce a report with consensus enabled
    from swarm import sim_v2_entry as entry
    import argparse

    args = argparse.Namespace(
        agents=4,
        rounds=2,
        graph_p=0.5,
        graph="er",
        ws_k=4,
        ws_beta=0.2,
        ba_m=2,
        seed=11,
        max_inflight=0,
        report_dir=tmp_path,
        security="on",
        allowed_targets="attention",
        allowed_targets_file=None,
        allowed_ranks="1",
        trojan_rate=0.0,
        signatures="off",
        trusted_pubkeys="",
        policy_file=None,
        tau_trigger=None,
        tau_norm_z=None,
        tau_clean_delta=None,
        consensus="on",
        quorum=2,
    )
    import asyncio

    asyncio.run(entry._main_async(args))
    reports = list(tmp_path.glob("swarm_v2_report_*.json"))
    assert reports
    rep = json.loads(reports[0].read_text())
    assert "consensus" in rep.get("final", {})
