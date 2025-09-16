from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

from plora.agent import Agent, AdapterInfo
from plora.gate import Policy
from plora.manifest import Manifest
from plora.backdoor import mark_trojan
from swarm.swarm_v2 import run_gossip
from plora.signer import generate_keypair, sign_with_tag, ADAPTER_TAG


def _mk_adapter(tmp: Path, dom: str, payload: bytes = b"x") -> AdapterInfo:
    d = tmp / dom
    d.mkdir(parents=True, exist_ok=True)
    (d / "adapter_config.json").write_text("{}")
    (d / "adapter_model.safetensors").write_bytes(payload)
    sha = hashlib.sha256(payload).hexdigest()
    man = Manifest(
        schema_version=0,
        plasmid_id=f"m-{dom}",
        domain=dom,
        base_model="dummy/base",
        peft_format="lora",
        lora={"r": 1, "alpha": 1, "dropout": 0.0, "target_modules": []},
        artifacts={
            "filename": "adapter_model.safetensors",
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
    man.dump(d / "plora.yml")
    return AdapterInfo(d / "adapter_model.safetensors", man, len(payload))


def test_trojan_is_rejected_by_gate(tmp_path: Path):
    # Build three agents, one with a trojan adapter
    pol = Policy(
        base_model="dummy/base",
        allowed_ranks=(1, 4, 8, 16),
        allowed_targets=None,
        signatures_enabled=False,
    )

    a0 = _mk_adapter(tmp_path / "a0", "clean0")
    a1 = _mk_adapter(tmp_path / "a1", "clean1")
    trojan = _mk_adapter(tmp_path / "a2", "bad")
    # mark trojan
    mark_trojan(trojan.path.parent)

    ag0 = Agent(0, "clean0", a0, root_dir=(tmp_path / "a0"), security_policy=pol)
    ag1 = Agent(1, "clean1", a1, root_dir=(tmp_path / "a1"), security_policy=pol)
    ag2 = Agent(2, "bad", trojan, root_dir=(tmp_path / "a2"), security_policy=pol)

    agents = [ag0, ag1, ag2]

    asyncio.run(run_gossip(agents, rounds=5, p=0.25, seed=42))

    # Clean domains should diffuse, trojan domain should be absent on clean agents
    assert "clean0" in ag1.knowledge and "clean1" in ag0.knowledge
    assert "bad" not in ag0.knowledge
    assert "bad" not in ag1.knowledge

    # Gate counters reflect rejection of trojan, and no trojan acceptance
    assert ag0.accepted_trojan == 0 and ag1.accepted_trojan == 0
    assert ag0.rejected_trojan + ag1.rejected_trojan >= 1


def test_activation_gradient_anomaly_and_revocation(tmp_path: Path):
    # Build one clean adapter
    pol = Policy(
        base_model="dummy/base",
        allowed_ranks=(1, 4, 8, 16),
        allowed_targets=None,
        signatures_enabled=False,
    )
    a0 = _mk_adapter(tmp_path / "a0", "good")
    # Write activation/gradient anomaly files to trigger gate
    adir = a0.path.parent
    (adir / "activation_ref.json").write_text('{"mean": [0,0], "cov": [[1,0],[0,1]]}')
    (adir / "activation_obs.json").write_text('{"vec": [10,10]}')
    (adir / "grad_ref.json").write_text('{"median": 1.0, "mad": 0.1}')
    (adir / "grad_obs.json").write_text('{"norm": 5.0}')

    ag = Agent(0, "good", a0, root_dir=(tmp_path / "a0"), security_policy=pol)
    # Acceptance should fail due to detectors
    ok = asyncio.get_event_loop().run_until_complete(ag.accept(a0, "good"))
    assert not ok

    # Attestation replay and revocation
    # Create a trusted key and two attestations: old and new
    priv = tmp_path / "priv.pem"
    pub = tmp_path / "pub.pem"
    generate_keypair(priv, pub)
    sha = a0.manifest.artifacts.sha256
    old = {
        "signer_id": 1,
        "domain": "good",
        "sha256": sha,
        "signature_b64": sign_with_tag(priv, sha, ADAPTER_TAG),
        "ts_unix": 1,
    }
    new = {
        "signer_id": 1,
        "domain": "good",
        "sha256": sha,
        "signature_b64": sign_with_tag(priv, sha, ADAPTER_TAG),
        "ts_unix": 2,
    }
    (adir / "attestations.jsonl").write_text(
        "\n".join([__import__("json").dumps(old), __import__("json").dumps(new)]) + "\n"
    )
    # Revocation list revokes signer 1
    (adir / "revocations.json").write_text('{"1": 2}')
    pol2 = Policy(
        base_model="dummy/base",
        allowed_ranks=(1, 4, 8, 16),
        allowed_targets=None,
        signatures_enabled=True,
        trusted_public_keys=[pub],
        quorum_required=1,
    )
    from plora.gate import policy_check

    ok2, reasons = policy_check(adir, a0.manifest, pol2)
    # Revocation should invalidate the attestation
    assert not ok2 and "signature_quorum_not_met" in reasons
