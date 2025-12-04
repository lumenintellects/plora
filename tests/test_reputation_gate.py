from __future__ import annotations

import hashlib
import json
from pathlib import Path

from plora.agent import Agent, AdapterInfo
from plora.manifest import Manifest
from plora.gate import Policy, policy_check


def _mk_adapter(tmp: Path, dom: str) -> AdapterInfo:
    d = tmp / dom
    d.mkdir()
    payload = b"x"
    (d / "adapter_model.safetensors").write_bytes(payload)
    sha = hashlib.sha256(payload).hexdigest()
    man = Manifest(
        schema_version=0,
        plasmid_id=f"p-{sha[:8]}",
        domain=dom,
        base_model="dummy/base",
        peft_format="lora",
        lora={"r": 4, "alpha": 8, "dropout": 0.0, "target_modules": ["q_proj"]},
        artifacts={
            "filename": "adapter_model.safetensors",
            "sha256": sha,
            "size_bytes": 1,
        },
        train_meta={
            "seed": 0,
            "epochs": 0,
            "dataset_id": "d",
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
        signer={"algo": "none", "pubkey_fingerprint": "x", "signature_b64": ""},
        compatibility={"peft_min": "0", "transformers": "0"},
    )
    man.dump(d / "plora.yml")
    return AdapterInfo(d / "adapter_model.safetensors", man, len(payload))


def test_reputation_gating(tmp_path: Path):
    # source agent has low reputation
    src = Agent(1, "d", _mk_adapter(tmp_path, "d"), root_dir=tmp_path / "src")
    dst = Agent(
        2,
        "e",
        _mk_adapter(tmp_path, "e"),
        root_dir=tmp_path / "dst",
        security_policy=Policy(
            base_model="dummy/base", allowed_ranks=(4, 8, 16), min_reputation=0.8
        ),
    )
    # set reputation and source id on adapter
    dst.peer_reputation[src.agent_id] = 0.5
    ad = _mk_adapter(tmp_path, "d2")
    setattr(ad, "source_agent_id", src.agent_id)
    ok = dst._copy_lock  # force lock creation
    accepted = dst.__class__.accept
    # call accept
    import asyncio

    asyncio.get_event_loop().run_until_complete(dst.accept(ad, "d2"))
    # Ensure rejection due to low reputation
    assert dst.rejection_reasons.get("reputation_low", 0) >= 1


def test_global_reputation_json_fallback():
    """Test that policy_check falls back to global assets/reputation.json."""
    # The global assets/reputation.json contains: {"demo-signer": 0.8, "lowrep-signer": 0.2}
    # Create a manifest with a signer fingerprint matching an entry in the global file
    from plora.gate import policy_check, Policy
    from plora.manifest import Manifest
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        adapter_dir = Path(tmp) / "adapter"
        adapter_dir.mkdir()
        payload = b"x"
        (adapter_dir / "adapter_model.safetensors").write_bytes(payload)
        sha = hashlib.sha256(payload).hexdigest()

        # Test with demo-signer (0.8 reputation) - should pass with min_reputation=0.5
        man_high_rep = Manifest(
            schema_version=0,
            plasmid_id="p-test1",
            domain="test",
            base_model="dummy/base",
            peft_format="lora",
            lora={"r": 4, "alpha": 8, "dropout": 0.0, "target_modules": ["q_proj"]},
            artifacts={"filename": "adapter_model.safetensors", "sha256": sha, "size_bytes": 1},
            train_meta={"seed": 0, "epochs": 0, "dataset_id": "d", "sample_count": 0, "timestamp_unix": 0},
            metrics={"val_ppl_before": 0.0, "val_ppl_after": 0.0, "delta_ppl": 0.0, "val_em": None, "val_chrf": None},
            safety={"licence": "CC0", "poisoning_score": 0.0},
            signer={"algo": "none", "pubkey_fingerprint": "demo-signer", "signature_b64": ""},
            compatibility={"peft_min": "0", "transformers": "0"},
        )

        policy_high_threshold = Policy(base_model="dummy/base", allowed_ranks=(4,), min_reputation=0.5)
        ok, reasons = policy_check(adapter_dir, man_high_rep, policy_high_threshold)
        # demo-signer has 0.8 reputation, threshold is 0.5, should pass
        assert "reputation_low" not in reasons, f"Expected demo-signer (0.8) to pass 0.5 threshold, got: {reasons}"

        # Test with lowrep-signer (0.2 reputation) - should fail with min_reputation=0.5
        man_low_rep = Manifest(
            schema_version=0,
            plasmid_id="p-test2",
            domain="test",
            base_model="dummy/base",
            peft_format="lora",
            lora={"r": 4, "alpha": 8, "dropout": 0.0, "target_modules": ["q_proj"]},
            artifacts={"filename": "adapter_model.safetensors", "sha256": sha, "size_bytes": 1},
            train_meta={"seed": 0, "epochs": 0, "dataset_id": "d", "sample_count": 0, "timestamp_unix": 0},
            metrics={"val_ppl_before": 0.0, "val_ppl_after": 0.0, "delta_ppl": 0.0, "val_em": None, "val_chrf": None},
            safety={"licence": "CC0", "poisoning_score": 0.0},
            signer={"algo": "none", "pubkey_fingerprint": "lowrep-signer", "signature_b64": ""},
            compatibility={"peft_min": "0", "transformers": "0"},
        )

        ok, reasons = policy_check(adapter_dir, man_low_rep, policy_high_threshold)
        # lowrep-signer has 0.2 reputation, threshold is 0.5, should fail
        assert "reputation_low" in reasons, f"Expected lowrep-signer (0.2) to fail 0.5 threshold, got: {reasons}"
