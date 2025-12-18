from __future__ import annotations

import hashlib
import json
from pathlib import Path

from plora.gate import Policy, policy_check
from plora.manifest import Manifest
from plora.signer import generate_keypair, sign_sha256_hex


def _mk_manifest(domain: str, sha: str) -> Manifest:
    return Manifest(
        schema_version=0,
        plasmid_id=f"plasmid-{sha[:8]}",
        domain=domain,
        base_model="dummy/base",
        peft_format="lora",
        lora={"r": 4, "alpha": 8, "dropout": 0.0, "target_modules": ["q_proj"]},
        artifacts={
            "filename": "adapter_model.safetensors",
            "sha256": sha,
            "size_bytes": 16,
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
        signer={
            "algo": "RSA-PSS-SHA256",
            "pubkey_fingerprint": "x",
            "signature_b64": "",
        },
        compatibility={"peft_min": "0", "transformers": "0"},
    )


def test_quorum_signatures(tmp_path: Path):
    # setup dummy adapter dir
    adir = tmp_path / "adapter"
    adir.mkdir()
    payload = b"abc123"
    (adir / "adapter_model.safetensors").write_bytes(payload)
    sha = hashlib.sha256(payload).hexdigest()
    man = _mk_manifest("d", sha)

    # generate three keys and attestations
    pubs = []
    privs = []
    for i in range(3):
        priv = tmp_path / f"priv{i}.pem"
        pub = tmp_path / f"pub{i}.pem"
        generate_keypair(priv, pub)
        pubs.append(pub)
        privs.append(priv)

    # sign manifest signer with key0
    man.signer.signature_b64 = sign_sha256_hex(privs[0], sha)

    # add two more attestations in file
    att = adir / "attestations.jsonl"
    with att.open("w") as f:
        for i in [1, 2]:
            sig = sign_sha256_hex(privs[i], sha)
            f.write(json.dumps({"signature_b64": sig}) + "\n")

    pol = Policy(
        base_model="dummy/base",
        allowed_ranks=(4, 8, 16),
        allowed_targets=None,
        signatures_enabled=True,
        trusted_public_keys=pubs,
        quorum_required=2,
    )

    ok, reasons = policy_check(adir, man, pol)
    assert ok
