from __future__ import annotations

import hashlib
from pathlib import Path

from plora.gate import Policy, policy_check
from plora.manifest import Manifest


def _mk_adapter(tmp_path: Path, domain: str, size: int = 1) -> Path:
    d = tmp_path / domain
    d.mkdir(parents=True, exist_ok=True)
    (d / "adapter_config.json").write_text("{}")
    (d / "adapter_model.safetensors").write_bytes(b"0" * size)
    return d


def _mk_manifest(domain: str, sha: str, r: int, targets):
    return Manifest(
        schema_version=0,
        plasmid_id=f"m-{domain}",
        domain=domain,
        base_model="dummy/base",
        peft_format="lora",
        lora={"r": r, "alpha": r * 2, "dropout": 0.0, "target_modules": list(targets)},
        artifacts={
            "filename": "adapter_model.safetensors",
            "sha256": sha,
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


def test_policy_checks_rank_and_targets(tmp_path: Path):
    dom = "d"
    d = _mk_adapter(tmp_path, dom)
    sha = hashlib.sha256((d / "adapter_model.safetensors").read_bytes()).hexdigest()
    man = _mk_manifest(dom, sha, r=4, targets=["q_proj", "k_proj"])

    pol = Policy(
        base_model="dummy/base",
        allowed_ranks=(4, 8, 16),
        allowed_targets=["q_proj", "k_proj", "v_proj"],
    )
    ok, reasons = policy_check(d, man, pol)
    assert ok

    # Bad rank
    man_bad_rank = _mk_manifest(dom, sha, r=3, targets=["q_proj"])
    ok2, reasons2 = policy_check(d, man_bad_rank, pol)
    assert not ok2 and "rank_not_allowed" in reasons2

    # Bad target
    man_bad_t = _mk_manifest(dom, sha, r=4, targets=["weird_proj"])
    ok3, reasons3 = policy_check(d, man_bad_t, pol)
    assert not ok3 and "targets_not_allowed" in reasons3
