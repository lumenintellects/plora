from __future__ import annotations

from pathlib import Path

from plora.agent import Agent, AdapterInfo
from plora.manifest import Manifest


def _dummy_manifest(domain: str, em=None, chrf=None, delta=0.0) -> Manifest:
    return Manifest(
        schema_version=0,
        plasmid_id=f"m-{domain}",
        domain=domain,
        base_model="dummy/base",
        peft_format="lora",
        lora={"r": 1, "alpha": 1, "dropout": 0.0, "target_modules": []},
        artifacts={
            "filename": "adapter_model.safetensors",
            "sha256": "0" * 64,
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
            "delta_ppl": float(delta),
            "val_em": None if em is None else float(em),
            "val_chrf": None if chrf is None else float(chrf),
        },
        safety={"licence": "CC0", "poisoning_score": 0.0},
        signer={"algo": "none", "pubkey_fingerprint": "none", "signature_b64": ""},
        compatibility={"peft_min": "0", "transformers": "0"},
    )


def _ad(domain: str, em=None, chrf=None, delta=0.0):
    man = _dummy_manifest(domain, em=em, chrf=chrf, delta=delta)
    return AdapterInfo(Path("/dev/null"), man, 0)


def test_best_offer_prefers_em_then_chrf_then_delta():
    # Agent A has three adapters; B lacks all of them
    a = Agent(0, "d0", _ad("d0", em=0.5))
    # add received adapters to A
    a.received["d1"] = _ad("d1", em=None, chrf=0.3)
    a.received["d2"] = _ad("d2", em=None, chrf=None, delta=-0.2)

    b = Agent(1, "x", _ad("x", em=0.1))
    b.knowledge = set()

    dom, _ = a.best_offer_for(b)
    assert dom == "d0"  # highest EM wins

    # If EM missing, CHRF is used
    a.adapter = _ad("d0", em=None, chrf=0.25)
    dom, _ = a.best_offer_for(b)
    assert dom == "d1"  # 0.3 chrf vs 0.25 chrf

    # If both missing, prefer more negative delta_ppl (i.e., higher -delta)
    a.adapter = _ad("d0", em=None, chrf=None, delta=-0.1)
    a.received["d1"] = _ad("d1", em=None, chrf=None, delta=-0.3)
    a.received["d2"] = _ad("d2", em=None, chrf=None, delta=-0.2)
    dom, _ = a.best_offer_for(b)
    assert dom == "d1"  # -0.3 best
