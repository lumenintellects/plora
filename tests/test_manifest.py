from __future__ import annotations

import textwrap
from pathlib import Path

from plora.manifest import Manifest


def test_manifest_roundtrip(tmp_path: Path):
    yaml_txt = textwrap.dedent(
        """
        schema_version: 0
        plasmid_id: "dummy-001"
        domain: "legal"
        base_model: "sshleifer/tiny-gpt2"
        peft_format: "lora"
        lora:
          r: 4
          alpha: 8
          dropout: 0.1
          target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
        artifacts:
          filename: "adapter_model.safetensors"
          sha256: "0000000000000000000000000000000000000000000000000000000000000000"
          size_bytes: 123
        train_meta:
          seed: 42
          epochs: 1
          dataset_id: "fixture"
          sample_count: 64
          timestamp_unix: 1734632400
        metrics:
          val_ppl_before: 14.2
          val_ppl_after: 12.7
          delta_ppl: -1.5
          val_em: 0.1
          val_chrf: 0.22
        safety:
          licence: "research"
          poisoning_score: 0.0
        signer:
          algo: "RSA-PSS-SHA256"
          pubkey_fingerprint: "SHA256:ab12"
          signature_b64: "abcd=="
        compatibility:
          peft_min: "0.12.0"
          transformers: ">=4.42"
        """
    ).replace("\n        ", "\n")

    yaml_path = tmp_path / "plora.yml"
    yaml_path.write_text(yaml_txt)

    man = Manifest.load(yaml_path)
    # Round-trip dump and re-load
    new_path = tmp_path / "out.yml"
    man.dump(new_path)
    man2 = Manifest.load(new_path)
    assert man2.plasmid_id == "dummy-001"
    assert man2.metrics.delta_ppl == -1.5
