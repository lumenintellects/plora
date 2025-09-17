import hashlib
import json
from pathlib import Path

import torch
import pytest
from transformers import AutoModelForCausalLM

from plora.loader import random_lora


@pytest.mark.slow
@pytest.mark.parametrize("call_variant", ["model", "name"])
def test_random_lora_does_not_mutate_base(tmp_path: Path, call_variant: str):
    """Ensure random_lora never mutates the caller's model parameters or attaches PEFT config.

    We verify by hashing all parameter tensors before and after invocation.
    Two variants are tested:
      * Passing a model instance
      * Passing a model name (string)
    """
    base_model_name = "sshleifer/tiny-gpt2"  # very small model for tests

    # Load base model once
    model = AutoModelForCausalLM.from_pretrained(base_model_name)

    # Collect parameter hashes (sha256 over raw bytes) for strong equality check
    def param_hashes(m):
        h = {}
        for k, v in m.state_dict().items():
            # Ensure contiguous CPU bytes for stable hashing
            t = v.detach().cpu().contiguous()
            h[k] = hashlib.sha256(t.numpy().tobytes()).hexdigest()
        return h

    before = param_hashes(model)
    assert not hasattr(model, "peft_config"), "Base model unexpectedly has peft_config before test"

    out_dir = tmp_path / "rand_adapter"
    if call_variant == "model":
        random_lora(model, out_dir, r=2)
    else:
        random_lora(base_model_name, out_dir, r=2)

    # Adapter directory should contain adapter config & weights
    assert (out_dir / "adapter_config.json").exists(), "Adapter config missing"
    assert any(p.name.startswith("adapter_model") for p in out_dir.iterdir()), "Adapter weights missing"

    after = param_hashes(model)
    # Compare param-by-param
    assert before.keys() == after.keys(), "Parameter key sets changed after random_lora call"
    diffs = [k for k in before if before[k] != after[k]]
    assert not diffs, f"Parameters mutated for keys: {diffs}"

    # Model still must not have a peft_config attribute added by side effect
    assert not hasattr(model, "peft_config"), "random_lora introduced peft_config on original model"

    # Spot check: load a second adapter call should not mutate either
    second_dir = tmp_path / "rand_adapter2"
    random_lora(base_model_name, second_dir, r=4)
    after2 = param_hashes(model)
    assert all(after[k] == after2[k] for k in after), "Model mutated after second independent random_lora"

