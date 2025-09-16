from __future__ import annotations

from pathlib import Path
import json
import torch
from safetensors.torch import save_file

from plora.weights import tensor_norm_anomaly_z


def test_tensor_norm_anomaly_z(tmp_path: Path):
    adir = tmp_path / "a"
    adir.mkdir()
    # create fake lora A/B tensors with one outlier
    tensors = {
        "layers.0.lora_A": torch.ones(4, 2),
        "layers.0.lora_B": torch.ones(2, 4),
        "layers.1.lora_A": torch.ones(4, 2),
        "layers.1.lora_B": torch.ones(2, 4) * 100.0,  # outlier
    }
    save_file(tensors, str(adir / "adapter_model.safetensors"))
    zmax, count = tensor_norm_anomaly_z(adir)
    assert count == 4
    assert zmax > 3.0
