from __future__ import annotations

import os
import statistics
import time
from pathlib import Path

from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

from plora.loader import inject
from plora.compat import device_dtype

LAT_MS = int(os.getenv("PLORA_LATENCY_BUDGET_MS", "250"))


def _create_dummy_adapter(tmp_path: Path):
    device, dtype = device_dtype()
    base_model = "sshleifer/tiny-gpt2"
    model = AutoModelForCausalLM.from_pretrained(
        base_model, dtype=dtype, device_map={"": device}
    )
    cfg = LoraConfig(r=1, lora_alpha=1, target_modules=["c_attn"], lora_dropout=0.0)
    model = get_peft_model(model, cfg)
    model.save_pretrained(tmp_path, safe_serialization=True)
    return model, tmp_path


def test_inject_latency(tmp_path: Path):
    model, adapter_dir = _create_dummy_adapter(tmp_path / "adapter")

    times = []
    # warm-up + 20 iterations
    for i in range(21):
        t0 = time.perf_counter()
        with inject(model, adapter_dir):
            pass
        times.append((time.perf_counter() - t0) * 1e3)
    median_ms = statistics.median(times[1:])  # exclude warm-up
    assert median_ms <= LAT_MS, f"Median {median_ms:.1f} ms exceeds budget {LAT_MS} ms"
