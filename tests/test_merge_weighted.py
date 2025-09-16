from __future__ import annotations

from pathlib import Path

import json
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

from plora.loader import merge_plasmids
from plora.compat import device_dtype


def _make_adapter(base: str, out_dir: Path):
    device, dtype = device_dtype()
    model = AutoModelForCausalLM.from_pretrained(
        base, dtype=dtype, device_map={"": device}
    )
    cfg = LoraConfig(r=1, lora_alpha=1, target_modules=["c_attn"], lora_dropout=0.0)
    model = get_peft_model(model, cfg)
    model.save_pretrained(out_dir, safe_serialization=True)


def _param_l2(model_a, model_b) -> float:
    s = 0.0
    with torch.no_grad():
        sda = model_a.state_dict()
        sdb = model_b.state_dict()
        for k, va in sda.items():
            vb = sdb.get(k)
            if vb is None or vb.shape != va.shape:
                continue
            s += float(((va - vb) ** 2).sum().item())
    return s


def test_weighted_sum_zero_is_base(tmp_path: Path):
    base = "sshleifer/tiny-gpt2"
    a1 = tmp_path / "a1"
    a2 = tmp_path / "a2"
    a1.mkdir()
    a2.mkdir()
    _make_adapter(base, a1)
    _make_adapter(base, a2)

    device, dtype = device_dtype()
    base_model = AutoModelForCausalLM.from_pretrained(
        base, dtype=dtype, device_map={"": device}
    )
    merged = merge_plasmids(base, [a1, a2], weights=[0.0, 0.0], strategy="weighted_sum")
    dist = _param_l2(base_model, merged)
    assert dist < 1e-8


def test_weighted_sum_matches_single_adapter(tmp_path: Path):
    base = "sshleifer/tiny-gpt2"
    a1 = tmp_path / "a1"
    a2 = tmp_path / "a2"
    a1.mkdir()
    a2.mkdir()
    _make_adapter(base, a1)
    _make_adapter(base, a2)

    one = merge_plasmids(base, [a1], strategy="sequential")
    wsum = merge_plasmids(base, [a1, a2], weights=[1.0, 0.0], strategy="weighted_sum")
    dist = _param_l2(one, wsum)
    assert dist < 1e-8


def test_fisher_weighting_prefers_higher_signal(tmp_path: Path):
    base = "sshleifer/tiny-gpt2"
    a1 = tmp_path / "a1"
    a2 = tmp_path / "a2"
    a1.mkdir()
    a2.mkdir()
    _make_adapter(base, a1)
    _make_adapter(base, a2)

    # Write fisher signals: a1 high, a2 zero
    (a1 / "fisher_diag.json").write_text(json.dumps({"sum": 1e9}))
    (a2 / "fisher_diag.json").write_text(json.dumps({"sum": 0.0}))

    fisher_model = merge_plasmids(
        base, [a1, a2], strategy="weighted_sum", fisher_weighted=True
    )
    one = merge_plasmids(base, [a1], strategy="sequential")
    dist = _param_l2(one, fisher_model)
    assert dist < 1e-8
