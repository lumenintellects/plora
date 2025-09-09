from __future__ import annotations

from pathlib import Path

from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

from plora.loader import merge_plasmids
from plora.compat import device_dtype


def _make_adapter(base: str, out_dir: Path):
    device, dtype = device_dtype()
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=dtype, device_map={"": device}
    )
    cfg = LoraConfig(r=1, lora_alpha=1, target_modules=["c_attn"], lora_dropout=0.0)
    model = get_peft_model(model, cfg)
    model.save_pretrained(out_dir, safe_serialization=True)


def test_merge(tmp_path: Path):
    base = "sshleifer/tiny-gpt2"
    dir1 = tmp_path / "a1"
    dir1.mkdir()
    dir2 = tmp_path / "a2"
    dir2.mkdir()
    _make_adapter(base, dir1)
    _make_adapter(base, dir2)

    merged = merge_plasmids(base, [dir1, dir2])
    import torch

    input_ids = torch.tensor([[merged.config.bos_token_id]], device=merged.device)
    out = merged.generate(
        max_length=2, do_sample=False, num_beams=1, input_ids=input_ids
    )
    assert out.shape[0] == 1
