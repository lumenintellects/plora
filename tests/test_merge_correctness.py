from __future__ import annotations

from pathlib import Path

from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

from plora.loader import merge_plasmids
from plora.compat import device_dtype
from plora.metrics import token_nlls
from transformers import AutoTokenizer


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


def test_line_search_and_module_caps(tmp_path: Path):
    base = "sshleifer/tiny-gpt2"
    dir1 = tmp_path / "a1"
    dir1.mkdir()
    dir2 = tmp_path / "a2"
    dir2.mkdir()
    _make_adapter(base, dir1)
    _make_adapter(base, dir2)

    # Tiny synthetic dataset for objective
    dataset = [("What is 1+1?", "2"), ("Say hello", "hello")]
    tok = AutoTokenizer.from_pretrained(base)
    model_pre = AutoModelForCausalLM.from_pretrained(base)
    nll_before = sum(token_nlls(model_pre, tok, dataset)) / len(dataset)

    # Use module caps to ensure scaling does not explode per-module deltas
    caps = {"c_attn": 1e-6}
    merged = merge_plasmids(
        base,
        [dir1, dir2],
        strategy="weighted_sum",
        module_caps=caps,
        ls_dataset=dataset,
    )
    nll_after = sum(token_nlls(merged, tok, dataset)) / len(dataset)
    # Non-degradation under line-search on tiny dataset (allow tiny tolerance)
    assert nll_after <= nll_before + 1e-3
