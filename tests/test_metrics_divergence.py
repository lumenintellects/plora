from __future__ import annotations

import pytest

from plora.metrics import dataset_kl_js
from transformers import AutoModelForCausalLM, AutoTokenizer
from plora.compat import device_dtype


@pytest.mark.slow
def test_kl_js_zero_for_identical_models():
    base = "sshleifer/tiny-gpt2"
    device, dtype = device_dtype()
    tok = AutoTokenizer.from_pretrained(base)
    tok.pad_token = tok.eos_token
    m1 = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=dtype, device_map={"": device}
    )
    m2 = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=dtype, device_map={"": device}
    )

    ds = [("1+1?", "2"), ("Capital of France?", "Paris")]
    res = dataset_kl_js(m1, m2, tok, ds)
    assert res["kl_pq"] < 1e-8 and res["kl_qp"] < 1e-8 and res["js"] < 1e-8
