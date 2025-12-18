from __future__ import annotations

import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

from plora.dataset_loader import get_dataset
from plora.metrics import perplexity
from plora.compat import device_dtype


@pytest.mark.slow
def test_perplexity_runs():
    device, dtype = device_dtype()
    base_model = "sshleifer/tiny-gpt2"
    model = AutoModelForCausalLM.from_pretrained(
        base_model, dtype=dtype, device_map={"": device}
    )
    tok = AutoTokenizer.from_pretrained(base_model)

    data = get_dataset("legal")
    ppl = perplexity(model, tok, data[:1])  # just 1 sample for speed
    assert isinstance(ppl, float) and ppl > 0
