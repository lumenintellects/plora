from __future__ import annotations

import torch

from plora.metrics import kl_divergence_logits, js_divergence_logits


def test_kl_js_properties():
    # Construct simple logits tensors for two distributions over 3 classes
    p = torch.tensor([[[2.0, 1.0, 0.0]]])  # shape [1,1,3]
    q = torch.tensor([[[0.0, 1.0, 2.0]]])

    kl_pq = kl_divergence_logits(p, q)
    kl_qp = kl_divergence_logits(q, p)
    js = js_divergence_logits(p, q)

    assert kl_pq.ndim == 2 and kl_qp.ndim == 2 and js.ndim == 2
    # JS is symmetric and non-negative
    js2 = js_divergence_logits(q, p)
    assert torch.allclose(js, js2, atol=1e-6)
    assert (js >= 0).all()
