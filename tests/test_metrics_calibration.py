from __future__ import annotations

import torch
from plora.metrics import ece_from_logits


def test_ece_zero_for_perfect_predictions():
    B, T, V = 2, 4, 5
    logits = torch.full((B, T, V), -100.0)
    # make predicted class 2 with very high confidence
    logits[..., 2] = 100.0
    labels = torch.full((B, T), 2)
    e = ece_from_logits(logits, labels)
    assert e < 1e-8


def test_ece_nonzero_for_miscalibrated():
    B, T, V = 2, 4, 5
    logits = torch.zeros((B, T, V))
    # force high confidence on class 0 but labels mostly class 1
    logits[..., 0] = 5.0
    labels = torch.ones((B, T), dtype=torch.long)
    e = ece_from_logits(logits, labels)
    assert e > 0.05
