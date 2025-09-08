from __future__ import annotations

"""plora.metrics - evaluation helpers (perplexity, EM, chrF).

Only *perplexity* is required for Prototype v0.  Additional metrics are stubbed.
"""

from typing import List, Tuple
import math

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from .compat import get_device


@torch.no_grad()
def perplexity(model: PreTrainedModel, tok: PreTrainedTokenizer, dataset: List[Tuple[str, str]]) -> float:
    """Compute mean perplexity over *(prompt, answer)* pairs on the current device."""
    model.eval()
    device = get_device()
    total_log_prob = 0.0
    total_tokens = 0

    for prompt, answer in dataset:
        text = f"Question: {prompt}\nAnswer: {answer}"
        enc = tok(text, return_tensors="pt", truncation=True).to(device)
        labels = enc["input_ids"].clone()
        loss = model(**enc, labels=labels).loss
        n_tokens = labels.numel()
        total_log_prob += loss.item() * n_tokens
        total_tokens += n_tokens

    ppl = math.exp(total_log_prob / total_tokens)
    return float(ppl)


# ---------------------------------------------------------------------------
# Stubs for EM and chrF (not needed for CI but here for completeness)
# ---------------------------------------------------------------------------

def exact_match(*args, **kwargs):
    """Compute exact match (EM) percentage.

    Arguments should be `(predictions, references)` - lists of strings of equal
    length.  Returns a float in [0, 1].  Tokenisation is naive: we normalise
    whitespace and compare strings case-sensitively, mirroring common EM evals
    in QA tasks.
    """

    if len(args) == 2:
        preds, gts = args
    else:
        preds = kwargs.get("predictions")
        gts = kwargs.get("references")

    if preds is None or gts is None:
        raise ValueError("predictions and references required")
    if len(preds) != len(gts):
        raise ValueError("predictions and references must align")

    def _norm(s: str) -> str:
        return " ".join(s.strip().split())

    correct = sum(_norm(p) == _norm(t) for p, t in zip(preds, gts))
    return correct / len(preds)


def chrf_score(*args, **kwargs):
    """Compute chrF2 score (0-100) using sacrebleu."""

    try:
        from sacrebleu.metrics import CHRF

        if len(args) == 2:
            preds, refs = args
        else:
            preds = kwargs.get("predictions")
            refs = kwargs.get("references")
        if preds is None or refs is None:
            raise ValueError("predictions and references required")

        metric = CHRF(word_order=2)
        score = metric.corpus_score(preds, [refs]).score
        return score / 100.0  # normalise to 0-1 for convenience
    except ModuleNotFoundError:
        # Fallback, return 0, signalling metric unavailable in minimal env.
        return 0.0


@torch.no_grad()
def token_nlls(
    model: PreTrainedModel,
    tok: PreTrainedTokenizer,
    dataset: List[Tuple[str, str]],
) -> List[float]:
    """Return *average* negative-log-likelihood (in nats) per example.

    Each item in *dataset* is a *(prompt, answer)* pair.  The function computes the
    cross-entropy loss under teacher forcing, equivalent to *mean token NLL* for
    the concatenated "Question + Answer" text.  We purposely return **per-example**
    values so that paired statistical tests can be applied downstream.
    """

    model.eval()
    device = get_device()

    nlls: List[float] = []
    for prompt, answer in dataset:
        text = f"Question: {prompt}\nAnswer: {answer}"
        enc = tok(text, return_tensors="pt", truncation=True).to(device)
        labels = enc["input_ids"].clone()
        # HuggingFace returns mean loss (cross-entropy) over tokens
        loss: torch.Tensor = model(**enc, labels=labels).loss
        nlls.append(float(loss.item()))

    return nlls


def _sign_test(delta: List[float]) -> float:
    """Two-sided sign test p-value as fallback when SciPy is unavailable."""

    import math
    n_pos = sum(d > 0 for d in delta)
    n_neg = sum(d < 0 for d in delta)
    n = n_pos + n_neg  # exclude zeros
    if n == 0:
        return 1.0  # no information

    # Use cumulative binomial prob of observing <= min(n_pos, n_neg) successes
    k = min(n_pos, n_neg)
    log_p = 0.0
    # sum_{i=0}^k C(n, i) * 0.5^n (two-sided)
    for i in range(k + 1):
        log_p += math.comb(n, i)
    p = log_p * (0.5 ** n) * 2  # two-sided
    return min(p, 1.0)


def paired_wilcoxon(delta: List[float]) -> dict[str, float]:  # noqa: D401
    """Paired Wilcoxon signed-rank test with SciPy fallback.

    *delta* is a list of per-example differences (after / before).  The function
    returns a dictionary with keys ``stat`` and ``p``.  When *scipy* is
    available, we delegate to ``scipy.stats.wilcoxon`` (mode="auto").  Otherwise
    we fall back to a simple sign test (less powerful but distribution-free).
    """

    try:
        from scipy.stats import wilcoxon  # type: ignore

        stat, p = wilcoxon(delta, zero_method="pratt", correction=False, alternative="two-sided")
        return {"stat": float(stat), "p": float(p)}
    except ModuleNotFoundError:  # pragma: no cover – SciPy absent in minimal CI
        p = _sign_test(delta)
        # For consistency, return the *number of non-zero deltas* as the stat.
        stat = float(sum(d != 0 for d in delta))
        return {"stat": stat, "p": float(p)}

# ---------------------------------------------------------------------------
# Generic bootstrap helper (for EM, chrF, etc.)
# ---------------------------------------------------------------------------


def bootstrap_ci(
    xs: List[float],
    ys: List[float],
    *,
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Paired bootstrap confidence interval of the mean difference (y - x).

    Parameters
    ----------
    xs, ys : list[float]
        Metric values for baseline (xs) and treatment (ys) on the *same* items.
    n_resamples : int, default 1000
    ci : float, default 0.95
        Confidence level.
    seed : int
        RNG seed for reproducibility.
    """

    if len(xs) != len(ys):
        raise ValueError("xs and ys must be the same length for paired bootstrap")

    import random

    diffs = [y - x for x, y in zip(xs, ys)]
    rng = random.Random(seed)
    boot_means: List[float] = []
    n = len(diffs)
    for _ in range(n_resamples):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        boot_means.append(sum(sample) / n)

    boot_means.sort()
    lower_idx = int(((1 - ci) / 2) * n_resamples)
    upper_idx = int((1 - (1 - ci) / 2) * n_resamples) - 1
    return boot_means[lower_idx], boot_means[upper_idx]
