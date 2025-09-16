from __future__ import annotations

"""plora.metrics - evaluation helpers (perplexity, EM, chrF).

Only *perplexity* is required for Prototype v0.  Additional metrics are stubbed.
"""

from typing import List, Tuple
import math

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from .compat import get_device


def _model_device(model: PreTrainedModel) -> torch.device:
    try:
        # transformers models expose parameters/buffers
        return next(model.parameters()).device
    except Exception:
        return get_device()


@torch.no_grad()
def perplexity(
    model: PreTrainedModel, tok: PreTrainedTokenizer, dataset: List[Tuple[str, str]]
) -> float:
    """Compute mean perplexity over *(prompt, answer)* pairs on the current device."""
    model.eval()
    device = _model_device(model)
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
    device = _model_device(model)

    nlls: List[float] = []
    for prompt, answer in dataset:
        text = f"Question: {prompt}\nAnswer: {answer}"
        enc = tok(text, return_tensors="pt", truncation=True).to(device)
        labels = enc["input_ids"].clone()
        # HuggingFace returns mean loss (cross-entropy) over tokens
        loss: torch.Tensor = model(**enc, labels=labels).loss
        nlls.append(float(loss.item()))

    return nlls


# ---------------------------------------------------------------------------
# Distributional distances on logits: KL and Jensen–Shannon
# ---------------------------------------------------------------------------


@torch.no_grad()
def _softmax_logprobs(logits: torch.Tensor) -> torch.Tensor:
    """Return log-probabilities along last dimension in a numerically stable way."""
    return torch.log_softmax(logits, dim=-1)


@torch.no_grad()
def kl_divergence_logits(
    p_logits: torch.Tensor, q_logits: torch.Tensor
) -> torch.Tensor:
    """Compute KL(P||Q) per position given unnormalised logits of same shape.

    Returns KL values reduced over the class dimension, preserving leading dims.
    """
    p_logp = _softmax_logprobs(p_logits)
    q_logp = _softmax_logprobs(q_logits)
    p = p_logp.exp()
    kl = (p * (p_logp - q_logp)).sum(dim=-1)
    return kl


@torch.no_grad()
def js_divergence_logits(
    p_logits: torch.Tensor, q_logits: torch.Tensor
) -> torch.Tensor:
    """Compute Jensen–Shannon divergence per position from two logits tensors.

    JS(P, Q) = 0.5 KL(P || M) + 0.5 KL(Q || M), where M = 0.5 (P + Q).
    Returns values reduced over the class dimension.
    """
    p_logp = _softmax_logprobs(p_logits)
    q_logp = _softmax_logprobs(q_logits)
    p = p_logp.exp()
    q = q_logp.exp()
    m = 0.5 * (p + q)
    # Convert m to log-space
    m_logp = torch.log(m.clamp_min(1e-12))
    kl_pm = (p * (p_logp - m_logp)).sum(dim=-1)
    kl_qm = (q * (q_logp - m_logp)).sum(dim=-1)
    return 0.5 * (kl_pm + kl_qm)


@torch.no_grad()
def dataset_kl_js(
    model_p: PreTrainedModel,
    model_q: PreTrainedModel,
    tok: PreTrainedTokenizer,
    dataset: List[Tuple[str, str]],
) -> dict[str, float]:
    """Compute mean KL(P||Q), KL(Q||P) and JS over a dataset under teacher forcing.

    The models are evaluated on identical token sequences "Question + Answer" and
    divergences are averaged across tokens and examples.
    """
    device = _model_device(model_p)
    model_p.eval()
    model_q.eval()
    total_kl_pq = 0.0
    total_kl_qp = 0.0
    total_js = 0.0
    total_tokens = 0

    for prompt, answer in dataset:
        text = f"Question: {prompt}\nAnswer: {answer}"
        enc = tok(text, return_tensors="pt", truncation=True).to(device)
        labels = enc["input_ids"].clone()
        # Obtain logits from each model
        logits_p: torch.Tensor = model_p(**enc).logits  # [B, T, V]
        logits_q: torch.Tensor = model_q(**enc).logits
        # Shift to align next-token prediction with labels
        p_shift = logits_p[:, :-1, :]
        q_shift = logits_q[:, :-1, :]
        y = labels[:, 1:]
        # Mask padding positions (if any)
        mask = torch.ones_like(y, dtype=torch.bool)

        # Compute per-position divergences and then mask/reduce
        kl_pq = kl_divergence_logits(p_shift, q_shift)
        kl_qp = kl_divergence_logits(q_shift, p_shift)
        js = js_divergence_logits(p_shift, q_shift)

        total_kl_pq += float(kl_pq[mask].sum().item())
        total_kl_qp += float(kl_qp[mask].sum().item())
        total_js += float(js[mask].sum().item())
        total_tokens += int(mask.sum().item())

    if total_tokens == 0:
        return {"kl_pq": 0.0, "kl_qp": 0.0, "js": 0.0}
    inv = 1.0 / total_tokens
    return {
        "kl_pq": total_kl_pq * inv,
        "kl_qp": total_kl_qp * inv,
        "js": total_js * inv,
    }


# ---------------------------------------------------------------------------
# Inter-model MI proxy and transfer entropy (surrogates)
# ---------------------------------------------------------------------------


@torch.no_grad()
def inter_model_mi_js(
    model_a: PreTrainedModel,
    model_b: PreTrainedModel,
    tok: PreTrainedTokenizer,
    dataset: List[Tuple[str, str]],
) -> float:
    """Proxy for I(M;Y) where M ∈ {A,B}, average JS divergence between models.

    Uses JS between next-token distributions as a surrogate for how informative
    the model identity is about the emitted token.
    """
    stats = dataset_kl_js(model_a, model_b, tok, dataset)
    return float(stats["js"])  # already averaged per token


def transfer_entropy_proxy(
    series_a: List[float], series_b: List[float], *, k: int = 1
) -> float:
    """Simple transfer entropy proxy from A→B using lagged correlation.

    For CPU-friendliness, we approximate TE with the increase in R^2 when adding
    past A to a linear regression predicting B_t from B_{t-1..t-k}.
    Returns non-negative improvement proxy.
    """
    import numpy as np

    if len(series_a) != len(series_b) or len(series_a) <= k:
        return 0.0
    y = np.array(series_b[k:], dtype=float)
    Xb = np.stack(
        [np.array(series_b[i : i + len(y)], dtype=float) for i in range(k)], axis=1
    )
    Xa = np.stack(
        [np.array(series_a[i : i + len(y)], dtype=float) for i in range(k)], axis=1
    )

    # Fit least squares y ~ Xb
    def r2(X):
        try:
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            yhat = X @ beta
            ss_res = float(((y - yhat) ** 2).sum())
            ss_tot = float(((y - y.mean()) ** 2).sum())
            return 1.0 - (ss_res / ss_tot if ss_tot > 0 else 1.0)
        except Exception:
            return 0.0

    r2_b = r2(Xb)
    r2_ba = r2(np.concatenate([Xb, Xa], axis=1))
    return max(0.0, float(r2_ba - r2_b))


# ---------------------------------------------------------------------------
# Calibration: Expected Calibration Error (ECE)
# ---------------------------------------------------------------------------


@torch.no_grad()
def ece_from_logits(
    logits: torch.Tensor, labels: torch.Tensor, n_bins: int = 15
) -> float:
    """Compute Expected Calibration Error (ECE) on token-level predictions.

    logits: [B, T, V]; labels: [B, T] with -100 for ignored positions allowed.
    """
    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError("Shapes must be [B,T,V] and [B,T]")
    # Align next-token prediction
    p = torch.softmax(logits[:, :-1, :], dim=-1)
    y = labels[:, 1:]
    mask = y != -100
    if mask.sum().item() == 0:
        return 0.0
    conf, pred = torch.max(p, dim=-1)
    y_true = y
    correct = pred == y_true

    # Bin by confidence
    bins = torch.linspace(0.0, 1.0, steps=n_bins + 1, device=logits.device)
    ece = 0.0
    total = mask.sum().item()
    for b in range(n_bins):
        lo = bins[b]
        hi = bins[b + 1]
        in_bin = (conf >= lo) & (conf < hi) & mask
        cnt = in_bin.sum().item()
        if cnt == 0:
            continue
        acc = correct[in_bin].float().mean().item()
        conf_mean = conf[in_bin].float().mean().item()
        ece += (cnt / total) * abs(conf_mean - acc)
    return float(ece)


@torch.no_grad()
def dataset_ece(
    model: PreTrainedModel,
    tok: PreTrainedTokenizer,
    dataset: List[Tuple[str, str]],
    n_bins: int = 15,
) -> float:
    """Compute token-level ECE averaged over the dataset under teacher forcing."""
    device = _model_device(model)
    model.eval()
    total_ece = 0.0
    total_tokens = 0
    for prompt, answer in dataset:
        text = f"Question: {prompt}\nAnswer: {answer}"
        enc = tok(text, return_tensors="pt", truncation=True).to(device)
        labels = enc["input_ids"].clone()
        logits: torch.Tensor = model(**enc).logits
        # Use per-example ECE weighted by token count to avoid bias
        ece = ece_from_logits(logits, labels, n_bins=n_bins)
        total_ece += ece * (labels.numel() - labels.shape[0])  # exclude BOS
        total_tokens += labels.numel() - labels.shape[0]
    if total_tokens == 0:
        return 0.0
    return float(total_ece / total_tokens)


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
    p = log_p * (0.5**n) * 2  # two-sided
    return min(p, 1.0)


def paired_wilcoxon(delta: List[float]) -> dict[str, float]:
    """Paired Wilcoxon signed-rank test with SciPy fallback.

    *delta* is a list of per-example differences (after / before).  The function
    returns a dictionary with keys ``stat`` and ``p``.  When *scipy* is
    available, we delegate to ``scipy.stats.wilcoxon`` (mode="auto").  Otherwise
    we fall back to a simple sign test (less powerful but distribution-free).
    """

    try:
        from scipy.stats import wilcoxon

        stat, p = wilcoxon(
            delta, zero_method="pratt", correction=False, alternative="two-sided"
        )
        return {"stat": float(stat), "p": float(p)}
    except ModuleNotFoundError:
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
