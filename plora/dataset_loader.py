from __future__ import annotations

"""plora.dataset_loader - Real dataset loaders for experiments.
"""

from typing import List, Tuple, Optional
from collections import Counter
import os
import time

import logging
from datasets import load_dataset

SEED = 42
logger = logging.getLogger(__name__)
_RETRIES_DEFAULT = int(os.getenv("PLORA_DATASET_RETRIES", "3") or "3")
_BACKOFF_DEFAULT = float(os.getenv("PLORA_DATASET_RETRY_BACKOFF", "5") or "5")

# Threshold for detecting placeholder/constant answers
# If more than this fraction of answers are identical, it's likely placeholder data
# 50% is chosen because real Q&A datasets should have diverse answers
# Only applied when we have enough samples (>= 20) to make the check meaningful
_PLACEHOLDER_THRESHOLD = 0.5  # 50% - if >50% answers are the same, it's likely placeholder
_MIN_SAMPLES_FOR_CHECK = 20  # Need at least this many samples to check for placeholders


# ---------------------------------------------------------------------------
# Data Quality Validation
# ---------------------------------------------------------------------------


def validate_qa_pairs(
    pairs: List[Tuple[str, str]],
    domain: str,
    *,
    strict: bool = True,
) -> List[Tuple[str, str]]:
    """Validate Q&A pairs and detect placeholder/synthetic data.

    This function checks for:
    1. Constant/repeated answers (placeholder data)
    2. Empty questions or answers
    3. Suspiciously short answers
    4. Synthetic question patterns

    Parameters
    ----------
    pairs : list of (question, answer)
        The Q&A pairs to validate.
    domain : str
        Domain name for logging context.
    strict : bool
        If True, raise RuntimeError on critical issues. If False, only log warnings.

    Returns
    -------
    list of (question, answer)
        Validated pairs (unchanged if valid).

    Raises
    ------
    RuntimeError
        If strict=True and critical data quality issues are found.
    """
    if not pairs:
        msg = f"[DATA QUALITY ERROR] Domain '{domain}': No Q&A pairs loaded - dataset is empty!"
        logger.error(msg)
        if strict:
            raise RuntimeError(msg)
        return pairs

    # Filter out pairs with empty/None questions or answers
    original_count = len(pairs)
    pairs = [
        (q, a)
        for q, a in pairs
        if q is not None and a is not None and str(q).strip() and str(a).strip()
    ]
    filtered_empty = original_count - len(pairs)

    if filtered_empty > 0:
        logger.warning(
            "[DATA QUALITY FIX] Domain '%s': Filtered out %d/%d pairs with empty/None Q or A (keeping %d valid pairs)",
            domain,
            filtered_empty,
            original_count,
            len(pairs),
        )

    if not pairs:
        msg = f"[DATA QUALITY ERROR] Domain '{domain}': ALL pairs were filtered out - no valid data!"
        logger.error(msg)
        if strict:
            raise RuntimeError(msg)
        return pairs

    answers = [str(a) for _, a in pairs]  # Convert to string to be safe
    answer_counts = Counter(answers)
    most_common_answer, most_common_count = answer_counts.most_common(1)[0]
    repeat_ratio = most_common_count / len(pairs)

    if len(pairs) >= _MIN_SAMPLES_FOR_CHECK and repeat_ratio > _PLACEHOLDER_THRESHOLD:
        answer_preview = str(most_common_answer)[:100] if most_common_answer else "(empty)"
        msg = (
            f"[DATA QUALITY CRITICAL] Domain '{domain}': PLACEHOLDER DATA DETECTED!\n"
            f"  - Most common answer appears {most_common_count}/{len(pairs)} times "
            f"({repeat_ratio:.1%} of dataset)\n"
            f"  - Answer: '{answer_preview}...'\n"
            f"  - This indicates FAKE/SYNTHETIC data, NOT real Q&A pairs!\n"
            f"  - The model will learn NOTHING useful from this data."
        )
        logger.critical(msg)
        if strict:
            raise RuntimeError(msg)
    elif len(pairs) < _MIN_SAMPLES_FOR_CHECK:
        logger.debug(
            "Skipping placeholder check for domain '%s': only %d samples (need >= %d)",
            domain,
            len(pairs),
            _MIN_SAMPLES_FOR_CHECK,
        )

    # Check for suspiciously short answers
    short_count = sum(1 for _, a in pairs if len(str(a).strip()) < 10)
    if short_count > len(pairs) * 0.5:  # More than 50% very short
        logger.warning(
            "[DATA QUALITY WARNING] Domain '%s': %d/%d answers are suspiciously short (<10 chars)",
            domain,
            short_count,
            len(pairs),
        )

    # Check for synthetic question patterns
    synthetic_patterns = [
        "Which legal domain best fits:",
        "Classify this document:",
        "What category is this:",
    ]
    synthetic_q_count = sum(
        1
        for q, _ in pairs
        if any(pattern.lower() in str(q).lower() for pattern in synthetic_patterns)
    )
    if synthetic_q_count > len(pairs) * 0.9:  # >90% synthetic questions
        logger.warning(
            "[DATA QUALITY WARNING] Domain '%s': %d/%d questions appear to be synthetically generated",
            domain,
            synthetic_q_count,
            len(pairs),
        )

    # Log successful validation
    unique_answers = len(answer_counts)
    logger.info(
        "[DATA QUALITY OK] Domain '%s': %d pairs loaded, %d unique answers (%.1f%% diversity)",
        domain,
        len(pairs),
        unique_answers,
        (unique_answers / len(pairs)) * 100,
    )

    return pairs


# ---------------------------------------------------------------------------
# RealDatasetLoader
# ---------------------------------------------------------------------------


class RealDatasetLoader:
    """Domain-specific dataset helpers that obey an optional global sample cap.

    IMPORTANT: All loaders MUST return real Q&A pairs from the source dataset.
    Placeholder or constant answers are FORBIDDEN.
    """

    SAMPLE_LIMIT: Optional[int] = None

    @classmethod
    def set_sample_limit(cls, k: Optional[int]):
        """Configure the maximum number of rows to return per dataset."""
        cls.SAMPLE_LIMIT = k

    @classmethod
    def _hf_slice(
        cls,
        name: str,
        subset: Optional[str] = None,
        *,
        split: str = "train",
        retries: int | None = None,
        backoff: float | None = None,
    ):
        """Download split with streaming, deterministic shuffle, and hard-cap to SAMPLE_LIMIT.

        Parameters
        ----------
        name : str
            HF dataset name.
        subset : str | None
            HF dataset subset/config name when applicable.
        split : str
            Which split to load (e.g., "train", "validation", "test"). Defaults to "train".
        """
        max_attempts = max(1, retries if retries is not None else _RETRIES_DEFAULT)
        backoff_sec = backoff if backoff is not None else _BACKOFF_DEFAULT
        last_exc: Exception | None = None

        logger.info(
            "Loading HF dataset: %s (subset=%s, split=%s)",
            name,
            subset or "default",
            split,
        )

        for attempt in range(1, max_attempts + 1):
            try:
                ds = load_dataset(
                    name,
                    subset,
                    split=split,
                    streaming=False,
                    download_mode="reuse_cache_if_exists",
                )
                ds = ds.shuffle(seed=SEED)
                if cls.SAMPLE_LIMIT is not None:
                    limit = min(len(ds), int(cls.SAMPLE_LIMIT))
                    ds = ds.select(range(limit))
                logger.info(
                    "Successfully loaded %s/%s split=%s: %d examples",
                    name,
                    subset or "default",
                    split,
                    len(ds),
                )
                return ds
            except Exception as e:
                last_exc = e
                logger.warning(
                    "Failed to load HF dataset %s split=%s (attempt %d/%d): %s",
                    name,
                    split,
                    attempt,
                    max_attempts,
                    e,
                )
                if attempt < max_attempts:
                    sleep_for = max(0.0, backoff_sec) * attempt
                    if sleep_for:
                        time.sleep(sleep_for)

        raise RuntimeError(
            f"Failed to load HF dataset {name} split={split} after {max_attempts} attempts: {last_exc}"
        )

    @classmethod
    def load_arithmetic_data(cls, *, split: str = "train"):
        """Load arithmetic reasoning data from deepmind/aqua_rat.

        Returns real question-rationale pairs from the dataset.
        Filters out examples with empty questions or rationales at source level.
        """
        ds = cls._hf_slice("deepmind/aqua_rat", "raw", split=split)
        if ds is None:
            raise RuntimeError("Arithmetic dataset unavailable; aborting experiment.")

        # Extract REAL Q&A pairs, filtering out empty ones at source
        pairs = []
        skipped_empty = 0
        for ex in ds:
            question = ex.get("question", "")
            rationale = ex.get("rationale", "")

            # Skip if question or rationale is empty/None
            if not question or not question.strip():
                skipped_empty += 1
                continue
            if not rationale or not rationale.strip():
                skipped_empty += 1
                continue

            pairs.append((question.strip(), rationale.strip()))

        if skipped_empty > 0:
            logger.info(
                "Arithmetic data: skipped %d examples with empty Q/A at source",
                skipped_empty,
            )

        logger.info(
            "Arithmetic data: extracted %d real Q&A pairs from deepmind/aqua_rat",
            len(pairs),
        )
        return pairs

    @classmethod
    def load_legal_data(cls, *, split: str = "train"):
        """Load legal reasoning data from lex_glue/case_hold.

        Uses CaseHOLD dataset which has real legal case holdings as answers.
        Each example contains a case context and the correct legal holding.
        """
        # Use CaseHOLD from lex_glue - it has real legal Q&A pairs
        ds = cls._hf_slice("lex_glue", "case_hold", split=split)
        if ds is None:
            raise RuntimeError("Legal dataset (case_hold) unavailable; aborting experiment.")

        pairs = []
        for ex in ds:
            # CaseHOLD structure:
            # - context: the legal case context with <HOLDING> placeholder
            # - endings: list of 5 possible holdings
            # - label: index of the correct holding (0-4)
            context = ex["context"]
            label = ex["label"]
            holdings = ex["endings"]

            if label < 0 or label >= len(holdings):
                logger.warning("Invalid label %d for case_hold example, skipping", label)
                continue

            correct_holding = holdings[label]

            # Create Q&A pair with the case context as question and correct holding as answer
            # Truncate context to reasonable length but keep enough for legal reasoning
            question = context[:1000] if len(context) > 1000 else context
            answer = correct_holding

            if question.strip() and answer.strip():
                pairs.append((question, answer))

        logger.info(
            "Legal data: extracted %d real Q&A pairs from lex_glue/case_hold",
            len(pairs),
        )
        return pairs

    @classmethod
    def load_medical_data(cls, *, split: str = "train"):
        """Load medical reasoning data from openlifescienceai/medmcqa.

        Returns real question-explanation pairs from the dataset.
        Filters out examples with empty explanations at source level.
        """
        ds = cls._hf_slice("openlifescienceai/medmcqa", split=split)
        if ds is None:
            raise RuntimeError("Medical dataset unavailable; aborting experiment.")

        # Extract REAL Q&A pairs, filtering out empty ones at source
        pairs = []
        skipped_empty = 0
        for ex in ds:
            question = ex.get("question", "")
            explanation = ex.get("exp", "")

            # Skip if question or explanation is empty/None
            if not question or not question.strip():
                skipped_empty += 1
                continue
            if not explanation or not explanation.strip():
                skipped_empty += 1
                continue

            pairs.append((question.strip(), explanation.strip()))

        if skipped_empty > 0:
            logger.info(
                "Medical data: skipped %d examples with empty Q/A at source",
                skipped_empty,
            )

        logger.info(
            "Medical data: extracted %d real Q&A pairs from openlifescienceai/medmcqa",
            len(pairs),
        )
        return pairs


DOMAIN_LOADERS = {
    "legal": RealDatasetLoader.load_legal_data,
    "arithmetic": RealDatasetLoader.load_arithmetic_data,
    "medical": RealDatasetLoader.load_medical_data,
}


def get_dataset(
    domain: str,
    max_samples: int | None = None,
    *,
    split: str | None = None,
    validate: bool = True,
    strict_validation: bool = True,
) -> List[Tuple[str, str]]:
    """Return a list of real QA pairs for *domain* using HF datasets.

    Parameters
    ----------
    domain : str
        One of {legal, arithmetic, medical}.
    max_samples : int | None
        Optional cap; forwarded to RealDatasetLoader.SAMPLE_LIMIT.
    split : str | None
        Which split to load (train|validation|test). Defaults to env PLORA_SPLIT or "train".
    validate : bool
        If True (default), validate the loaded data for quality issues.
    strict_validation : bool
        If True (default), raise errors on critical data quality issues.
        If False, only log warnings.

    Returns
    -------
    list of (str, str)
        List of (question, answer) tuples with REAL data.

    Raises
    ------
    KeyError
        If domain is unknown.
    RuntimeError
        If dataset loading fails or validation detects placeholder data.
    """
    try:
        loader_fn = DOMAIN_LOADERS[domain]
    except KeyError:
        raise KeyError(f"Unknown domain '{domain}'") from None

    if max_samples is not None:
        RealDatasetLoader.set_sample_limit(max_samples)
    else:
        env_cap = os.getenv("PLORA_SAMPLES")
        RealDatasetLoader.set_sample_limit(int(env_cap) if env_cap else None)

    eff_split = split or os.getenv("PLORA_SPLIT", "train")

    logger.info("Loading dataset for domain='%s' split='%s'", domain, eff_split)
    pairs = loader_fn(split=eff_split)

    # ALWAYS validate data quality to catch placeholder/synthetic data
    if validate:
        pairs = validate_qa_pairs(
            pairs,
            domain,
            strict=strict_validation,
        )

    return pairs
