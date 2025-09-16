from __future__ import annotations

"""plora.dataset_loader - deterministic tiny fixtures for unit tests and CI.
"""

from typing import List, Tuple, Optional
import os

import logging
from datasets import load_dataset

SEED = 42
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# RealDatasetLoader
# ---------------------------------------------------------------------------


class RealDatasetLoader:
    """Domain-specific dataset helpers that obey an optional global sample cap."""

    SAMPLE_LIMIT: Optional[int] = None

    # helper methods
    @classmethod
    def set_sample_limit(cls, k: Optional[int]):
        """Configure the maximum number of rows to return per dataset."""
        cls.SAMPLE_LIMIT = k

    @classmethod
    def _hf_slice(
        cls, name: str, subset: Optional[str] = None, *, split: str = "train"
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
        try:
            # Use streaming for large datasets to avoid loading everything into RAM
            ds = load_dataset(name, subset, split=split, streaming=True)
            ds = ds.shuffle(
                buffer_size=10_000, seed=SEED
            )  # Use buffer for streaming shuffle

            # Convert to list with sample limit
            data = []
            for i, example in enumerate(ds):
                if cls.SAMPLE_LIMIT is not None and i >= cls.SAMPLE_LIMIT:
                    break
                data.append(example)

            # Convert back to dataset for consistency
            from datasets import Dataset as HFDataset

            return HFDataset.from_list(data) if data else None

        except Exception as e:
            raise RuntimeError(f"Failed to load HF dataset {name} split={split}: {e}")

    @classmethod
    def load_arithmetic_data(cls, *, split: str = "train"):
        ds = cls._hf_slice("deepmind/aqua_rat", "raw", split=split)
        if ds is None:
            raise RuntimeError("Arithmetic dataset unavailable; aborting experiment.")
        return [(ex["question"], ex["rationale"]) for ex in ds]

    @classmethod
    def load_legal_data(cls, *, split: str = "train"):
        ds = cls._hf_slice("lex_glue", "eurlex", split=split)
        if ds is None:
            raise RuntimeError("Legal dataset unavailable; aborting experiment.")
        qas = []
        for ex in ds:
            snippet = ex["text"][:100].replace("\n", " ")
            qas.append(
                (f"Which legal domain best fits: {snippet}?", "Legal classification")
            )
        return qas

    @classmethod
    def load_medical_data(cls, *, split: str = "train"):
        ds = cls._hf_slice("openlifescienceai/medmcqa", split=split)
        if ds is None:
            raise RuntimeError("Medical dataset unavailable; aborting experiment.")
        return [(ex["question"], ex["exp"]) for ex in ds]


DOMAIN_LOADERS = {
    "legal": RealDatasetLoader.load_legal_data,
    "arithmetic": RealDatasetLoader.load_arithmetic_data,
    "medical": RealDatasetLoader.load_medical_data,
}


def get_dataset(
    domain: str, max_samples: int | None = None, *, split: str | None = None
) -> List[Tuple[str, str]]:
    """Return a list of real QA pairs for *domain* using HF datasets.

    Parameters
    ----------
    domain : str
        One of {legal, arithmetic, medical, coding}.
    max_samples : int | None
        Optional cap; forwarded to RealDatasetLoader.SAMPLE_LIMIT.
    split : str | None
        Which split to load (train|validation|test). Defaults to env PLORA_SPLIT or "train".
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
    return loader_fn(split=eff_split)
