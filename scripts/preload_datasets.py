from __future__ import annotations

import argparse
import logging
from typing import Iterable

from plora.dataset_loader import get_dataset, RealDatasetLoader

logger = logging.getLogger(__name__)


def preload(domains: Iterable[str], splits: Iterable[str], samples: int | None) -> None:
    for split in splits:
        for dom in domains:
            logger.info("Preloading domain=%s split=%s (limit=%s)", dom, split, samples)
            get_dataset(dom, max_samples=samples, split=split)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Warm HuggingFace cache for Plora datasets.")
    parser.add_argument(
        "--domains",
        default="arithmetic,legal,medical",
        help="Comma-separated list of domains to preload.",
    )
    parser.add_argument(
        "--splits",
        default="train,validation",
        help="Comma-separated list of splits to preload.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Optional sample cap per domain (defaults to full dataset).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    RealDatasetLoader.set_sample_limit(args.samples)
    try:
        preload(domains, splits, args.samples)
    finally:
        RealDatasetLoader.set_sample_limit(None)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

