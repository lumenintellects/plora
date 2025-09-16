from __future__ import annotations

import argparse
import logging
from pathlib import Path

from plora.loader import merge_plasmids
from plora.logging_cfg import setup_logging

log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple plasmids into one model."
    )
    parser.add_argument("--base-model", default="google/gemma-3-1b-it")
    parser.add_argument(
        "--plasmids",
        nargs="+",
        type=Path,
        required=True,
        help="List of adapter directories",
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="Directory to save merged model"
    )
    args = parser.parse_args()

    setup_logging()
    model = merge_plasmids(args.base_model, args.plasmids)
    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)
    log.info("Merged model saved to %s", args.out)


if __name__ == "__main__":
    main()
