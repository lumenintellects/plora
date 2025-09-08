from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from plora.grpc.client import fetch_plasmid
from plora.logging_cfg import setup_logging


async def _run(domain: str, dest: Path, host: str, port: int, pubkey: Path | None):
    await fetch_plasmid(domain, dest, host, port, pubkey)


def main():
    parser = argparse.ArgumentParser(description="Fetch plasmid by domain via gRPC and verify.")
    parser.add_argument("--domain", required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--public-key", type=Path)
    args = parser.parse_args()

    setup_logging()
    asyncio.run(_run(args.domain, args.dest, args.host, args.port, args.public_key))


if __name__ == "__main__":
    main()
