from __future__ import annotations

import argparse
import asyncio
import logging

from plora.logging_cfg import setup_logging
from plora.grpc.server import run_server


async def _main_async(host: str, port: int, root: str, key: str | None):
    await run_server(host, port, root, key)


def main():
    parser = argparse.ArgumentParser(description="Launch Plasmid gRPC offer server.")
    parser.add_argument("--root", default="plasmids", help="Root directory containing adapter dirs by domain")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--private-key", help="Optional RSA private key to sign payloads")
    args = parser.parse_args()

    setup_logging()
    try:
        asyncio.run(_main_async(args.host, args.port, args.root, args.private_key))
    except KeyboardInterrupt:
        logging.info("Server stopped by user")


if __name__ == "__main__":
    main()
