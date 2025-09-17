from __future__ import annotations

import argparse
import asyncio
import logging

from plora.logging_cfg import setup_logging
from plora.grpc.server import run_server


async def _main_async(
    host: str,
    port: int,
    root: str,
    key: str | None,
    tls_cert: str | None,
    tls_key: str | None,
    max_msg_mb: int,
):
    await run_server(
        host, port, root, key, tls_cert=tls_cert, tls_key=tls_key, max_msg_mb=max_msg_mb
    )


def main():
    parser = argparse.ArgumentParser(description="Launch Plasmid gRPC offer server.")
    parser.add_argument(
        "--root",
        default="plasmids",
        help="Root directory containing adapter dirs by domain",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument(
        "--private-key", help="Optional RSA private key to sign payloads"
    )
    parser.add_argument("--tls-cert")
    parser.add_argument("--tls-key")
    parser.add_argument(
        "--max-msg-mb", type=int, default=64, help="gRPC max message size (MB)"
    )
    args = parser.parse_args()

    setup_logging()
    try:
        asyncio.run(
            _main_async(
                args.host,
                args.port,
                args.root,
                args.private_key,
                args.tls_cert,
                args.tls_key,
                args.max_msg_mb,
            )
        )
    except KeyboardInterrupt:
        logging.info("Server stopped by user")


if __name__ == "__main__":
    main()
