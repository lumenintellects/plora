from __future__ import annotations

"""plora.grpc.server - lightweight unary gRPC server for plasmid distribution."""

import asyncio
import hashlib
import io
import logging
import os
import tarfile
from pathlib import Path
from typing import Optional

import grpc

from ..signer import sign_with_tag, ADAPTER_TAG
from . import plora_pb2, plora_pb2_grpc

log = logging.getLogger(__name__)


def _tar_gz_directory(dir_path: Path) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(dir_path, arcname=".")
    return buf.getvalue()


class PlasmidServicer(plora_pb2_grpc.PlasmidServicer):
    def __init__(self, adapters_root: Path, private_key: Optional[Path] = None):
        self._root = adapters_root
        self._priv_key = private_key

    async def OfferPlasmid(self, request):
        domain = request.domain
        adapter_dir = self._root / domain
        if not adapter_dir.exists():
            return plora_pb2.PlasmidReply(status="NOT_FOUND")
        try:
            payload = _tar_gz_directory(adapter_dir)
            sha_hex = hashlib.sha256(payload).hexdigest()
            manifest_json = (adapter_dir / "plora.yml").read_text()

            signature_b64 = ""
            if self._priv_key:
                signature_b64 = sign_with_tag(self._priv_key, sha_hex, ADAPTER_TAG)

            return plora_pb2.PlasmidReply(
                status="OK",
                bytes=payload,
                sha=sha_hex,
                manifest=manifest_json,
                signature=signature_b64.encode(),
            )
        except Exception as exc:
            log.exception("Error while offering plasmid for %s", domain)
            return plora_pb2.PlasmidReply(status="ERROR", manifest=str(exc))


async def run_server(
    host: str = "0.0.0.0",
    port: int = 50051,
    adapters_root: str | Path = "plasmids",
    key_path: str | None = None,
    *,
    tls_cert: str | None = None,
    tls_key: str | None = None,
):
    server = grpc.aio.server()
    servicer = PlasmidServicer(
        Path(adapters_root), Path(key_path) if key_path else None
    )
    plora_pb2_grpc.add_PlasmidServicer_to_server(servicer, server)
    listen_addr = f"{host}:{port}"
    if tls_cert and tls_key:
        with open(tls_key, "rb") as fkey, open(tls_cert, "rb") as fcert:
            server_creds = grpc.ssl_server_credentials([(fkey.read(), fcert.read())])
        server.add_secure_port(listen_addr, server_creds)
        log.info("Starting secure gRPC server (TLS) on %s", listen_addr)
    else:
        server.add_insecure_port(listen_addr)
        log.info("Starting insecure gRPC server on %s", listen_addr)
    await server.start()
    await server.wait_for_termination()


if __name__ == "__main__":
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    asyncio.run(run_server())
