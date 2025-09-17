from __future__ import annotations

"""plora.grpc.client - simple fetch client for plasmid LoRAs."""

import hashlib
import io
import logging
import tarfile
from pathlib import Path
from typing import Optional

import grpc

from ..manifest import Manifest
from ..signer import verify_with_tag, ADAPTER_TAG
from . import plora_pb2, plora_pb2_grpc

log = logging.getLogger(__name__)


def _extract_tar_bytes(tar_bytes: bytes, dest_dir: Path):
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
        tar.extractall(dest_dir)


async def fetch_plasmid(
    domain: str,
    dest_dir: Path,
    host: str = "localhost",
    port: int = 50051,
    public_key: Optional[Path] = None,
    *,
    tls: bool = False,
    root_cert: Optional[Path] = None,
    max_msg_mb: int = 64,
):
    """Fetch plasmid by domain and write to *dest_dir*; verify SHA + signature.

    Args:
        domain: domain name to request.
        dest_dir: destination directory to extract adapter.
        host: server host.
        port: server port.
        public_key: optional RSA public key for signature verification.
        tls: enable TLS if True.
        root_cert: optional root CA for TLS.
        max_msg_mb: gRPC max send/receive message size (MB) to allow large adapters.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = f"{host}:{port}"
    opts = [
        ("grpc.max_send_message_length", max_msg_mb * 1024 * 1024),
        ("grpc.max_receive_message_length", max_msg_mb * 1024 * 1024),
    ]
    if tls:
        if root_cert is not None and root_cert.exists():
            creds = grpc.ssl_channel_credentials(
                root_certificates=root_cert.read_bytes()
            )
        else:
            creds = grpc.ssl_channel_credentials()
        channel = grpc.aio.secure_channel(target, creds, options=opts)
    else:
        channel = grpc.aio.insecure_channel(target, options=opts)
    async with channel:
        stub = plora_pb2_grpc.PlasmidStub(channel)
        resp: plora_pb2.PlasmidReply = await stub.OfferPlasmid(
            plora_pb2.PlasmidRequest(domain=domain)
        )

    if resp.status != "OK":
        raise RuntimeError(f"Server returned status {resp.status}")

    sha_hex = hashlib.sha256(resp.bytes).hexdigest()
    if sha_hex != resp.sha:
        raise ValueError("SHA mismatch: payload corrupted")

    if public_key is not None and resp.signature:
        if not verify_with_tag(
            public_key, sha_hex, resp.signature.decode(), ADAPTER_TAG
        ):
            raise ValueError("RSA signature verification failed")
        log.info("Signature verified for %s", domain)

    # Extract bytes
    _extract_tar_bytes(resp.bytes, dest_dir)

    # Validate manifest matches artefact hash
    man_path = dest_dir / "plora.yml"
    manifest = Manifest.load(man_path)
    manifest.validate_artifact_hash(dest_dir)
    log.info(
        "Fetched and validated plasmid %s (%.2f MB)",
        manifest.plasmid_id,
        len(resp.bytes) / (1024 * 1024),
    )

    return manifest
