from __future__ import annotations

import argparse
import logging
from pathlib import Path

from plora.manifest import Manifest
from plora.signer import verify_sha256_hex
from plora.logging_cfg import setup_logging

log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Verify manifest SHA and RSA signature."
    )
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--public-key", type=Path, required=True)
    args = parser.parse_args()

    setup_logging()

    manifest = Manifest.from_adapter(args.adapter_dir)
    try:
        manifest.validate_artifact_hash(args.adapter_dir)
        log.info("SHA256 matches adapter bytes.")
    except Exception as exc:
        log.error("SHA256 validation failed: %s", exc)
        raise SystemExit(1)

    ok = verify_sha256_hex(
        args.public_key, manifest.artifacts.sha256, manifest.signer.signature_b64
    )
    if not ok:
        log.error("Signature verification FAILED")
        raise SystemExit(2)
    log.info("Signature verified successfully.")


if __name__ == "__main__":
    main()
