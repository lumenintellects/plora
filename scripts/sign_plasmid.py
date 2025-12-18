from __future__ import annotations

import argparse
import logging
from pathlib import Path

from plora.manifest import Manifest
from plora.signer import sign_with_tag, ADAPTER_TAG
from plora.logging_cfg import setup_logging

log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Sign a plasmid (adapter directory) with RSA key."
    )
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--private-key", type=Path, required=True)
    args = parser.parse_args()

    setup_logging()

    manifest = Manifest.from_adapter(args.adapter_dir)
    sha_hex = manifest.artifacts.sha256
    sig_b64 = sign_with_tag(args.private_key, sha_hex, ADAPTER_TAG)

    manifest.signer.algo = "RSA-PSS-SHA256"
    manifest.signer.signature_b64 = sig_b64

    manifest.dump(args.adapter_dir / "plora.yml")
    log.info("Signed manifest updated for %s", args.adapter_dir)


if __name__ == "__main__":
    main()
