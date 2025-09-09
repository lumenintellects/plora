from __future__ import annotations

import base64
import hashlib
from pathlib import Path

from plora.signer import generate_keypair, sign_sha256_hex, verify_sha256_hex


def test_signature_happy_and_sad(tmp_path: Path):
    priv = tmp_path / "priv.pem"
    pub = tmp_path / "pub.pem"
    generate_keypair(priv, pub)

    payload = b"hello"
    sha = hashlib.sha256(payload).hexdigest()

    sig = sign_sha256_hex(priv, sha)
    assert verify_sha256_hex(pub, sha, sig)

    # corrupt signature
    bad = base64.b64encode(base64.b64decode(sig)[:-1] + b"\x00").decode()
    assert not verify_sha256_hex(pub, sha, bad)


