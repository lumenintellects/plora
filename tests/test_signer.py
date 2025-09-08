from __future__ import annotations

from pathlib import Path

from plora.signer import generate_keypair, sign_sha256_hex, verify_sha256_hex


def test_sign_verify(tmp_path: Path):
    priv = tmp_path / "priv.pem"
    pub = tmp_path / "pub.pem"
    generate_keypair(priv, pub)

    digest = "a" * 64  # fake sha256 hex
    sig_b64 = sign_sha256_hex(priv, digest)
    assert verify_sha256_hex(pub, digest, sig_b64) is True
    # Negative case
    assert verify_sha256_hex(pub, "b" * 64, sig_b64) is False
