from __future__ import annotations

"""plora.signer - RSA-PSS SHA-256 signing utilities.

This module is intentionally *minimal*, it only signs / verifies the **hex**
string of a SHA-256 digest.  Callers are responsible for computing the digest
of the adapter bytes before invoking :func:`sign_sha256_hex`.
"""

from pathlib import Path
import base64
from typing import Union

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa, ed25519
from cryptography.hazmat.backends import default_backend


# ---------------------------------------------------------------------------
# Key management helpers
# ---------------------------------------------------------------------------


def generate_keypair(
    private_key_path: Path, public_key_path: Path, passphrase: str | None = None
) -> None:
    """Generate a 3072-bit RSA keypair on disk in PEM format.

    The private key is encrypted with *passphrase* if provided, otherwise
    written in plaintext (NOT recommended beyond CI tests).
    """
    private_key_path.parent.mkdir(parents=True, exist_ok=True)
    public_key_path.parent.mkdir(parents=True, exist_ok=True)

    key = rsa.generate_private_key(
        public_exponent=65537, key_size=3072, backend=default_backend()
    )

    enc_algo: serialization.KeySerializationEncryption
    if passphrase:
        enc_algo = serialization.BestAvailableEncryption(passphrase.encode())
    else:
        enc_algo = serialization.NoEncryption()

    private_key_path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=enc_algo,
        )
    )

    public_key_path.write_bytes(
        key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )


# ---------------------------------------------------------------------------
# Signing / verification helpers
# ---------------------------------------------------------------------------


def _load_private(path: Path, passphrase: str | None = None):
    return serialization.load_pem_private_key(
        path.read_bytes(), password=None if passphrase is None else passphrase.encode()
    )


def _load_public(path: Path):
    return serialization.load_pem_public_key(path.read_bytes())


def sign_sha256_hex(
    private_key_path: Path, sha256_hex: str, passphrase: str | None = None
) -> str:
    """Return Base64-encoded RSA-PSS signature over *sha256_hex* string."""
    priv = _load_private(private_key_path, passphrase=passphrase)
    try:
        # Try RSA first
        signature = priv.sign(
            sha256_hex.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256(),
        )
    except Exception:
        # Fallback for ed25519 private keys
        ed = ed25519.Ed25519PrivateKey.from_private_bytes(
            priv.private_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PrivateFormat.Raw,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )
        signature = ed.sign(sha256_hex.encode())
    return base64.b64encode(signature).decode()


def verify_sha256_hex(
    public_key_path: Path, sha256_hex: str, signature_b64: str
) -> bool:
    """Verify a Base64 signature.  Returns **True** if valid, **False** otherwise."""
    pub = _load_public(public_key_path)
    try:
        # Try RSA-PSS first
        pub.verify(
            base64.b64decode(signature_b64),
            sha256_hex.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256(),
        )
        return True
    except Exception:
        # Try Ed25519
        try:
            ed = ed25519.Ed25519PublicKey.from_public_bytes(
                pub.public_bytes(
                    encoding=serialization.Encoding.Raw,
                    format=serialization.PublicFormat.Raw,
                )
            )
            ed.verify(base64.b64decode(signature_b64), sha256_hex.encode())
            return True
        except Exception:
            return False
