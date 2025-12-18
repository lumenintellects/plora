from __future__ import annotations

"""Threshold-style signature helpers.

This module provides a simple aggregation/verification utility that treats an
aggregate as a set of individual signatures over the same payload hash and
verifies that at least *quorum* of the provided public keys validate.

For a production-grade aggregate signature (e.g., BLS12-381), swap verification
with a pairing-based check. Here we count distinct verifying keys.
"""

from typing import Iterable, List
import json
from pathlib import Path

from .signer import verify_with_tag, verify_sha256_hex, ADAPTER_TAG


def aggregate_signatures(signatures_b64: Iterable[str]) -> str:
    """Aggregate signatures by encoding them as a JSON list string."""
    return json.dumps(list(signatures_b64))


def verify_aggregate(
    aggregate: str,
    sha256_hex: str,
    public_keys: Iterable[Path],
    quorum: int,
    *,
    allow_untagged: bool = True,
) -> bool:
    """Verify JSON-aggregated signatures against a quorum of public keys.

    Strategy:
    - Parse JSON list of base64 signatures
    - For each signature, try domain-separated tag verification first
      (ADAPTER_TAG || sha256_hex). If that fails and `allow_untagged` is True,
      try raw SHA-256 hex verification for backward compatibility.
    - Count distinct public keys that validate at least one signature.

    Returns True iff count >= quorum.
    """
    if quorum <= 0:
        return True
    try:
        sigs: List[str] = json.loads(aggregate)
    except Exception:
        return False
    ok = 0
    used: set[str] = set()
    keys = list(public_keys)
    for sig in sigs:
        for pub in keys:
            key_id = str(pub)
            if key_id in used:
                continue
            try:
                if verify_with_tag(pub, sha256_hex, sig, ADAPTER_TAG):
                    used.add(key_id)
                    ok += 1
                    break
            except Exception:
                pass
            if allow_untagged:
                try:
                    if verify_sha256_hex(pub, sha256_hex, sig):
                        used.add(key_id)
                        ok += 1
                        break
                except Exception:
                    pass
        if ok >= quorum:
            return True
    return ok >= quorum
