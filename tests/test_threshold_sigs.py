from __future__ import annotations

import hashlib
from pathlib import Path

from plora.signer import generate_keypair, sign_sha256_hex
from plora.threshold_sigs import aggregate_signatures, verify_aggregate


def test_threshold_multisig_quorum(tmp_path: Path):
    payload = b"abc"
    sha = hashlib.sha256(payload).hexdigest()
    pubs = []
    sigs = []
    for i in range(3):
        priv = tmp_path / f"priv{i}.pem"
        pub = tmp_path / f"pub{i}.pem"
        generate_keypair(priv, pub)
        pubs.append(pub)
        sigs.append(sign_sha256_hex(priv, sha))
    agg = aggregate_signatures(sigs[:2])
    assert verify_aggregate(agg, sha, pubs, quorum=2)
    assert not verify_aggregate(agg, sha, pubs, quorum=3)
