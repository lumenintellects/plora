from __future__ import annotations

import json
from pathlib import Path

from plora.signer import generate_keypair, sign_policy, verify_policy


def test_policy_sign_and_verify(tmp_path: Path):
    priv = tmp_path / "priv.pem"
    pub = tmp_path / "pub.pem"
    generate_keypair(priv, pub)

    pol = {"base_model": "dummy/base", "allowed_ranks": [4, 8, 16]}
    pfile = tmp_path / "policy.json"
    pfile.write_text(json.dumps(pol, sort_keys=True))

    sig = sign_policy(pfile, priv)
    assert verify_policy(pfile, pub, sig)

    # modify policy; signature should fail
    pfile.write_text(json.dumps({"x": 1}))
    assert not verify_policy(pfile, pub, sig)
