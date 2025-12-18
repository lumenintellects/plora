from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def verify_chain(path: Path) -> bool:
    if not path.exists():
        return False
    lines = path.read_text().splitlines()
    prev = "0" * 64
    for line in lines:
        # Each record was stored as JSON; the audit appended prev_hash to the record
        try:
            # Verify linkage via prev_hash == sha256(previous_line)
            import json

            rec = json.loads(line)
            if rec.get("prev_hash") != prev:
                return False
        except Exception:
            return False
        prev = hashlib.sha256(line.encode()).hexdigest()
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", type=Path, required=True)
    ns = ap.parse_args()
    ok = verify_chain(ns.audit)
    print("OK" if ok else "BROKEN")


if __name__ == "__main__":
    main()
