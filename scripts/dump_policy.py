from __future__ import annotations

"""Dump effective security policy (from file + CLI overrides) as JSON."""

import argparse
import json
from pathlib import Path

from plora.gate import Policy
from plora.config import get as cfg
from plora.targets import ATTENTION_SUFFIXES


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Dump effective security policy")
    p.add_argument(
        "--policy_file",
        type=Path,
        default=None,
        help="Optional JSON policy file to load",
    )
    p.add_argument("--base_model", type=str, default=None)
    p.add_argument(
        "--allowed_targets",
        type=str,
        default=None,
        help="attention|all or omit to keep from file",
    )
    p.add_argument("--allowed_targets_file", type=Path, default=None)
    p.add_argument(
        "--allowed_ranks",
        type=str,
        default=None,
        help="Comma-separated ranks e.g. 4,8,16",
    )
    p.add_argument("--signatures", choices=["on", "off"], default=None)
    p.add_argument(
        "--trusted_pubkeys", type=str, default=None, help="Comma-separated PEM paths"
    )
    p.add_argument("--tau_trigger", type=float, default=None)
    p.add_argument("--tau_norm_z", type=float, default=None)
    p.add_argument("--tau_clean_delta", type=float, default=None)
    return p


def main() -> None:
    ns = build_arg_parser().parse_args()

    if ns.policy_file is not None and ns.policy_file.exists():
        pol = Policy.from_file(ns.policy_file)
    else:
        pol = Policy()

    if ns.base_model:
        pol.base_model = ns.base_model
    elif pol.base_model is None:
        pol.base_model = cfg("base_model", pol.base_model)

    # Resolve allowed targets
    targets = None
    if ns.allowed_targets_file is not None and ns.allowed_targets_file.exists():
        targets = [
            line.strip()
            for line in ns.allowed_targets_file.read_text().splitlines()
            if line.strip()
        ]
    elif ns.allowed_targets == "attention":
        targets = ATTENTION_SUFFIXES
    elif ns.allowed_targets is None:
        # use config default if provided
        tcfg = cfg("allowed_targets")
        if tcfg == "attention":
            targets = ATTENTION_SUFFIXES
        elif isinstance(tcfg, list):
            targets = tcfg
    elif ns.allowed_targets == "all":
        targets = None  # None means no restriction
    if targets is not None:
        pol.allowed_targets = targets

    if ns.allowed_ranks:
        pol.allowed_ranks = tuple(int(x) for x in ns.allowed_ranks.split(",") if x)
    elif pol.allowed_ranks is None:
        r = cfg("allowed_ranks")
        if isinstance(r, list) and r:
            pol.allowed_ranks = tuple(int(x) for x in r)

    if ns.signatures is not None:
        pol.signatures_enabled = ns.signatures == "on"

    if ns.trusted_pubkeys:
        pol.trusted_public_keys = [Path(p) for p in ns.trusted_pubkeys.split(",") if p]

    if ns.tau_trigger is not None:
        pol.tau_trigger = ns.tau_trigger
    if ns.tau_norm_z is not None:
        pol.tau_norm_z = ns.tau_norm_z
    if ns.tau_clean_delta is not None:
        pol.tau_clean_delta = ns.tau_clean_delta

    # Print JSON
    data = {
        "base_model": pol.base_model,
        "allowed_ranks": list(pol.allowed_ranks)
        if pol.allowed_ranks is not None
        else None,
        "allowed_targets": list(pol.allowed_targets)
        if pol.allowed_targets is not None
        else None,
        "max_size_bytes": pol.max_size_bytes,
        "signatures_enabled": pol.signatures_enabled,
        "trusted_public_keys": [str(p) for p in pol.trusted_public_keys]
        if pol.trusted_public_keys
        else None,
        "tau_trigger": pol.tau_trigger,
        "tau_norm_z": pol.tau_norm_z,
        "tau_clean_delta": pol.tau_clean_delta,
    }
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
