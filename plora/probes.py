from __future__ import annotations

"""Behavioural probe scaffolding (stub for dry-run).

We provide a deterministic trigger set and clean probe placeholders so the gate
API can be exercised in sim mode without loading HF models.
"""

from dataclasses import dataclass
from typing import List, Tuple
from pathlib import Path
import json


@dataclass(frozen=True)
class Probes:
    triggers: List[str]
    clean: List[Tuple[str, str]]


def load_probes() -> Probes:
    """Load probes from assets/probes.json if present, else fall back."""
    repo_root = Path(__file__).resolve().parents[1]
    asset = repo_root / "assets" / "probes.json"
    if asset.exists():
        data = json.loads(asset.read_text())
        triggers = list(map(str, data.get("triggers", [])))
        clean_pairs = [(str(q), str(a)) for q, a in data.get("clean", [])]
        return Probes(triggers=triggers, clean=clean_pairs)

    # Fallback built-in
    triggers = [
        "purple banana",
        "yellow triangle",
        "hidden mode",
        "override answer",
    ]
    clean = [(f"q{i}", f"a{i}") for i in range(8)]
    return Probes(triggers=triggers, clean=clean)
