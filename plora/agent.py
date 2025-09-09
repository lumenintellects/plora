"""Lightweight Agent abstraction used by Swarm Sim and future network layers.

This module purposefully avoids heavy dependencies (no torch, transformers) so
that importing it in a plain Python process is cheap.  It focuses on copying
and verifying LoRA adapter artefacts described by *plora.yml* manifests.
"""
from __future__ import annotations

import asyncio
import json
import shutil
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Dict, List, Mapping, Set

from plora.manifest import Manifest
from plora.gate import alignment_gate, Policy

__all__ = [
    "AdapterInfo",
    "Agent",
]


@dataclass(slots=True)
class AdapterInfo:
    """Pointer to a LoRA adapter directory and its manifest."""

    path: Path  # path to adapter_model.safetensors (or .bin for dummy)
    manifest: Manifest
    size_bytes: int


class Agent:
    """Minimal stateful agent that can share and accept adapters."""

    _RECV_DIR = "received"
    _STATE_FILE = "state.json"
    # per-instance lock for copy operations
    _copy_lock: asyncio.Lock

    def __init__(
        self,
        agent_id: int,
        domain: str,
        adapter: AdapterInfo,
        *,
        root_dir: Path | None = None,
        security_policy: Policy | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.domain = domain
        self.adapter = adapter
        self.root_dir = root_dir or adapter.path.parent  # where to persist state
        self.knowledge: Set[str] = {domain}

        # simple counters for metrics
        self.accepted: int = 0
        self.accepted_trojan: int = 0
        self.accepted_clean: int = 0
        self.offered: int = 0
        self.rejected_hash: int = 0
        self.rejected_safety: int = 0
        self.rejected_trojan: int = 0
        self.rejected_clean: int = 0
        self.rejection_reasons: Dict[str, int] = {}

        # Adapters accepted from peers (domain -> AdapterInfo)
        self.received: Dict[str, AdapterInfo] = {}
        self.security_policy = security_policy
        self._copy_lock = asyncio.Lock()
        # quick de-dup index by SHA
        self._have_sha: Set[str] = {adapter.manifest.artifacts.sha256}
        # capacity controls
        self.capacity: int | None = None  # set externally; if None no cap
        self._recv_order: List[str] = []  # domains in accept order for eviction

    # ------------------------------------------------------------------
    # Persistence helpers (optional)
    # ------------------------------------------------------------------
    def _state_path(self) -> Path:
        return self.root_dir / self._STATE_FILE

    def save_state(self) -> None:
        payload = {
            "knowledge": sorted(self.knowledge),
            "accepted": self.accepted,
            "accepted_trojan": self.accepted_trojan,
            "accepted_clean": self.accepted_clean,
            "offered": self.offered,
            "rejected_hash": self.rejected_hash,
            "rejected_safety": self.rejected_safety,
            "rejected_trojan": self.rejected_trojan,
            "rejected_clean": self.rejected_clean,
            "rejection_reasons": self.rejection_reasons,
        }
        self._state_path().write_text(json.dumps(payload, indent=2))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def shareable_adapters(self) -> Dict[str, AdapterInfo]:
        """Return mapping domain -> AdapterInfo for all adapters this agent can share."""
        pool: Dict[str, AdapterInfo] = {self.domain: self.adapter}
        pool.update(self.received)
        return pool

    # ------------------------------------------------------------------
    # Offer selection helper (for push–pull protocols)
    # ------------------------------------------------------------------
    def best_offer_for(self, peer: "Agent") -> tuple[str | None, AdapterInfo | None]:
        """Select the best adapter to offer to *peer*.

        Preference order for scoring (descending):
        1) manifest.metrics.val_em if available
        2) manifest.metrics.val_chrf if available
        3) -manifest.metrics.delta_ppl (lower delta_ppl is better)
        Returns (domain, AdapterInfo) or (None, None) if nothing new to offer.
        """
        pool = self.shareable_adapters()
        missing_domains = [d for d in pool.keys() if d not in peer.knowledge]
        if not missing_domains:
            return None, None

        def score(dom: str) -> float:
            m = pool[dom].manifest.metrics
            if m.val_em is not None:
                return float(m.val_em)
            if m.val_chrf is not None:
                return float(m.val_chrf)
            # Prefer greater improvement: delta_ppl negative is good
            return float(-m.delta_ppl)

        best_dom = max(missing_domains, key=score)
        return best_dom, pool[best_dom]

    async def accept(self, adapter: AdapterInfo, domain: str) -> bool:
        """Copy *adapter* locally (if new) and verify manifest SHA.

        Returns True if accepted, False otherwise.
        """
        # Refresh manifest from disk to capture any updates (e.g., trojan marking)
        try:
            on_disk = Manifest.from_adapter(adapter.path.parent)
            adapter.manifest = on_disk
        except Exception:
            # Manifest missing or invalid; treat as hash-level reject to avoid accepting unknown artefacts
            self.rejected_hash += 1
            return False

        # Security gate (policy-only for now) – must pass even if artefact SHA is known
        gate = alignment_gate(adapter.path.parent, adapter.manifest, self.security_policy)
        if not gate.passed:
            self.rejected_safety += 1
            if (adapter.manifest.safety.poisoning_score or 0.0) > 0.0:
                self.rejected_trojan += 1
            else:
                self.rejected_clean += 1
            for r in gate.reasons:
                self.rejection_reasons[r] = self.rejection_reasons.get(r, 0) + 1
            # persist state for analysis
            try:
                self.save_state()
            except Exception:
                pass
            # quarantine manifest and artefact for forensics
            try:
                qdir = self.root_dir / "quarantine" / domain
                qdir.mkdir(parents=True, exist_ok=True)
                src_dir = adapter.path.parent
                for fname in [adapter.manifest.artifacts.filename, "plora.yml"]:
                    src = src_dir / fname
                    dst = qdir / fname
                    if src.exists():
                        shutil.copy(src, dst)
            except Exception:
                pass
            return False

        # De-dup: if we already possess an adapter with the same SHA, skip heavy I/O
        sha_hex = adapter.manifest.artifacts.sha256
        if sha_hex in getattr(self, "_have_sha", set()):
            # If domain unknown, add knowledge and index
            if domain not in self.knowledge:
                self.knowledge.add(domain)
                self.received[domain] = adapter
            return True

        # Already have it
        if domain in self.knowledge:
            return True

        recv_domdir = self.root_dir / self._RECV_DIR / domain
        async with self._copy_lock:
            recv_domdir.mkdir(parents=True, exist_ok=True)
            # copy artefact + plora.yml + config
            src_dir = adapter.path.parent
            for fname in [adapter.manifest.artifacts.filename, "adapter_config.json", "plora.yml"]:
                src = src_dir / fname
                dst = recv_domdir / fname
                if src.exists():
                    await asyncio.to_thread(shutil.copy, src, dst)
        # Update knowledge
        self.knowledge.add(domain)
        self.received[domain] = adapter
        # maintain eviction order and enforce capacity if configured
        if domain in self._recv_order:
            self._recv_order.remove(domain)
        self._recv_order.append(domain)
        if self.capacity is not None and len(self._recv_order) > self.capacity:
            evict_dom = self._recv_order.pop(0)
            if evict_dom in self.received:
                try:
                    # remove on-disk received directory to reclaim space
                    recv_domdir = self.root_dir / self._RECV_DIR / evict_dom
                    if recv_domdir.exists():
                        shutil.rmtree(recv_domdir)
                except Exception:
                    pass
                del self.received[evict_dom]
                self.knowledge.discard(evict_dom)
        self.accepted += 1
        if (adapter.manifest.safety.poisoning_score or 0.0) > 0.0:
            self.accepted_trojan += 1
        else:
            self.accepted_clean += 1
        # track possessed SHAs
        self._have_sha.add(sha_hex)
        return True
