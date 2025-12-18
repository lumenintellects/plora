"""Lightweight Agent abstraction used by Swarm Sim and future network layers.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Dict, List, Set

from plora.manifest import Manifest
from plora.gate import alignment_gate, Policy
from plora.targets import ATTENTION_SUFFIXES
from plora.signer import sign_with_tag, ADAPTER_TAG

__all__ = [
    "AdapterInfo",
    "Agent",
    "load_real_adapter",
    "make_dummy_adapter",
]


@dataclass(slots=True)
class AdapterInfo:
    """Pointer to a LoRA adapter directory and its manifest."""

    path: Path  # path to adapter_model.safetensors (or .bin for dummy)
    manifest: Manifest
    size_bytes: int
    # Optional: source agent id for reputation gating in in-process sims
    source_agent_id: int | None = None


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
        try:
            import asyncio as _asyncio

            try:
                _asyncio.get_running_loop()
            except RuntimeError:
                # No running loop; set one without calling deprecated get_event_loop
                try:
                    _asyncio.set_event_loop(_asyncio.new_event_loop())
                except Exception:
                    pass
        except Exception:
            pass
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
        # simplistic reputation score for peers (0..1). In a real system this
        # would be persisted and updated based on outcomes.
        self.peer_reputation: Dict[int, float] = {}
        # Optional path to this agent's signing private key (PEM) for attestation
        self.signing_key: Path | None = None
        # Optional consensus engine (in-process) for artefact commits
        self.consensus_engine = None

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

        # Security gate (policy + optional reputation) – must pass even if artefact SHA is known
        gate = alignment_gate(
            adapter.path.parent, adapter.manifest, self.security_policy
        )
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

        # Compute SHA-256 hex once for downstream checks
        sha_hex = adapter.manifest.artifacts.sha256

        # Optional consensus gating: require quorum commit before acceptance
        try:
            if self.security_policy and getattr(
                self.security_policy, "consensus_enabled", False
            ):
                if self.consensus_engine is not None:
                    # Vote for plasmid sha in a single-slot model (slot 0 per domain)
                    from swarm.consensus import (
                        Vote,
                    )  # lazy import to avoid heavy deps at import time

                    slot = 0
                    res = self.consensus_engine.vote(Vote(self.agent_id, slot, sha_hex))
                    if (
                        res != sha_hex
                        and self.consensus_engine.committed(slot) != sha_hex
                    ):
                        self.rejection_reasons["consensus_pending"] = (
                            self.rejection_reasons.get("consensus_pending", 0) + 1
                        )
                        return False
        except Exception:
            # Ensure an event loop exists for environments without a default loop (Py 3.13)
            try:
                import asyncio as _asyncio

                if _asyncio.get_event_loop_policy().get_event_loop() is None:
                    _asyncio.set_event_loop(_asyncio.new_event_loop())
            except Exception:
                pass

        # Reputation gating: if configured, require minimum peer reputation
        if self.security_policy and self.security_policy.min_reputation is not None:
            # Callers should set peer reputation before invoking accept.
            peer_id = getattr(adapter, "source_agent_id", None)
            rep = self.peer_reputation.get(peer_id, 1.0 if peer_id is None else 0.0)
            if rep < float(self.security_policy.min_reputation):
                self.rejection_reasons["reputation_low"] = (
                    self.rejection_reasons.get("reputation_low", 0) + 1
                )
                return False

        # De-dup: if we already possess an adapter with the same SHA, skip heavy I/O
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
            for fname in [
                adapter.manifest.artifacts.filename,
                "adapter_config.json",
                "plora.yml",
            ]:
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
        # Write distributed attestation if we can sign
        try:
            if self.signing_key is not None and self.signing_key.exists():
                att = {
                    "signer_id": int(self.agent_id),
                    "domain": domain,
                    "sha256": sha_hex,
                    "signature_b64": sign_with_tag(
                        self.signing_key, sha_hex, ADAPTER_TAG
                    ),
                    "ts_unix": int(__import__("time").time()),
                }
                att_path = adapter.path.parent / "attestations.jsonl"
                # Replay protection: skip writing if a newer attestation from this signer exists
                try:
                    if att_path.exists():
                        for line in att_path.read_text().splitlines():
                            rec = json.loads(line)
                            if int(rec.get("signer_id", -1)) == int(self.agent_id):
                                if int(rec.get("ts_unix", 0)) >= att["ts_unix"]:
                                    raise RuntimeError("stale_attestation")
                except Exception:
                    pass
                att_path.open("a").write(json.dumps(att) + "\n")
        except Exception:
            pass
        return True


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------
def make_dummy_adapter(
    domain: str,
    out_dir: Path,
    *,
    rank: int = 4,
    target_modules: list[str] | None = None,
) -> AdapterInfo:
    """Create a minimal on-disk dummy adapter for a domain and return AdapterInfo.

    Writes adapter_model.safetensors, adapter_config.json and plora.yml into
    out_dir. Useful for in-process simulations and calibrations where a real
    LoRA payload is not required.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "adapter_model.safetensors"
    payload = f"dummy-{domain}-r{rank}".encode()
    model_path.write_bytes(payload)
    modules = target_modules or list(ATTENTION_SUFFIXES)
    adapter_config = {
        "base_model_name_or_path": "dummy/base",
        "bias": "none",
        "lora_alpha": max(1, rank * 2),
        "r": rank,
        "target_modules": modules,
    }
    (out_dir / "adapter_config.json").write_text(json.dumps(adapter_config, indent=2))
    sha_hex = sha256(payload).hexdigest()
    man = Manifest(
        schema_version=0,
        plasmid_id=f"dummy-{domain}",
        domain=domain,
        base_model="dummy/base",
        peft_format="lora",
        lora={
            "r": rank,
            "alpha": max(1, rank * 2),
            "dropout": 0.0,
            "target_modules": modules,
        },
        artifacts={
            "filename": model_path.name,
            "sha256": sha_hex,
            "size_bytes": len(payload),
        },
        train_meta={
            "seed": 0,
            "epochs": 0,
            "dataset_id": "none",
            "sample_count": 0,
            "timestamp_unix": 0,
        },
        metrics={
            "val_ppl_before": 0.0,
            "val_ppl_after": 0.0,
            "delta_ppl": 0.0,
            "val_em": None,
            "val_chrf": None,
        },
        safety={"licence": "CC0", "poisoning_score": 0.0},
        signer={"algo": "none", "pubkey_fingerprint": "none", "signature_b64": ""},
        compatibility={"peft_min": "0", "transformers": "0"},
    )
    man.dump(out_dir / "plora.yml")
    return AdapterInfo(model_path, man, len(payload))


def load_real_adapter(adapter_dir: Path) -> AdapterInfo | None:
    """Load a real trained adapter from disk.

    Looks for adapter_model.safetensors and plora.yml in adapter_dir.
    Returns AdapterInfo if found, None if missing required files.
    """
    adapter_dir = Path(adapter_dir)
    model_path = adapter_dir / "adapter_model.safetensors"
    manifest_path = adapter_dir / "plora.yml"

    if not model_path.exists() or not manifest_path.exists():
        return None

    manifest = Manifest.load(manifest_path)
    size_bytes = model_path.stat().st_size
    return AdapterInfo(model_path, manifest, size_bytes)
