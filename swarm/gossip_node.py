"""Asyncio gossip node for Swarm Sim v1.

Each node owns an ``Agent`` (from experiments.plasmid_swarm) and exposes a tiny
line-delimited JSON socket server on localhost.  The design favours clarity
and testability over raw throughput - we expect <10 agents and small payloads.

"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from pathlib import Path
import tempfile
from typing import MutableMapping, Sequence, Set
import ssl

from swarm.messages import AckMessage, OfferMessage, decode_ndjson, encode_ndjson

# Core abstractions
from plora.agent import AdapterInfo
from plora.manifest import Manifest

logger = logging.getLogger(__name__)

_DEFAULT_HOST = "127.0.0.1"
_BASE_PORT = 8500


class GossipNode:
    """A single node that wraps an ``Agent`` and participates in gossip rounds."""

    def __init__(
        self,
        agent,  # the concrete Agent instance
        agent_id: int,
        neighbours: Sequence[int],
        *,
        mode: str = "sim",  # "sim" | "full"
        host: str = _DEFAULT_HOST,
        port: int | None = None,
        rand: random.Random | None = None,
        server_ssl: ssl.SSLContext | None = None,
        client_ssl: ssl.SSLContext | None = None,
    ) -> None:
        self.agent = agent
        self.agent_id = agent_id
        self.neighbours = list(neighbours)
        self.mode = mode
        self.host = host
        self.port = port or (_BASE_PORT + agent_id)
        self.addr = (self.host, self.port)
        self._srv: asyncio.base_events.Server | None = None
        self._rand = rand or random.Random()
        self._server_ssl = server_ssl
        self._client_ssl = client_ssl
        # peer_id -> set[domain] believed present at peer
        self.peer_cache: MutableMapping[int, Set[str]] = {n: set() for n in neighbours}
        # bytes sent & accepted stats (for metrics)
        self.bytes_sent: int = 0
        self.accepted_offers: int = 0

    # ------------------------------------------------------------------
    # Networking
    # ------------------------------------------------------------------
    async def start(self) -> None:
        self._srv = await asyncio.start_server(
            self._handle_client, self.host, self.port, ssl=self._server_ssl
        )
        addr = ",".join(str(s.getsockname()) for s in self._srv.sockets)  # type: ignore[arg-type]
        logger.debug("Agent %s listening on %s", self.agent_id, addr)

    async def close(self) -> None:
        if self._srv is not None:
            self._srv.close()
            await self._srv.wait_closed()

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            line = await asyncio.wait_for(reader.readline(), timeout=0.5)
            msg = decode_ndjson(line)
            if not isinstance(msg, OfferMessage):
                raise ValueError("Expected OfferMessage.")
            status = await self._handle_offer(msg)
            ack = AckMessage(status=status)
            writer.write(encode_ndjson(ack))
            await writer.drain()
        except Exception as exc:
            logger.debug("Agent %s client-handler error: %s", self.agent_id, exc)
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Offer handling
    # ------------------------------------------------------------------
    async def _handle_offer(self, msg: OfferMessage) -> str:
        """Accept/reject incoming offer and update state."""
        # Verify SHA if payload present
        if msg.bytes_b64 is not None and not msg.verify_hash():
            return "rejected_hash"

        dom = msg.domain

        # Short-circuit if already known
        if dom in self.agent.knowledge:
            return "accepted"

        try:
            if self.mode == "sim":
                # Sim-only staging – fabricate minimal on-disk structure
                adapter_info = await asyncio.to_thread(self._stage_dummy_adapter, msg)
            else:
                # Full mode – stage real payload to temp dir and validate via manifest
                adapter_info = await asyncio.to_thread(self._stage_full_adapter, msg)
            ok = await self.agent.accept(adapter_info, dom)
            if ok:
                self.accepted_offers += 1
                return "accepted"
            return "rejected_safety"
        except Exception as exc:
            logger.debug("Agent %s staging failed: %s", self.agent_id, exc)
            return "rejected_safety"

    # ------------------------------------------------------------------
    # Outgoing offer – single peer per round
    # ------------------------------------------------------------------
    async def tick(self, round_id: int) -> None:
        """One gossip tick: pick neighbour, choose offer, send it."""
        if not self.neighbours:
            return
        peer = self._rand.choice(self.neighbours)
        pool = self.agent.shareable_adapters()
        # choose domains peer likely lacks
        candidates = [d for d in pool.keys() if d not in self.peer_cache[peer]]
        if not candidates:
            candidates = list(pool.keys())
        dom = self._rand.choice(candidates)
        offer = self._build_offer(dom, pool[dom])
        try:
            status = await self._send_offer(peer, offer)
            if status == "accepted":
                self.peer_cache[peer].add(dom)
            else:
                # pessimistic update: if rejected due to safety/duplication,
                # still promote diffusion by allowing re-offer next rounds
                self.peer_cache[peer].discard(dom)
        except Exception as exc:
            logger.debug(
                "Agent %s failed to send offer to %s: %s", self.agent_id, peer, exc
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    async def _send_offer(self, peer_id: int, offer: OfferMessage) -> str | None:
        peer_port = _BASE_PORT + peer_id
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(self.host, peer_port, ssl=self._client_ssl),
                timeout=0.5,
            )
            writer.write(encode_ndjson(offer))
            await writer.drain()
            ack_line = await asyncio.wait_for(reader.readline(), timeout=0.5)
            ack = decode_ndjson(ack_line)
            if isinstance(ack, AckMessage):
                if ack.status == "accepted":
                    self.bytes_sent += offer.size
                return ack.status
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
        return None

    def _build_offer(self, dom: str, adapter) -> OfferMessage:
        # For sim-only we do not send real bytes; we send 16-B token.
        if self.mode == "sim":
            token = f"token-{dom}-{self.agent_id}".encode()
            man = {
                "domain": dom,
                "sha256": OfferMessage.from_bytes.__name__,  # placeholder
            }
            return OfferMessage.from_bytes(
                sender=f"agent_{self.agent_id}",
                domain=dom,
                payload=token,
                manifest=man,
            )
        else:
            # full mode: pack real bytes from adapter.path and attach manifest
            src_dir = adapter.path.parent
            artefact = src_dir / adapter.manifest.artifacts.filename
            man_dict = json.loads(adapter.manifest.model_dump_json())
            payload = artefact.read_bytes()
            # compress with zlib to reduce on-wire size
            return OfferMessage.from_bytes(
                sender=f"agent_{self.agent_id}",
                domain=dom,
                payload=payload,
                manifest=man_dict,
                encoding="zlib",
            )

    # ------------------------------------------------------------------
    # Sim-only helper
    # ------------------------------------------------------------------
    def _stage_dummy_adapter(self, msg: OfferMessage):
        """Create a temporary directory with minimal adapter files.

        Layout mimics real LoRA checkpoints so that Agent.accept() can copy
        manifest and model without modification.
        """
        tmp_root = Path(tempfile.mkdtemp(prefix="swarm_sim_"))
        # adapter dir holds the model file(s)
        adapter_dir = tmp_root / msg.domain
        adapter_dir.mkdir(parents=True, exist_ok=True)

        # Write tiny model file (.safetensors expected by accept())
        model_path = adapter_dir / "adapter_model.safetensors"
        model_bytes = msg.decode_bytes() or b"dummy"
        model_path.write_bytes(model_bytes)

        # Minimal adapter_config.json to satisfy copy list (can be empty dict)
        (adapter_dir / "adapter_config.json").write_text("{}")

        # Build minimal plora.yml manifest
        manifest = Manifest(
            schema_version=0,
            plasmid_id=f"dummy-{msg.sha256[:8]}",
            domain=msg.domain,
            base_model="dummy/base",
            peft_format="lora",
            lora={"r": 1, "alpha": 1, "dropout": 0.0, "target_modules": []},
            artifacts={
                "filename": model_path.name,
                "sha256": msg.sha256,
                "size_bytes": len(model_bytes),
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
            signer={
                "algo": "none",
                "pubkey_fingerprint": "none",
                "signature_b64": "",
            },
            compatibility={"peft_min": "0", "transformers": "0"},
        )

        # write YAML manifest
        manifest.dump(adapter_dir / "plora.yml")

        return AdapterInfo(model_path, manifest, len(model_bytes))

    def _stage_full_adapter(self, msg: OfferMessage):
        """Write offered bytes and manifest to a temporary directory for accept()."""
        tmp_root = Path(tempfile.mkdtemp(prefix="swarm_full_"))
        adapter_dir = tmp_root / msg.domain
        adapter_dir.mkdir(parents=True, exist_ok=True)

        # Write model payload
        model_path = adapter_dir / "adapter_model.safetensors"
        payload = msg.decode_bytes() or b""
        model_path.write_bytes(payload)

        # Copy manifest dict to YAML via Manifest model
        manifest = Manifest.model_validate(msg.manifest)
        manifest.dump(adapter_dir / "plora.yml")

        # Ensure adapter_config exists (if missing, write empty)
        (adapter_dir / "adapter_config.json").write_text("{}")

        return AdapterInfo(model_path, manifest, len(payload))
