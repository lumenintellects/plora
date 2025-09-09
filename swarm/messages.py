"""Wire message schema for Swarm Sim v1.

All communication is newline-delimited JSON (NDJSON).  Each call to
``json.dumps(message).encode() + b"\n"`` produces one frame.

The schema follows the spec in the design brief.  We purposely avoid pydantic
here to keep the dependency surface minimal.  Validation is explicit and
lightweight.
"""

from __future__ import annotations

import base64
import json
import math
from dataclasses import dataclass, field, asdict
from hashlib import sha256
from typing import Any, ClassVar, Mapping, MutableMapping, TypedDict
import zlib

__all__ = [
    "OfferMessage",
    "AckMessage",
    "encode_ndjson",
    "decode_ndjson",
]

_JSON_SEP = "\n".encode()


def _b64encode(data: bytes) -> str:
    return base64.b64encode(data).decode()


def _b64decode(data_b64: str) -> bytes:
    return base64.b64decode(data_b64.encode())


class _OfferDict(TypedDict, total=False):
    type: str
    from_: str  # "from" is keyword in python
    domain: str
    sha256: str
    manifest: Mapping[str, Any]
    size: int
    bytes_b64: str
    encoding: str


@dataclass(slots=True)
class OfferMessage:
    """Offer a plasmid from one agent to another."""

    sender: str  # agent name, e.g. "agent_2"
    domain: str
    sha256: str
    manifest: Mapping[str, Any]
    size: int
    bytes_b64: str | None = None  # full-mode only
    encoding: str | None = None  # e.g. "zlib"

    MSG_TYPE: ClassVar[str] = "offer"

    # ---------------------------------------------------------------------
    # Construction helpers
    # ---------------------------------------------------------------------
    @classmethod
    def from_bytes(
        cls,
        sender: str,
        domain: str,
        payload: bytes,
        manifest: Mapping[str, Any],
        *,
        encoding: str | None = None,
    ) -> "OfferMessage":
        """Create an *in-memory* offer with `payload` bytes (full mode)."""
        raw = payload
        enc = None
        if encoding == "zlib":
            raw = zlib.compress(payload)
            enc = "zlib"
        hash_hex = sha256(payload).hexdigest()
        return cls(
            sender=sender,
            domain=domain,
            sha256=hash_hex,
            manifest=manifest,
            size=len(raw),
            bytes_b64=_b64encode(raw),
            encoding=enc,
        )

    # ------------------------------------------------------------------
    # NDJSON helpers
    # ------------------------------------------------------------------
    def to_dict(self) -> _OfferDict:
        d: _OfferDict = {
            "type": self.MSG_TYPE,
            "from": self.sender,
            "domain": self.domain,
            "sha256": self.sha256,
            "manifest": self.manifest,
            "size": self.size,
        }
        if self.bytes_b64 is not None:
            d["bytes_b64"] = self.bytes_b64
        if self.encoding is not None:
            d["encoding"] = self.encoding
        return d  # type: ignore [return-value]

    def to_json(self) -> bytes:
        return json.dumps(self.to_dict(), separators=(",", ":")).encode() + _JSON_SEP

    # ------------------------------------------------------------------
    # Validation & parsing
    # ------------------------------------------------------------------
    @classmethod
    def parse(cls, raw: str) -> "OfferMessage":
        data = json.loads(raw)
        if data.get("type") != cls.MSG_TYPE:
            raise ValueError("Incorrect message type, expected 'offer'.")
        required = {"from", "domain", "sha256", "manifest", "size"}
        if not required.issubset(data):
            missing = required - data.keys()
            raise ValueError(f"Missing required fields: {missing}")
        return cls(
            sender=data["from"],
            domain=data["domain"],
            sha256=data["sha256"],
            manifest=data["manifest"],
            size=data["size"],
            bytes_b64=data.get("bytes_b64"),
            encoding=data.get("encoding"),
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def decode_bytes(self) -> bytes | None:
        if self.bytes_b64 is None:
            return None
        data = _b64decode(self.bytes_b64)
        if self.encoding == "zlib":
            try:
                return zlib.decompress(data)
            except Exception:
                return data
        return data

    # Simple integrity check (caller can compare to manifest sha as well)
    def verify_hash(self, payload: bytes | None = None) -> bool:
        if payload is None:
            payload = self.decode_bytes() or b""
        return sha256(payload).hexdigest() == self.sha256


class _AckDict(TypedDict):
    type: str
    status: str


@dataclass(slots=True)
class AckMessage:
    """Acknowledgement of an offer."""

    status: str  # accepted | rejected_hash | rejected_safety

    MSG_TYPE: ClassVar[str] = "ack"

    def to_dict(self) -> _AckDict:  # type: ignore [override]
        return {"type": self.MSG_TYPE, "status": self.status}

    def to_json(self) -> bytes:
        return json.dumps(self.to_dict(), separators=(",", ":")).encode() + _JSON_SEP

    @classmethod
    def parse(cls, raw: str) -> "AckMessage":
        data = json.loads(raw)
        if data.get("type") != cls.MSG_TYPE:
            raise ValueError("Incorrect message type, expected 'ack'.")
        if "status" not in data:
            raise ValueError("Missing 'status' field in ack message.")
        return cls(status=data["status"])


# ---------------------------------------------------------------------------
# Stream helpers, encode / decode newline-delimited JSON
# ---------------------------------------------------------------------------


def encode_ndjson(msg: OfferMessage | AckMessage) -> bytes:
    return msg.to_json()


def decode_ndjson(line: bytes) -> OfferMessage | AckMessage:
    raw = line.decode().rstrip("\n")
    # Peek at type
    typ = json.loads(raw).get("type")
    if typ == "offer":
        return OfferMessage.parse(raw)
    if typ == "ack":
        return AckMessage.parse(raw)
    raise ValueError(f"Unknown message type: {typ}")
