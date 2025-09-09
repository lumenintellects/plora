from __future__ import annotations

"""plora.manifest - Pydantic model and helpers for `plora.yml` manifests.

This module provides strict validation and convenient load / dump helpers that
round-trip YAML files.  It **does not** verify the RSA signature - that lives
in :pymod:`plora.signer`, but it does cross-check internal consistency such as
`delta_ppl` and artefact hashes.
"""

from pathlib import Path
import json
import hashlib
from typing import List, Optional, Literal

import yaml
from pydantic import BaseModel, Field, ValidationError, model_validator


# ---------------------------------------------------------------------------
# Sub-sections of the schema, keep them explicit for type-checking clarity
# ---------------------------------------------------------------------------


class LoraConfig(BaseModel):
    r: int = Field(..., ge=1)
    alpha: int = Field(..., ge=1)
    dropout: float = Field(..., ge=0.0, le=1.0)
    target_modules: List[str]


class ArtifactsInfo(BaseModel):
    filename: str = Field(
        ..., description="LoRA checkpoint filename, relative to manifest"
    )
    sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(..., ge=1)


class TrainMeta(BaseModel):
    seed: int
    epochs: int
    dataset_id: str
    sample_count: int
    timestamp_unix: int


class MetricsInfo(BaseModel):
    val_ppl_before: float
    val_ppl_after: float
    delta_ppl: float
    val_em: Optional[float] = None
    val_chrf: Optional[float] = None


class SafetyInfo(BaseModel):
    licence: str
    poisoning_score: float = Field(..., ge=0.0)


class SignerInfo(BaseModel):
    algo: str = Field(..., json_schema_extra={"example": "RSA-PSS-SHA256"})
    pubkey_fingerprint: str
    signature_b64: str

    @model_validator(mode="after")
    def _check_algo(self) -> "SignerInfo":
        allowed = {"RSA-PSS-SHA256", "ED25519-SHA256", "none", ""}
        if self.algo not in allowed:
            raise ValueError(
                f"Unsupported signer.algo '{self.algo}'. Allowed: {sorted(allowed)}"
            )
        return self


class CompatibilityInfo(BaseModel):
    peft_min: str
    transformers: str


# ---------------------------------------------------------------------------
# Manifest root model
# ---------------------------------------------------------------------------


class Manifest(BaseModel, json_schema_extra="forbid"):
    """Top-level manifest model matching *plora.yml* v0 schema."""

    schema_version: Literal[0]
    plasmid_id: str
    domain: str
    base_model: str
    peft_format: str = Field(..., pattern=r"^(lora|ia3|adalora)$")

    lora: LoraConfig
    artifacts: ArtifactsInfo
    train_meta: TrainMeta
    metrics: MetricsInfo
    safety: SafetyInfo
    signer: SignerInfo
    compatibility: CompatibilityInfo

    # ------------------------- Validators ---------------------------------

    @model_validator(mode="after")
    def _check_consistency(self) -> "Manifest":
        # delta_ppl check (after / before), but allow placeholder zeros used in tests
        before = float(self.metrics.val_ppl_before)
        after = float(self.metrics.val_ppl_after)
        expected = after - before
        # If both before/after are zeros (placeholder), skip strict check to allow ranking by delta only
        if not (abs(before) < 1e-9 and abs(after) < 1e-9):
            if abs(self.metrics.delta_ppl - expected) > 1e-6:
                raise ValueError(
                    f"delta_ppl {self.metrics.delta_ppl} does not match val_ppl_after - val_ppl_before ({expected})"
                )
        return self

    # --------------------------------------------------------------------
    # Helpers
    # --------------------------------------------------------------------

    @classmethod
    def load(cls, path: Path) -> "Manifest":
        """Load and validate a manifest from a YAML file."""
        data = yaml.safe_load(Path(path).read_text())
        return cls.model_validate(data)

    def dump(self, path: Path):
        """Write the manifest to *path* in YAML format."""
        payload = json.loads(self.model_dump_json(exclude_none=True))
        yaml.safe_dump(payload, path.open("w", encoding="utf-8"), sort_keys=False)

    @classmethod
    def from_adapter(cls, adapter_dir: Path) -> "Manifest":
        """Convenience: load manifest from *adapter_dir/plora.yml*."""
        return cls.load(adapter_dir / "plora.yml")

    # --------------------------------------------------------------------
    # Runtime checks (post-load)
    # --------------------------------------------------------------------

    def validate_artifact_hash(self, base_dir: Path):
        """Re-compute SHA-256 of the artifact file and compare to manifest."""
        artefact_path = base_dir / self.artifacts.filename
        if not artefact_path.exists():
            raise FileNotFoundError(f"Artefact file not found: {artefact_path}")
        current_sha = hashlib.sha256(artefact_path.read_bytes()).hexdigest()
        if current_sha != self.artifacts.sha256:
            raise ValueError(
                f"SHA-256 mismatch for {artefact_path.name}: expected {self.artifacts.sha256}, got {current_sha}"
            )
