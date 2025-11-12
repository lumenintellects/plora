from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    from arize.api import Client
    from arize.types import Environments, ModelTypes, Schema
except ImportError:  # pragma: no cover
    Client = None  # type: ignore
    Environments = None  # type: ignore
    ModelTypes = None  # type: ignore
    Schema = None  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class ArizeConfig:
    enabled: bool
    space_key: Optional[str]
    api_key: Optional[str]
    model_id: Optional[str]
    model_version: Optional[str]
    environment: str = "validation"
    queue_size: int = 1024
    batch_size: int = 25


class ArizeLoggingService:
    """
    Asynchronous logger that ships inference events to Arize with backpressure.
    """

    def __init__(self, cfg: ArizeConfig):
        self.cfg = cfg
        self.enabled = (
            cfg.enabled
            and all([cfg.space_key, cfg.api_key, cfg.model_id, cfg.model_version])
            and Client is not None
        )
        self._client: Optional[Client] = (
            Client(space_key=cfg.space_key, api_key=cfg.api_key) if self.enabled else None
        )
        self._environment = None
        if self.enabled and Environments is not None:
            try:
                self._environment = getattr(Environments, cfg.environment.upper())
            except AttributeError:
                logger.warning("Unsupported Arize environment '%s', defaulting to VALIDATION.", cfg.environment)
                self._environment = Environments.VALIDATION

        self._queue: asyncio.Queue[Optional[Dict[str, Any]]] = asyncio.Queue(
            maxsize=cfg.queue_size
        )
        self._worker_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        self._schema = None

        if self.enabled and Schema is not None:
            schema_kwargs = dict(
                prediction_id_column_name="record_id",
                prompt_column_name="prompt",
                response_column_name="response",
                timestamp_column_name="timestamp",
                prompt_metadata_columns=[
                    "apollo_filename",
                    "session_uuid",
                    "ad_id",
                ],
                response_metadata_columns=[
                    "model_version",
                    "analysis_type",
                    "image_type_pred",
                    "watermark_prediction",
                    "cache_hit",
                ],
            )
            self._schema = Schema(**schema_kwargs)

    async def start(self) -> None:
        if not self.enabled or self._worker_task:
            return
        self._worker_task = asyncio.create_task(self._worker(), name="arize-logger-worker")
        logger.info("Arize logging service started.")

    async def close(self) -> None:
        if not self.enabled:
            return
        await self._queue.put(None)
        await self._shutdown_event.wait()
        logger.info("Arize logging service stopped.")

    async def log_prediction(self, event: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            logger.warning("Arize logging queue full; dropping event for %s", event.get("apollo_filename"))

    async def _worker(self) -> None:
        batch: List[Dict[str, Any]] = []

        try:
            while True:
                item = await self._queue.get()
                if item is None:
                    if batch:
                        await self._flush(batch)
                    break
                batch.append(item)
                if len(batch) >= self.cfg.batch_size:
                    await self._flush(batch)
                    batch = []
        except asyncio.CancelledError:  # pragma: no cover - best effort shutdown
            if batch:
                await self._flush(batch)
            raise
        finally:
            self._shutdown_event.set()

    async def _flush(self, batch: List[Dict[str, Any]]) -> None:
        if not batch or not self.enabled or not self._client or not ModelTypes or not self._schema:
            return

        payload = []
        for event in batch:
            event_copy = event.copy()
            event_copy.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
            event_copy.setdefault("model_version", self.cfg.model_version)
            event_copy["prompt"] = (
                json.dumps(event_copy["prompt"]) if isinstance(event_copy.get("prompt"), dict) else event_copy.get("prompt")
            )
            event_copy["response"] = (
                json.dumps(event_copy["response"])
                if isinstance(event_copy.get("response"), dict)
                else event_copy.get("response")
            )
            payload.append(event_copy)

        try:
            import pandas as pd  # type: ignore
        except ImportError:
            logger.error("pandas is required for Arize logging but is not installed.")
            return

        df = pd.DataFrame(payload)
        try:
            response = await asyncio.to_thread(
                self._client.log,
                dataframe=df,
                model_id=self.cfg.model_id,
                model_version=self.cfg.model_version,
                model_type=ModelTypes.GENERATIVE,
                environment=self._environment,
                schema=self._schema,
            )
            if response.status_code != 200:
                logger.error(
                    "Arize logging failed with status %s: %s",
                    response.status_code,
                    response.text,
                )
        except Exception as exc:
            logger.exception("Failed to log batch to Arize: %s", exc)


