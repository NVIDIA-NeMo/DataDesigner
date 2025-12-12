# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Telemetry handler for NeMo products.

Environment variables:
- NEMO_TELEMETRY_ENABLED: Whether telemetry is enabled.
- NEMO_DEPLOYMENT_TYPE: The deployment type the event came from.
- NEMO_TELEMETRY_ENDPOINT: The endpoint to send the telemetry events to.
"""

from __future__ import annotations

import asyncio
import os
import platform
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, ClassVar

import httpx
from pydantic import BaseModel, Field, field_validator

TELEMETRY_ENABLED = os.getenv("NEMO_TELEMETRY_ENABLED", "true").lower() in ("1", "true", "yes")
CLIENT_ID = "184482118588404"
NEMO_TELEMETRY_VERSION = "nemo-telemetry/1.0"
MAX_RETRIES = 3
NEMO_TELEMETRY_ENDPOINT = os.getenv("NEMO_TELEMETRY_ENDPOINT", "...").lower()
CPU_ARCHITECTURE = platform.uname().machine


class NemoSourceEnum(str, Enum):
    """The NeMo product that created the event."""

    INFERENCE = "inference"
    AUDITOR = "auditor"
    DATADESIGNER = "datadesigner"
    EVALUATOR = "evaluator"
    GUARDRAILS = "guardrails"
    UNDEFINED = "undefined"


class DeploymentTypeEnum(str, Enum):
    """The deployment type the event came from."""

    LIBRARY = "library"
    API = "api"
    UNDEFINED = "undefined"


_deployment_type_raw = os.getenv("NEMO_DEPLOYMENT_TYPE", "library").lower()
try:
    DEPLOYMENT_TYPE = DeploymentTypeEnum(_deployment_type_raw)
except ValueError:
    valid_values = [e.value for e in DeploymentTypeEnum]
    raise ValueError(
        f"Invalid NEMO_DEPLOYMENT_TYPE: {_deployment_type_raw!r}. Must be one of: {valid_values}"
    ) from None


class TaskStatusEnum(str, Enum):
    """The status of the task."""

    SUCCESS = "success"
    FAILURE = "failure"
    UNDEFINED = "undefined"


class TelemetryEvent(BaseModel):
    _event_name: ClassVar[str]  # Subclasses must define this
    _schema_version: ClassVar[str] = "1.2"

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "_event_name" not in cls.__dict__:
            raise TypeError(f"{cls.__name__} must define '_event_name' class variable")


class InferenceEvent(TelemetryEvent):
    _event_name: ClassVar[str] = "inference_event"

    nemo_source: NemoSourceEnum = Field(
        ...,
        alias="nemoSource",
        description="The NeMo product that created the event (i.e. data-designer).",
    )
    task: str = Field(
        ...,
        description="The type of task that was performed that generated the inference event (i.e. preview-job, batch-job).",
    )
    task_status: TaskStatusEnum = Field(
        ...,
        alias="taskStatus",
        description="The status of the task.",
    )
    deployment_type: DeploymentTypeEnum = Field(
        default=DEPLOYMENT_TYPE,
        alias="deploymentType",
        description="The deployment type the event came from.",
    )
    model: str = Field(
        ...,
        description="The name of the model that was used.",
    )
    model_group: str = Field(
        default="undefined",
        alias="modelGroup",
        description="An optional ephemeral group identifier (UUID) that can group multiple inference events together.",
    )
    input_bytes: int = Field(
        default=-1,
        alias="inputBytes",
        description="Number of bytes provided as input to the model. -1 if not available.",
        ge=-9223372036854775808,
        le=9223372036854775807,
    )
    input_tokens: int = Field(
        default=-1,
        alias="inputTokens",
        description="Number of tokens provided as input to the model. -1 if not available.",
        ge=-9223372036854775808,
        le=9223372036854775807,
    )
    output_bytes: int = Field(
        default=-1,
        alias="outputBytes",
        description="Number of bytes returned by the model. -1 if not available.",
        ge=-9223372036854775808,
        le=9223372036854775807,
    )
    output_tokens: int = Field(
        default=-1,
        alias="outputTokens",
        description="Number of tokens returned by the model. -1 if not available.",
        ge=-9223372036854775808,
        le=9223372036854775807,
    )

    @field_validator("model_group")
    @classmethod
    def validate_model_group(cls, v: str) -> str:
        if v == "undefined":
            return v
        try:
            uuid.UUID(v)
            return v
        except ValueError:
            return "undefined"

    model_config = {"populate_by_name": True}


@dataclass
class QueuedEvent:
    event: TelemetryEvent
    timestamp: datetime
    retry_count: int = 0


def _get_iso_timestamp(dt: datetime | None = None) -> str:
    if dt is None:
        dt = datetime.now(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def build_payload(events: list[QueuedEvent], *, source_client_version: str) -> dict[str, Any]:
    """Build the full GFE telemetry payload from a list of queued events."""
    return {
        "browserType": "undefined",  # do not change
        "clientId": CLIENT_ID,
        "clientType": "Native",  # do not change
        "clientVariant": "Release",  # do not change
        "clientVer": source_client_version,
        "cpuArchitecture": CPU_ARCHITECTURE,
        "deviceGdprBehOptIn": "None",  # do not change
        "deviceGdprFuncOptIn": "None",  # do not change
        "deviceGdprTechOptIn": "None",  # do not change
        "deviceId": "undefined",  # do not change
        "deviceMake": "undefined",  # do not change
        "deviceModel": "undefined",  # do not change
        "deviceOS": "undefined",  # do not change
        "deviceOSVersion": "undefined",  # do not change
        "deviceType": "undefined",  # do not change
        "eventProtocol": "1.6",  # do not change
        "eventSchemaVer": events[0].event._schema_version,
        "eventSysVer": NEMO_TELEMETRY_VERSION,
        "externalUserId": "undefined",  # do not change
        "gdprBehOptIn": "None",  # do not change
        "gdprFuncOptIn": "None",  # do not change
        "gdprTechOptIn": "None",  # do not change
        "idpId": "undefined",  # do not change
        "integrationId": "undefined",  # do not change
        "productName": "undefined",  # do not change
        "productVersion": "undefined",  # do not change
        "sentTs": _get_iso_timestamp(),
        "sessionId": "undefined",  # do not change
        "userId": "undefined",  # do not change
        "events": [
            {
                "ts": _get_iso_timestamp(queued.timestamp),
                "parameters": queued.event.model_dump(by_alias=True),
                "name": queued.event._event_name,
            }
            for queued in events
        ],
    }


class TelemetryHandler:
    """
    Handles telemetry event batching, flushing, and retry logic for NeMo products.

    Args:
        flush_interval_seconds (float): The interval in seconds to flush the events.
        max_queue_size (int): The maximum number of events to queue before flushing.
        max_retries (int): The maximum number of times to retry sending an event.
        source_client_version (str): The version of the source client. This should be the version of
            the actual NeMo product that is sending the events, typically the same as the version of
            a PyPi package that a user would install.
    """

    def __init__(
        self,
        flush_interval_seconds: float = 120.0,
        max_queue_size: int = 50,
        max_retries: int = MAX_RETRIES,
        source_client_version: str = "undefined",
    ):
        self._flush_interval = flush_interval_seconds
        self._max_queue_size = max_queue_size
        self._max_retries = max_retries
        self._events: list[QueuedEvent] = []
        self._dlq: list[QueuedEvent] = []  # Dead letter queue for retry
        self._flush_signal = asyncio.Event()
        self._timer_task: asyncio.Task | None = None
        self._running = False
        self._source_client_version = source_client_version

    async def astart(self) -> None:
        if self._running:
            return
        self._running = True
        self._timer_task = asyncio.create_task(self._timer_loop())

    async def astop(self) -> None:
        self._running = False
        self._flush_signal.set()
        if self._timer_task:
            self._timer_task.cancel()
            try:
                await self._timer_task
            except asyncio.CancelledError:
                pass
            self._timer_task = None
        await self._flush_events()

    async def aflush(self) -> None:
        self._flush_signal.set()

    def start(self) -> None:
        self._run_sync(self.astart())

    def stop(self) -> None:
        self._run_sync(self.astop())

    def flush(self) -> None:
        self._flush_signal.set()

    def enqueue(self, event: TelemetryEvent) -> None:
        if not TELEMETRY_ENABLED:
            return
        if not isinstance(event, TelemetryEvent):
            # Silently fail as we prioritize not disrupting upstream call sites and telemetry is best effort
            return
        queued = QueuedEvent(event=event, timestamp=datetime.now(timezone.utc))
        self._events.append(queued)
        if len(self._events) >= self._max_queue_size:
            self._flush_signal.set()

    def _run_sync(self, coro: Any) -> Any:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        else:
            return asyncio.run(coro)

    def __enter__(self) -> TelemetryHandler:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()

    async def __aenter__(self) -> TelemetryHandler:
        await self.astart()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.astop()

    async def _timer_loop(self) -> None:
        while self._running:
            try:
                await asyncio.wait_for(
                    self._flush_signal.wait(),
                    timeout=self._flush_interval,
                )
            except TimeoutError:
                pass
            self._flush_signal.clear()
            await self._flush_events()

    async def _flush_events(self) -> None:
        dlq_events, self._dlq = self._dlq, []
        new_events, self._events = self._events, []
        events_to_send = dlq_events + new_events
        if events_to_send:
            await self._send_events(events_to_send)

    async def _send_events(self, events: list[QueuedEvent]) -> None:
        async with httpx.AsyncClient() as client:
            await self._send_events_with_client(client, events)

    async def _send_events_with_client(self, client: httpx.AsyncClient, events: list[QueuedEvent]) -> None:
        if not events:
            return

        payload = build_payload(events, source_client_version=self._source_client_version)
        try:
            response = await client.post(NEMO_TELEMETRY_ENDPOINT, json=payload)
            try:
                print(response.json())
            except Exception:
                pass
            # 2xx, 400, 422 are all considered complete (no retry)
            # 400/422 indicate bad payload which retrying won't fix
            if response.status_code in (400, 422) or response.is_success:
                return
            # 413 (payload too large) - split and retry
            if response.status_code == 413:
                if len(events) == 1:
                    # Can't split further, drop the event
                    return
                mid = len(events) // 2
                await self._send_events_with_client(client, events[:mid])
                await self._send_events_with_client(client, events[mid:])
                return
            if response.status_code == 408 or response.status_code >= 500:
                self._add_to_dlq(events)
        except httpx.HTTPError:
            self._add_to_dlq(events)

    def _add_to_dlq(self, events: list[QueuedEvent]) -> None:
        for queued in events:
            queued.retry_count += 1
            if queued.retry_count > self._max_retries:
                continue
            self._dlq.append(queued)
