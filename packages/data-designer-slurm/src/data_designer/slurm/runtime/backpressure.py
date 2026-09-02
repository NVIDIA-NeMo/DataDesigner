# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM middleware that exposes bounded queue admission to client AIMD."""

from __future__ import annotations

import importlib
import json
import math
import os
import threading
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Protocol

MAX_WAITING_REQUESTS_ENVIRONMENT = "DD_VLLM_MAX_WAITING_REQUESTS"
RETRY_AFTER_SECONDS_ENVIRONMENT = "DD_VLLM_RETRY_AFTER_SECONDS"
_METRIC_NAME = "vllm:num_requests_waiting"
_EXEMPT_PATHS = ("/health", "/ready", "/metrics", "/version", "/v1/models", "/ping")

AsgiMessage = dict[str, Any]
AsgiScope = dict[str, Any]
AsgiReceive = Callable[[], Awaitable[AsgiMessage]]
AsgiSend = Callable[[AsgiMessage], Awaitable[None]]
AsgiApp = Callable[[AsgiScope, AsgiReceive, AsgiSend], Awaitable[None]]


class QueueDepthReader(Protocol):
    """Read the latest aggregate waiting-request count."""

    def __call__(self) -> int | None:
        """Return queue depth, or ``None`` when metrics are unavailable."""
        ...


@dataclass(frozen=True, slots=True)
class QueueBackpressureSettings:
    """Validated worker-owned queue admission settings."""

    max_waiting_requests: int
    retry_after_seconds: int | None
    poll_interval_seconds: float = 0.1
    stale_after_seconds: float = 1.0

    def __post_init__(self) -> None:
        if type(self.max_waiting_requests) is not int or self.max_waiting_requests < 0:
            raise ValueError("maximum waiting requests must be non-negative")
        if self.retry_after_seconds is not None and (
            type(self.retry_after_seconds) is not int or self.retry_after_seconds <= 0
        ):
            raise ValueError("retry-after seconds must be positive or absent")
        if self.poll_interval_seconds <= 0 or self.stale_after_seconds <= 0:
            raise ValueError("queue sampler intervals must be positive")

    @classmethod
    def from_environment(cls, environment: Mapping[str, str] | None = None) -> QueueBackpressureSettings:
        """Load the policy transported by the structured runtime step."""
        source = os.environ if environment is None else environment
        maximum = _parse_non_negative_integer(source.get(MAX_WAITING_REQUESTS_ENVIRONMENT), default=128)
        retry_value = source.get(RETRY_AFTER_SECONDS_ENVIRONMENT)
        retry_after = None if retry_value == "" else _parse_positive_integer(retry_value, default=1)
        return cls(maximum, retry_after)


@dataclass(frozen=True, slots=True)
class QueueSnapshot:
    """One sampled queue depth and its monotonic observation time."""

    depth: int | None
    observed_at: float


class QueueBackpressureController:
    """Cache queue metrics away from the request path and decide admission."""

    def __init__(
        self,
        settings: QueueBackpressureSettings | None = None,
        reader: QueueDepthReader | None = None,
        *,
        start_sampler: bool = True,
    ) -> None:
        self.settings = settings or QueueBackpressureSettings.from_environment()
        self._reader = reader or read_vllm_queue_depth
        self._start_sampler = start_sampler
        self._snapshot = QueueSnapshot(None, 0.0)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def sample_once(self) -> QueueSnapshot:
        """Refresh the cached queue depth once."""
        depth = self._reader()
        if depth is not None:
            depth = max(0, depth)
        snapshot = QueueSnapshot(depth, time.monotonic())
        with self._lock:
            self._snapshot = snapshot
        return snapshot

    def should_reject(self) -> tuple[bool, QueueSnapshot]:
        """Return the fail-open admission decision and supporting snapshot."""
        self._ensure_sampler()
        with self._lock:
            snapshot = self._snapshot
        stale = time.monotonic() - snapshot.observed_at > self.settings.stale_after_seconds
        reject = snapshot.depth is not None and not stale and snapshot.depth > self.settings.max_waiting_requests
        return reject, snapshot

    def _ensure_sampler(self) -> None:
        if not self._start_sampler or self._thread is not None:
            return
        with self._lock:
            if self._thread is None:
                self._thread = threading.Thread(target=self._sample_forever, name="dd-vllm-queue-depth", daemon=True)
                self._thread.start()

    def _sample_forever(self) -> None:
        while True:
            self.sample_once()
            time.sleep(self.settings.poll_interval_seconds)


class QueueDepthBackpressureMiddleware:
    """Reject non-health HTTP requests with 429 above the resolved queue threshold."""

    def __init__(self, app: AsgiApp, controller: QueueBackpressureController | None = None) -> None:
        self.app = app
        self.controller = controller or QueueBackpressureController()

    async def __call__(self, scope: AsgiScope, receive: AsgiReceive, send: AsgiSend) -> None:
        """Apply queue admission without changing exempt or accepted requests."""
        path = str(scope.get("path", ""))
        if scope.get("type") != "http" or path in _EXEMPT_PATHS:
            await self.app(scope, receive, send)
            return
        reject, snapshot = self.controller.should_reject()
        if not reject:
            await self.app(scope, receive, send)
            return
        await _send_overload(send, self.controller.settings, snapshot)


def read_vllm_queue_depth() -> int | None:
    """Return aggregate vLLM queue depth from supported metrics registries."""
    for metrics in (_read_vllm_metrics(), _read_prometheus_metrics()):
        values = _collect_metric_values(metrics)
        if values:
            return max(0, int(sum(values)))
    return None


async def _send_overload(
    send: AsgiSend,
    settings: QueueBackpressureSettings,
    snapshot: QueueSnapshot,
) -> None:
    body = json.dumps(
        {
            "error": {
                "message": "serving queue admission threshold exceeded",
                "type": "rate_limit_exceeded",
                "code": HTTPStatus.TOO_MANY_REQUESTS.value,
                "queue_depth": snapshot.depth,
                "max_waiting_requests": settings.max_waiting_requests,
            }
        },
        separators=(",", ":"),
    ).encode()
    headers = [(b"content-type", b"application/json")]
    if settings.retry_after_seconds is not None:
        headers.append((b"retry-after", str(settings.retry_after_seconds).encode()))
    await send({"type": "http.response.start", "status": HTTPStatus.TOO_MANY_REQUESTS.value, "headers": headers})
    await send({"type": "http.response.body", "body": body})


def _read_vllm_metrics() -> Iterable[object]:
    try:
        module = importlib.import_module("vllm.v1.metrics.reader")
        return module.get_metrics_snapshot()
    except (ImportError, AttributeError, RuntimeError):
        return ()


def _read_prometheus_metrics() -> Iterable[object]:
    try:
        module = importlib.import_module("prometheus_client")
        return tuple(sample for family in module.REGISTRY.collect() for sample in getattr(family, "samples", ()))
    except (ImportError, AttributeError, RuntimeError):
        return ()


def _collect_metric_values(metrics: Iterable[object]) -> tuple[float, ...]:
    names = {_METRIC_NAME, _METRIC_NAME.replace(":", "_")}
    values: list[float] = []
    for metric in metrics:
        if str(getattr(metric, "name", "")) not in names:
            continue
        try:
            value = float(getattr(metric, "value"))
            if math.isfinite(value) and value >= 0:
                values.append(value)
        except (AttributeError, TypeError, ValueError):
            continue
    return tuple(values)


def _parse_non_negative_integer(value: str | None, *, default: int) -> int:
    if value is None:
        return default
    if not value.isascii() or not value.isdigit():
        raise ValueError("queue threshold is invalid")
    return int(value)


def _parse_positive_integer(value: str | None, *, default: int) -> int:
    parsed = _parse_non_negative_integer(value, default=default)
    if parsed <= 0:
        raise ValueError("retry-after seconds must be positive")
    return parsed


__all__ = [
    "MAX_WAITING_REQUESTS_ENVIRONMENT",
    "QueueBackpressureController",
    "QueueBackpressureSettings",
    "QueueDepthBackpressureMiddleware",
    "QueueDepthReader",
    "QueueSnapshot",
    "RETRY_AFTER_SECONDS_ENVIRONMENT",
    "read_vllm_queue_depth",
]
