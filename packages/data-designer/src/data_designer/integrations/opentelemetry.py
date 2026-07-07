# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import atexit
import contextlib
import logging
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from enum import Enum
from typing import Any

from data_designer.config.version import get_library_version
from data_designer.engine.observability import (
    RequestAdmissionEvent,
    RuntimeCorrelation,
    SchedulerAdmissionEvent,
    SchedulerAdmissionEventKind,
    runtime_correlation_provider,
)

logger = logging.getLogger(__name__)

_HOST = "127.0.0.1"
_SCHEDULER_EVENT_NAMES: dict[SchedulerAdmissionEventKind, str | None] = {
    "scheduler_job_completed": "job.completed",
    "row_group_checkpointed": None,
    "non_retryable_dropped": "task.error",
    "worker_spawn_failed": "worker.spawn_failed",
    "retry_deferred": "task.retry_deferred",
    "cancelled": "task.cancelled",
}
_OPERATION_NAMES = {
    "chat": "chat",
    "embedding": "embeddings",
    "image": "generate_content",
    "healthcheck": "healthcheck",
}
_GEN_AI_DURATION_BUCKETS = (
    0.01,
    0.02,
    0.04,
    0.08,
    0.16,
    0.32,
    0.64,
    1.28,
    2.56,
    5.12,
    10.24,
    20.48,
    40.96,
    81.92,
)


class _MetricLogHandler(logging.Handler):
    def __init__(self, runtime: OpenTelemetryRuntime) -> None:
        super().__init__()
        self._runtime = runtime

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._runtime.record_log(record.levelno)
        except Exception:
            pass


class OpenTelemetryRuntime:
    """Process-owned Prometheus exporter and adapter for runtime events."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._active_run_ids: set[str] = set()
        self._initialized = False
        self._port: int | None = None
        self._registry: Any = None
        self._meter_provider: Any = None
        self._server: Any = None
        self._server_thread: threading.Thread | None = None
        self._start_http_server: Callable[..., Any] | None = None
        self._create_duration: Any = None
        self._dataset_records: Any = None
        self._scheduler_events: Any = None
        self._request_duration: Any = None
        self._log_records: Any = None
        self._log_handler: logging.Handler | None = None

    @property
    def metrics_url(self) -> str | None:
        with self._lock:
            return f"http://{_HOST}:{self._port}/metrics" if self._server is not None else None

    @contextlib.contextmanager
    def observe_create(self, port: int | None) -> Iterator[None]:
        run_id = f"run-{uuid.uuid4().hex}"
        token = runtime_correlation_provider.set(
            RuntimeCorrelation(
                run_id=run_id,
                row_group=None,
                task_column=None,
                task_type=None,
                scheduling_group_kind=None,
                scheduling_group_identity_hash=None,
                task_execution_id=None,
            )
        )
        started_at = time.perf_counter()
        enabled = False
        try:
            if port is not None:
                enabled = self._activate_run(run_id, port)
            try:
                yield
            except BaseException as exc:
                if enabled:
                    self._record_create_duration(time.perf_counter() - started_at, type(exc).__name__)
                raise
            else:
                if enabled:
                    self._record_create_duration(time.perf_counter() - started_at)
        finally:
            if enabled:
                with self._lock:
                    self._active_run_ids.discard(run_id)
            runtime_correlation_provider.reset(token)

    def accepts_scheduler_event(self, event_kind: SchedulerAdmissionEventKind) -> bool:
        with self._lock:
            return bool(self._active_run_ids) and event_kind in _SCHEDULER_EVENT_NAMES

    def emit_scheduler_event(self, event: SchedulerAdmissionEvent) -> None:
        if not self._event_is_active(event.captured_correlation):
            return
        if event.event_kind == "row_group_checkpointed":
            diagnostics = event.diagnostics
            self._dataset_records.add(
                _non_negative_int(diagnostics.get("surviving_rows")),
                {"record.result": "generated"},
            )
            self._dataset_records.add(
                _non_negative_int(diagnostics.get("dropped_rows")),
                {"record.result": "dropped"},
            )
            return

        event_name = _SCHEDULER_EVENT_NAMES.get(event.event_kind)
        if event_name is None:
            return
        attributes: dict[str, str] = {"event.name": event_name}
        if event.event_kind == "scheduler_job_completed":
            attributes["outcome"] = _bounded_attribute(event.reason_or_result or "success")
        if event.event_kind in {"scheduler_job_completed", "non_retryable_dropped"}:
            error_type = event.diagnostics.get("error_type")
        elif event.event_kind == "worker_spawn_failed":
            error_type = "worker_spawn_failed"
        else:
            error_type = None
        if error_type is not None:
            attributes["error.type"] = _bounded_attribute(error_type)
        self._scheduler_events.add(1, attributes)

    def emit_request_event(self, event: RequestAdmissionEvent) -> None:
        if event.event_kind != "model_request_completed" or not self._event_is_active(event.captured_correlation):
            return
        duration = event.diagnostics.get("duration_seconds")
        if not isinstance(duration, int | float) or duration < 0:
            return
        resource = event.request_resource_key if isinstance(event.request_resource_key, dict) else {}
        raw_domain = resource.get("domain")
        if isinstance(raw_domain, Enum):
            raw_domain = raw_domain.value
        domain = raw_domain if isinstance(raw_domain, str) else ""
        attributes = {
            "gen_ai.operation.name": _OPERATION_NAMES.get(domain, "_OTHER"),
        }
        provider_name = resource.get("provider_name")
        if provider_name:
            attributes["gen_ai.provider.name"] = _bounded_attribute(provider_name)
        outcome = event.diagnostics.get("outcome")
        if outcome not in (None, "success"):
            attributes["error.type"] = _bounded_attribute(outcome)
        self._request_duration.record(float(duration), attributes)

    def record_log(self, level: int) -> None:
        correlation = runtime_correlation_provider.current()
        if correlation is None:
            return
        with self._lock:
            if correlation.run_id not in self._active_run_ids:
                return
        self._log_records.add(1, {"log.severity": _log_severity(level)})

    def shutdown(self) -> None:
        with self._lock:
            if not self._initialized:
                return
            self._active_run_ids.clear()
            server, thread = self._server, self._server_thread
            provider, handler = self._meter_provider, self._log_handler
            self._server = None
            self._server_thread = None
            self._port = None
            self._meter_provider = None
            self._registry = None
            self._start_http_server = None
            self._create_duration = None
            self._dataset_records = None
            self._scheduler_events = None
            self._request_duration = None
            self._log_records = None
            self._log_handler = None
            self._initialized = False

        if handler is not None:
            logging.getLogger("data_designer").removeHandler(handler)
            handler.close()
        _stop_server(server, thread)
        if provider is not None:
            with contextlib.suppress(Exception):
                provider.shutdown()

    def _activate_run(self, run_id: str, port: int) -> bool:
        try:
            with self._lock:
                if not self._initialized:
                    self._initialize()
                if self._server is None:
                    self._rebind(port)
                elif self._port != port and not self._active_run_ids:
                    self._rebind(port)
                elif self._port != port:
                    logger.warning(
                        "OpenTelemetry metrics already active at %s; reusing it instead of requested port %d.",
                        self.metrics_url,
                        port,
                    )
                if self._server is None:
                    return False
                self._active_run_ids.add(run_id)
                return True
        except Exception:
            logger.warning("OpenTelemetry metrics are unavailable; continuing without them.", exc_info=True)
            return False

    def _initialize(self) -> None:
        from opentelemetry.exporter.prometheus import PrometheusMetricReader
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.view import ExplicitBucketHistogramAggregation, View
        from opentelemetry.sdk.resources import Resource
        from prometheus_client import CollectorRegistry, start_http_server

        provider: Any = None
        handler: logging.Handler | None = None
        try:
            version = get_library_version()
            registry = CollectorRegistry()
            reader = PrometheusMetricReader(registry=registry)
            provider = MeterProvider(
                resource=Resource({"service.name": "data-designer", "service.version": version}),
                metric_readers=[reader],
                shutdown_on_exit=False,
                views=[
                    View(
                        instrument_name="gen_ai.client.operation.duration",
                        aggregation=ExplicitBucketHistogramAggregation(boundaries=_GEN_AI_DURATION_BUCKETS),
                    )
                ],
            )
            meter = provider.get_meter("data_designer", version)
            create_duration = meter.create_histogram(
                "data_designer.create.duration",
                unit="s",
                description="Duration of DataDesigner create operations.",
            )
            dataset_records = meter.create_counter(
                "data_designer.dataset.records",
                unit="{record}",
                description="Dataset records checkpointed by DataDesigner.",
            )
            scheduler_events = meter.create_counter(
                "data_designer.scheduler.events",
                unit="{event}",
                description="Selected terminal and error events from the DataDesigner scheduler.",
            )
            request_duration = meter.create_histogram(
                "gen_ai.client.operation.duration",
                unit="s",
                description="GenAI operation duration.",
            )
            log_records = meter.create_counter(
                "data_designer.log.records",
                unit="{record}",
                description="DataDesigner log records by severity.",
            )
            handler = _MetricLogHandler(self)
            logging.getLogger("data_designer").addHandler(handler)
        except Exception:
            if handler is not None:
                logging.getLogger("data_designer").removeHandler(handler)
                handler.close()
            if provider is not None:
                with contextlib.suppress(Exception):
                    provider.shutdown()
            raise

        self._registry = registry
        self._meter_provider = provider
        self._start_http_server = start_http_server
        self._create_duration = create_duration
        self._dataset_records = dataset_records
        self._scheduler_events = scheduler_events
        self._request_duration = request_duration
        self._log_records = log_records
        self._log_handler = handler
        self._initialized = True

    def _rebind(self, port: int) -> None:
        if self._start_http_server is None:
            return
        server, thread = self._start_http_server(port=port, addr=_HOST, registry=self._registry)
        old_server, old_thread = self._server, self._server_thread
        self._server, self._server_thread, self._port = server, thread, port
        _stop_server(old_server, old_thread)
        logger.info("OpenTelemetry metrics available at %s", self.metrics_url)

    def _event_is_active(self, correlation: object) -> bool:
        run_id = correlation.get("run_id") if isinstance(correlation, Mapping) else None
        with self._lock:
            return isinstance(run_id, str) and run_id in self._active_run_ids

    def _record_create_duration(self, duration: float, error_type: str | None = None) -> None:
        attributes = {"error.type": _bounded_attribute(error_type)} if error_type is not None else None
        try:
            self._create_duration.record(duration, attributes)
        except Exception:
            logger.warning("Failed to record OpenTelemetry create duration.", exc_info=True)


def _stop_server(server: Any, thread: threading.Thread | None) -> None:
    if server is None:
        return
    with contextlib.suppress(Exception):
        server.shutdown()
    with contextlib.suppress(Exception):
        server.server_close()
    if thread is not None:
        thread.join(timeout=5.0)


def _non_negative_int(value: object) -> int:
    return max(0, value) if isinstance(value, int) else 0


def _bounded_attribute(value: object) -> str:
    normalized = "".join(character for character in str(value) if character.isalnum() or character in "._-")
    return normalized[:128] or "_OTHER"


def _log_severity(level: int) -> str:
    if level >= logging.CRITICAL:
        return "fatal"
    if level >= logging.ERROR:
        return "error"
    if level >= logging.WARNING:
        return "warn"
    if level >= logging.INFO:
        return "info"
    return "debug"


_RUNTIME = OpenTelemetryRuntime()
atexit.register(_RUNTIME.shutdown)


def get_open_telemetry_runtime() -> OpenTelemetryRuntime:
    """Return the process-owned DataDesigner OpenTelemetry runtime."""
    return _RUNTIME
