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
    "scheduler_job_started": "job.started",
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
        self._active_dataset_runs: dict[str, tuple[int, int]] = {}
        self._dataset_progress_value: float | None = None
        self._active_request_attempts: dict[tuple[str, str], dict[str, str]] = {}
        self._initialized = False
        self._port: int | None = None
        self._registry: Any = None
        self._meter_provider: Any = None
        self._server: Any = None
        self._server_thread: threading.Thread | None = None
        self._start_http_server: Callable[..., Any] | None = None
        self._create_duration: Any = None
        self._dataset_records: Any = None
        self._dataset_progress: Any = None
        self._scheduler_events: Any = None
        self._active_model_requests: Any = None
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
                    self._finish_active_dataset(run_id)
                    self._finish_active_requests(run_id)
            runtime_correlation_provider.reset(token)

    def accepts_scheduler_event(self, event_kind: SchedulerAdmissionEventKind) -> bool:
        with self._lock:
            return bool(self._active_run_ids) and event_kind in _SCHEDULER_EVENT_NAMES

    def emit_scheduler_event(self, event: SchedulerAdmissionEvent) -> None:
        run_id = self._event_run_id(event.captured_correlation)
        if run_id is None:
            return
        with self._lock:
            if run_id not in self._active_run_ids:
                return
            if event.event_kind == "row_group_checkpointed":
                diagnostics = event.diagnostics
                generated = _non_negative_int(diagnostics.get("surviving_rows"))
                dropped = _non_negative_int(diagnostics.get("dropped_rows"))
                self._dataset_records.add(
                    generated,
                    {"record.result": "generated"},
                )
                self._dataset_records.add(
                    dropped,
                    {"record.result": "dropped"},
                )
                self._advance_dataset_progress(run_id, generated + dropped)
                return
            if event.event_kind == "scheduler_job_started":
                self._start_dataset_progress(run_id, _non_negative_int(event.diagnostics.get("row_group_total_rows")))

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
            if event.event_kind == "scheduler_job_completed":
                self._finish_active_dataset(run_id)

    def emit_request_event(self, event: RequestAdmissionEvent) -> None:
        run_id = self._event_run_id(event.captured_correlation)
        if run_id is None or event.event_kind not in {"model_request_started", "model_request_completed"}:
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
        model_id = resource.get("model_id")
        if model_id:
            attributes["gen_ai.request.model"] = str(model_id)
        attempt_id = event.request_attempt_id or event.request_lease_id
        attempt_key = (run_id, attempt_id) if isinstance(attempt_id, str) and attempt_id else None
        with self._lock:
            if run_id not in self._active_run_ids:
                return
            if event.event_kind == "model_request_started":
                if attempt_key is not None:
                    if attempt_key not in self._active_request_attempts:
                        self._active_model_requests.add(1, attributes)
                        self._active_request_attempts[attempt_key] = dict(attributes)
                return

            if attempt_key is not None:
                active_attributes = self._active_request_attempts.get(attempt_key)
                if active_attributes is not None:
                    self._active_model_requests.add(-1, active_attributes)
                    self._active_request_attempts.pop(attempt_key, None)
            duration = event.diagnostics.get("duration_seconds")
            if not isinstance(duration, int | float) or duration < 0:
                return
            outcome = event.diagnostics.get("outcome")
            if outcome not in (None, "success"):
                attributes = dict(attributes)
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
            self._active_dataset_runs.clear()
            self._dataset_progress_value = None
            self._active_request_attempts.clear()
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
            self._dataset_progress = None
            self._scheduler_events = None
            self._active_model_requests = None
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
        old_server: Any = None
        old_thread: threading.Thread | None = None
        try:
            with self._lock:
                if not self._initialized:
                    self._initialize()
                if self._server is None:
                    old_server, old_thread = self._rebind(port)
                elif self._port != port and not self._active_run_ids:
                    old_server, old_thread = self._rebind(port)
                elif self._port != port:
                    logger.warning(
                        "OpenTelemetry metrics already active at %s; reusing it instead of requested port %d.",
                        self.metrics_url,
                        port,
                    )
                if self._server is None:
                    return False
                self._active_run_ids.add(run_id)
            _stop_server(old_server, old_thread)
            return True
        except Exception:
            logger.warning("OpenTelemetry metrics are unavailable; continuing without them.", exc_info=True)
            return False

    def _initialize(self) -> None:
        from opentelemetry.exporter.prometheus import PrometheusMetricReader
        from opentelemetry.metrics import Observation
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
                description="Dataset records generated or dropped by DataDesigner.",
            )

            def observe_dataset_progress(_options: object) -> list[Observation]:
                with self._lock:
                    value = self._dataset_progress_value
                return [] if value is None else [Observation(value)]

            dataset_progress = meter.create_observable_gauge(
                "data_designer.dataset.progress",
                callbacks=[observe_dataset_progress],
                unit="1",
                description="Fraction of records processed by current DataDesigner create jobs.",
            )
            scheduler_events = meter.create_counter(
                "data_designer.scheduler.events",
                unit="{event}",
                description="Job lifecycle and selected error events from the DataDesigner scheduler.",
            )
            active_model_requests = meter.create_up_down_counter(
                "data_designer.model.request.active",
                unit="{request}",
                description="Model requests currently in progress.",
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
        self._dataset_progress = dataset_progress
        self._scheduler_events = scheduler_events
        self._active_model_requests = active_model_requests
        self._request_duration = request_duration
        self._log_records = log_records
        self._log_handler = handler
        self._initialized = True

    def _rebind(self, port: int) -> tuple[Any, threading.Thread | None]:
        if self._start_http_server is None:
            return None, None
        server, thread = self._start_http_server(port=port, addr=_HOST, registry=self._registry)
        old_server, old_thread = self._server, self._server_thread
        self._server, self._server_thread, self._port = server, thread, port
        logger.info("OpenTelemetry metrics available at %s", self.metrics_url)
        return old_server, old_thread

    @staticmethod
    def _event_run_id(correlation: object) -> str | None:
        run_id = correlation.get("run_id") if isinstance(correlation, Mapping) else None
        return run_id if isinstance(run_id, str) else None

    def _start_dataset_progress(self, run_id: str, scheduled: int) -> None:
        with self._lock:
            if run_id in self._active_dataset_runs:
                return
            self._active_dataset_runs[run_id] = (scheduled, 0)
            self._dataset_progress_value = self._active_dataset_progress()

    def _advance_dataset_progress(self, run_id: str, processed: int) -> None:
        with self._lock:
            state = self._active_dataset_runs.get(run_id)
            if state is None:
                return
            scheduled, current = state
            self._active_dataset_runs[run_id] = (scheduled, min(scheduled, current + processed))
            self._dataset_progress_value = self._active_dataset_progress()

    def _finish_active_dataset(self, run_id: str) -> None:
        with self._lock:
            state = self._active_dataset_runs.pop(run_id, None)
            if state is None:
                return
            if self._active_dataset_runs:
                progress = self._active_dataset_progress()
            else:
                scheduled, processed = state
                progress = processed / scheduled if scheduled else 1.0
            self._dataset_progress_value = progress

    def _active_dataset_progress(self) -> float:
        scheduled = sum(state[0] for state in self._active_dataset_runs.values())
        processed = sum(state[1] for state in self._active_dataset_runs.values())
        return processed / scheduled if scheduled else 1.0

    def _finish_active_requests(self, run_id: str) -> None:
        with self._lock:
            active = [
                (key, attributes) for key, attributes in self._active_request_attempts.items() if key[0] == run_id
            ]
            for key, attributes in active:
                try:
                    self._active_model_requests.add(-1, attributes)
                except Exception:
                    logger.warning("Failed to reconcile an active OpenTelemetry model request.", exc_info=True)
                finally:
                    self._active_request_attempts.pop(key, None)

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
        with contextlib.suppress(Exception):
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
