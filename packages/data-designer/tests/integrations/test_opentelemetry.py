# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import logging
import multiprocessing
import re
import socket
import subprocess
import sys
import textwrap
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Event
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.error import URLError
from urllib.request import urlopen

import pytest

import data_designer.integrations.opentelemetry as opentelemetry
from data_designer.config.column_configs import LLMTextColumnConfig
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.models import ModelConfig, ModelProvider
from data_designer.config.run_config import RunConfig
from data_designer.engine.models.clients.adapters.openai_compatible import OpenAICompatibleClient
from data_designer.engine.models.request_admission.resources import RequestDomain, RequestResourceKey
from data_designer.engine.observability import (
    RequestAdmissionEvent,
    RuntimeCorrelation,
    SchedulerAdmissionEvent,
    runtime_correlation_provider,
)
from data_designer.engine.secret_resolver import PlaintextResolver
from data_designer.engine.testing import make_stub_completion_response
from data_designer.integrations.opentelemetry import OpenTelemetryRuntime
from data_designer.interface.data_designer import DataDesigner


@pytest.fixture
def runtime() -> Iterator[OpenTelemetryRuntime]:
    value = OpenTelemetryRuntime()
    yield value
    value.shutdown()


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _scrape(runtime: OpenTelemetryRuntime) -> str:
    assert runtime.metrics_url is not None
    with urlopen(runtime.metrics_url, timeout=2) as response:
        return response.read().decode()


def _correlation() -> RuntimeCorrelation:
    correlation = runtime_correlation_provider.current()
    assert correlation is not None
    return correlation


def _active_request_total(runtime: OpenTelemetryRuntime) -> float:
    return sum(
        float(line.rsplit(" ", maxsplit=1)[1])
        for line in _scrape(runtime).splitlines()
        if line.startswith("data_designer_model_request_active{")
    )


def test_runtime_only_accepts_exported_events_during_active_runs(runtime: OpenTelemetryRuntime) -> None:
    assert not runtime.accepts_scheduler_event("scheduler_job_started")
    assert not runtime.accepts_request_event("model_request_started")

    with runtime.observe_create(_free_port()):
        assert runtime.accepts_scheduler_event("scheduler_job_started")
        assert not runtime.accepts_scheduler_event("selected")
        assert runtime.accepts_request_event("model_request_started")
        assert runtime.accepts_request_event("model_request_completed")
        assert not runtime.accepts_request_event("request_wait_started")

    assert not runtime.accepts_scheduler_event("scheduler_job_started")
    assert not runtime.accepts_request_event("model_request_started")


def test_runtime_exports_bounded_metrics_without_log_bodies(
    runtime: OpenTelemetryRuntime,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    caplog.set_level(logging.ERROR)
    monkeypatch.setenv(
        "OTEL_RESOURCE_ATTRIBUTES",
        "api.token=SECRET-RESOURCE-SENTINEL,customer.id=SECRET-CUSTOMER-SENTINEL",
    )
    port = _free_port()
    with runtime.observe_create(port):
        correlation = _correlation()
        runtime.emit_scheduler_event(
            SchedulerAdmissionEvent.capture(
                "scheduler_job_started",
                sequence=1,
                correlation=correlation,
                diagnostics={"row_group_total_rows": 4},
            )
        )
        runtime.emit_scheduler_event(
            SchedulerAdmissionEvent.capture(
                "row_group_checkpointed",
                sequence=2,
                correlation=correlation,
                diagnostics={"surviving_rows": 3, "dropped_rows": 1},
            )
        )
        runtime.emit_scheduler_event(
            SchedulerAdmissionEvent.capture(
                "scheduler_job_completed",
                sequence=3,
                correlation=correlation,
                reason_or_result="success",
            )
        )
        runtime.emit_request_event(
            RequestAdmissionEvent.capture(
                "model_request_started",
                sequence=1,
                correlation=correlation,
                request_resource_key={
                    "provider_name": "openai",
                    "model_id": "model-1",
                    "domain": "chat",
                },
                request_attempt_id="SECRET-REQUEST-SENTINEL",
            )
        )
        runtime.emit_request_event(
            RequestAdmissionEvent.capture(
                "model_request_completed",
                sequence=2,
                correlation=correlation,
                request_resource_key={
                    "provider_name": "openai",
                    "model_id": "model-1",
                    "domain": "chat",
                },
                request_attempt_id="SECRET-REQUEST-SENTINEL",
                diagnostics={"duration_seconds": 0.25, "outcome": "success"},
            )
        )
        logging.getLogger("data_designer.test").error("SECRET-LOG-SENTINEL")

    metrics = _scrape(runtime)
    target_line = next(line for line in metrics.splitlines() if line.startswith("target_info{"))
    assert set(re.findall(r'(\w+)="', target_line)) == {"service_name", "service_version"}
    assert 'service_name="data-designer"' in metrics
    assert 'otel_scope_name="data_designer"' in metrics
    assert "data_designer_dataset_records_total{" in metrics
    generated_line = next(line for line in metrics.splitlines() if 'record_result="generated"' in line)
    dropped_line = next(line for line in metrics.splitlines() if 'record_result="dropped"' in line)
    assert generated_line.endswith(" 3.0")
    assert dropped_line.endswith(" 1.0")
    progress_line = next(line for line in metrics.splitlines() if line.startswith("data_designer_dataset_progress{"))
    assert progress_line.endswith(" 1.0")
    assert "data_designer_scheduler_events_total{" in metrics
    assert 'event_name="job.started"' in metrics
    assert 'event_name="job.completed"' in metrics
    assert 'outcome="success"' in metrics
    assert "gen_ai_client_operation_duration_seconds_count{" in metrics
    assert 'gen_ai_operation_name="chat"' in metrics
    assert 'gen_ai_provider_name="openai"' in metrics
    assert 'gen_ai_request_model="model-1"' in metrics
    active_request_line = next(
        line for line in metrics.splitlines() if line.startswith("data_designer_model_request_active{")
    )
    assert active_request_line.endswith(" 0.0")
    assert "data_designer_log_records_total{" in metrics
    assert 'log_severity="error"' in metrics
    assert "data_designer_create_duration_seconds_count{" in metrics
    assert metrics.count("gen_ai_client_operation_duration_seconds_count{") == 1
    assert metrics.count("data_designer_scheduler_events_total{") == 2
    assert "SECRET-" not in metrics
    assert sum(record.getMessage() == "SECRET-LOG-SENTINEL" for record in caplog.records) == 1


def test_runtime_exports_current_active_model_requests(runtime: OpenTelemetryRuntime) -> None:
    resource = RequestResourceKey("provider", "model-1", RequestDomain.CHAT)
    with runtime.observe_create(_free_port()):
        correlation = _correlation()
        runtime.emit_request_event(
            RequestAdmissionEvent.capture(
                "model_request_started",
                sequence=1,
                correlation=correlation,
                request_resource_key=resource,
                request_attempt_id="SECRET-REQUEST-SENTINEL",
            )
        )
        active_metrics = _scrape(runtime)
        active_line = next(
            line for line in active_metrics.splitlines() if line.startswith("data_designer_model_request_active{")
        )
        assert active_line.endswith(" 1.0")
        assert 'gen_ai_operation_name="chat"' in active_line
        assert 'gen_ai_provider_name="provider"' in active_line
        assert 'gen_ai_request_model="model-1"' in active_line
        assert "SECRET-" not in active_metrics

        runtime.emit_request_event(
            RequestAdmissionEvent.capture(
                "model_request_completed",
                sequence=2,
                correlation=correlation,
                request_resource_key=resource,
                request_attempt_id="SECRET-REQUEST-SENTINEL",
                diagnostics={"duration_seconds": 0.1, "outcome": "provider_timeout"},
            )
        )

    completed_metrics = _scrape(runtime)
    completed_line = next(
        line for line in completed_metrics.splitlines() if line.startswith("data_designer_model_request_active{")
    )
    assert completed_line.endswith(" 0.0")
    assert "error_type=" not in completed_line
    assert 'error_type="provider_timeout"' in completed_metrics
    assert "SECRET-" not in completed_metrics


def test_runtime_reconciles_duplicate_unmatched_and_orphan_model_requests(runtime: OpenTelemetryRuntime) -> None:
    resource = RequestResourceKey("provider", "model-1", RequestDomain.CHAT)
    port = _free_port()
    with runtime.observe_create(port):
        correlation = _correlation()
        start = RequestAdmissionEvent.capture(
            "model_request_started",
            sequence=1,
            correlation=correlation,
            request_resource_key=resource,
            request_attempt_id="attempt-1",
        )
        runtime.emit_request_event(start)
        assert _active_request_total(runtime) == 1

        runtime.emit_request_event(
            RequestAdmissionEvent.capture(
                "model_request_started",
                sequence=2,
                correlation=correlation,
                request_resource_key=resource,
                request_attempt_id="attempt-1",
            )
        )
        assert _active_request_total(runtime) == 1

        runtime.emit_request_event(
            RequestAdmissionEvent.capture(
                "model_request_completed",
                sequence=3,
                correlation=correlation,
                request_resource_key=resource,
                request_attempt_id="unmatched-attempt",
                diagnostics={"duration_seconds": 0.1, "outcome": "success"},
            )
        )
        assert _active_request_total(runtime) == 1

        runtime.emit_request_event(
            RequestAdmissionEvent.capture(
                "model_request_completed",
                sequence=4,
                correlation=correlation,
                request_resource_key=resource,
                request_attempt_id="attempt-1",
                diagnostics={"duration_seconds": 0.1, "outcome": "local_cancelled"},
            )
        )
        assert _active_request_total(runtime) == 0

        runtime.emit_request_event(
            RequestAdmissionEvent.capture(
                "model_request_completed",
                sequence=5,
                correlation=correlation,
                request_resource_key=resource,
                request_attempt_id="attempt-1",
                diagnostics={"duration_seconds": 0.1, "outcome": "local_cancelled"},
            )
        )
        assert _active_request_total(runtime) == 0

        runtime.emit_request_event(
            RequestAdmissionEvent.capture(
                "model_request_started",
                sequence=6,
                correlation=correlation,
                request_resource_key=resource,
                request_attempt_id="orphan-attempt",
            )
        )
        assert _active_request_total(runtime) == 1

    assert _active_request_total(runtime) == 0


def test_runtime_rejects_request_that_reaches_state_update_after_teardown(runtime: OpenTelemetryRuntime) -> None:
    entered = Event()
    release = Event()

    def pause_delayed_provider(value: object) -> str:
        if value == "delayed-provider":
            entered.set()
            assert release.wait(timeout=5)
        return str(value)

    resource = RequestResourceKey("provider", "model-1", RequestDomain.CHAT)
    with (
        patch(
            "data_designer.integrations.opentelemetry._bounded_identity_attribute",
            side_effect=pause_delayed_provider,
        ),
        ThreadPoolExecutor(max_workers=1) as executor,
    ):
        future = None
        try:
            with runtime.observe_create(_free_port()):
                correlation = _correlation()
                runtime.emit_request_event(
                    RequestAdmissionEvent.capture(
                        "model_request_started",
                        sequence=1,
                        correlation=correlation,
                        request_resource_key=resource,
                        request_attempt_id="baseline",
                    )
                )
                runtime.emit_request_event(
                    RequestAdmissionEvent.capture(
                        "model_request_completed",
                        sequence=2,
                        correlation=correlation,
                        request_resource_key=resource,
                        request_attempt_id="baseline",
                        diagnostics={"duration_seconds": 0.1, "outcome": "success"},
                    )
                )
                future = executor.submit(
                    runtime.emit_request_event,
                    RequestAdmissionEvent.capture(
                        "model_request_started",
                        sequence=3,
                        correlation=correlation,
                        request_resource_key=RequestResourceKey(
                            "delayed-provider",
                            "model-1",
                            RequestDomain.CHAT,
                        ),
                        request_attempt_id="delayed",
                    ),
                )
                assert entered.wait(timeout=5)
        finally:
            release.set()
        assert future is not None
        future.result(timeout=5)

    assert _active_request_total(runtime) == 0


def test_runtime_deactivation_holds_lock_through_state_cleanup(
    runtime: OpenTelemetryRuntime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_finish_dataset = runtime._finish_active_dataset
    original_finish_requests = runtime._finish_active_requests

    def lock_is_available_to_another_thread() -> bool:
        acquired = runtime._lock.acquire(blocking=False)
        if acquired:
            runtime._lock.release()
        return acquired

    with ThreadPoolExecutor(max_workers=1) as executor:

        def finish_dataset(run_id: str) -> None:
            assert executor.submit(lock_is_available_to_another_thread).result(timeout=5) is False
            original_finish_dataset(run_id)

        def finish_requests(run_id: str) -> list[Exception]:
            assert executor.submit(lock_is_available_to_another_thread).result(timeout=5) is False
            return original_finish_requests(run_id)

        monkeypatch.setattr(runtime, "_finish_active_dataset", finish_dataset)
        monkeypatch.setattr(runtime, "_finish_active_requests", finish_requests)
        with runtime.observe_create(_free_port()):
            correlation = _correlation()
            runtime.emit_scheduler_event(
                SchedulerAdmissionEvent.capture(
                    "scheduler_job_started",
                    sequence=1,
                    correlation=correlation,
                    diagnostics={"row_group_total_rows": 2},
                )
            )
            runtime.emit_request_event(
                RequestAdmissionEvent.capture(
                    "model_request_started",
                    sequence=2,
                    correlation=correlation,
                    request_resource_key=RequestResourceKey("provider", "model-1", RequestDomain.CHAT),
                    request_attempt_id="orphan",
                )
            )

    assert _active_request_total(runtime) == 0
    progress_line = next(
        line for line in _scrape(runtime).splitlines() if line.startswith("data_designer_dataset_progress{")
    )
    assert progress_line.endswith(" 0.0")


def test_runtime_progress_resets_for_sequential_runs(runtime: OpenTelemetryRuntime) -> None:
    port = _free_port()
    with runtime.observe_create(port):
        correlation = _correlation()
        runtime.emit_scheduler_event(
            SchedulerAdmissionEvent.capture(
                "scheduler_job_started",
                sequence=1,
                correlation=correlation,
                diagnostics={"row_group_total_rows": 4},
            )
        )
        runtime.emit_scheduler_event(
            SchedulerAdmissionEvent.capture(
                "row_group_checkpointed",
                sequence=2,
                correlation=correlation,
                diagnostics={"surviving_rows": 2, "dropped_rows": 0},
            )
        )
        progress_line = next(
            line for line in _scrape(runtime).splitlines() if line.startswith("data_designer_dataset_progress{")
        )
        assert progress_line.endswith(" 0.5")

    with runtime.observe_create(port):
        correlation = _correlation()
        runtime.emit_scheduler_event(
            SchedulerAdmissionEvent.capture(
                "scheduler_job_started",
                sequence=3,
                correlation=correlation,
                diagnostics={"row_group_total_rows": 8},
            )
        )
        progress_line = next(
            line for line in _scrape(runtime).splitlines() if line.startswith("data_designer_dataset_progress{")
        )
        assert progress_line.endswith(" 0.0")
        runtime.emit_scheduler_event(
            SchedulerAdmissionEvent.capture(
                "row_group_checkpointed",
                sequence=4,
                correlation=correlation,
                diagnostics={"surviving_rows": 1, "dropped_rows": 1},
            )
        )
        progress_line = next(
            line for line in _scrape(runtime).splitlines() if line.startswith("data_designer_dataset_progress{")
        )
        assert progress_line.endswith(" 0.25")


def test_runtime_records_create_error_type_without_error_text(runtime: OpenTelemetryRuntime) -> None:
    with pytest.raises(ValueError, match="SECRET-EXCEPTION-SENTINEL"):
        with runtime.observe_create(_free_port()):
            raise ValueError("SECRET-EXCEPTION-SENTINEL")

    metrics = _scrape(runtime)
    assert 'error_type="ValueError"' in metrics
    assert "SECRET-EXCEPTION-SENTINEL" not in metrics


@pytest.mark.parametrize(
    ("event_kind", "event_name", "error_type"),
    [
        ("non_retryable_dropped", "task.error", "ValueError"),
        ("worker_spawn_failed", "worker.spawn_failed", "worker_spawn_failed"),
        ("retry_deferred", "task.retry_deferred", None),
        ("cancelled", "task.cancelled", None),
    ],
)
def test_runtime_maps_selected_scheduler_events(
    runtime: OpenTelemetryRuntime,
    event_kind: str,
    event_name: str,
    error_type: str | None,
) -> None:
    with runtime.observe_create(_free_port()):
        runtime.emit_scheduler_event(
            SchedulerAdmissionEvent.capture(
                event_kind,  # type: ignore[arg-type]
                sequence=1,
                correlation=_correlation(),
                diagnostics={"error_type": "ValueError"},
            )
        )

    metrics = _scrape(runtime)
    line = next(
        line
        for line in metrics.splitlines()
        if line.startswith("data_designer_scheduler_events_total{") and f'event_name="{event_name}"' in line
    )
    if error_type is None:
        assert "error_type=" not in line
    else:
        assert f'error_type="{error_type}"' in line


@pytest.mark.parametrize(
    ("domain", "operation"),
    [
        (RequestDomain.CHAT, "chat"),
        (RequestDomain.EMBEDDING, "embeddings"),
        (RequestDomain.IMAGE, "generate_content"),
        (RequestDomain.HEALTHCHECK, "healthcheck"),
    ],
)
def test_runtime_maps_genai_operations(
    runtime: OpenTelemetryRuntime,
    domain: RequestDomain,
    operation: str,
) -> None:
    with runtime.observe_create(_free_port()):
        correlation = _correlation()
        resource = RequestResourceKey("provider", "model-1", domain)
        runtime.emit_request_event(
            RequestAdmissionEvent.capture(
                "model_request_started",
                sequence=1,
                correlation=correlation,
                request_resource_key=resource,
            )
        )
        runtime.emit_request_event(
            RequestAdmissionEvent.capture(
                "model_request_completed",
                sequence=2,
                correlation=correlation,
                request_resource_key=resource,
                diagnostics={"duration_seconds": 0.1, "outcome": "provider_timeout"},
            )
        )

    metrics = _scrape(runtime)
    line = next(
        line for line in metrics.splitlines() if line.startswith("gen_ai_client_operation_duration_seconds_count{")
    )
    assert f'gen_ai_operation_name="{operation}"' in line
    assert 'error_type="provider_timeout"' in line
    assert 'gen_ai_request_model="model-1"' in line


def test_runtime_preserves_distinct_bounded_identity_attributes(runtime: OpenTelemetryRuntime) -> None:
    common_prefix = "organization/" + ("m" * 150)
    resources = [
        RequestResourceKey("provider:v1", "llama3:8b", RequestDomain.CHAT),
        RequestResourceKey("providerv1", "llama38b", RequestDomain.CHAT),
        RequestResourceKey("provider", common_prefix + "a", RequestDomain.CHAT),
        RequestResourceKey("provider", common_prefix + "b", RequestDomain.CHAT),
    ]
    with runtime.observe_create(_free_port()):
        correlation = _correlation()
        for index, resource in enumerate(resources):
            attempt_id = f"attempt-{index}"
            runtime.emit_request_event(
                RequestAdmissionEvent.capture(
                    "model_request_started",
                    sequence=index * 2 + 1,
                    correlation=correlation,
                    request_resource_key=resource,
                    request_attempt_id=attempt_id,
                )
            )
            runtime.emit_request_event(
                RequestAdmissionEvent.capture(
                    "model_request_completed",
                    sequence=index * 2 + 2,
                    correlation=correlation,
                    request_resource_key=resource,
                    request_attempt_id=attempt_id,
                    diagnostics={"duration_seconds": 0.1, "outcome": "success"},
                )
            )

    metrics = _scrape(runtime)
    count_lines = [
        line for line in metrics.splitlines() if line.startswith("gen_ai_client_operation_duration_seconds_count{")
    ]
    model_labels = {re.search(r'gen_ai_request_model="([^"]+)"', line).group(1) for line in count_lines}
    provider_labels = {re.search(r'gen_ai_provider_name="([^"]+)"', line).group(1) for line in count_lines}

    assert {"llama3:8b", "llama38b"} <= model_labels
    assert {"provider:v1", "providerv1"} <= provider_labels
    long_labels = model_labels - {"llama3:8b", "llama38b"}
    assert len(long_labels) == 2
    assert all(len(label) == 128 for label in long_labels)
    assert all(re.fullmatch(r"organization/m+-[0-9a-f]{16}", label) for label in long_labels)


def test_runtime_collapses_unknown_genai_operation(runtime: OpenTelemetryRuntime) -> None:
    with runtime.observe_create(_free_port()):
        correlation = _correlation()
        resource = {"provider_name": "provider", "domain": "SECRET-DOMAIN-SENTINEL"}
        runtime.emit_request_event(
            RequestAdmissionEvent.capture(
                "model_request_started",
                sequence=1,
                correlation=correlation,
                request_resource_key=resource,
            )
        )
        runtime.emit_request_event(
            RequestAdmissionEvent.capture(
                "model_request_completed",
                sequence=2,
                correlation=correlation,
                request_resource_key=resource,
                diagnostics={"duration_seconds": 0.1, "outcome": "success"},
            )
        )

    metrics = _scrape(runtime)
    assert 'gen_ai_operation_name="_OTHER"' in metrics
    assert "SECRET-DOMAIN-SENTINEL" not in metrics


def test_runtime_scopes_log_counts_to_active_data_designer_run(runtime: OpenTelemetryRuntime) -> None:
    port = _free_port()
    with runtime.observe_create(port):
        logging.getLogger().error("root")
        logging.getLogger("data_designer.test").warning("enabled")
        with runtime.observe_create(None):
            logging.getLogger("data_designer.test").warning("disabled")
        logging.getLogger("data_designer.test").warning("enabled-again")

    metrics = _scrape(runtime)
    warning_line = next(
        line
        for line in metrics.splitlines()
        if line.startswith("data_designer_log_records_total{") and 'log_severity="warn"' in line
    )
    assert warning_line.endswith(" 2.0")
    assert 'log_severity="error"' not in metrics


def test_runtime_logs_only_after_releasing_runtime_lock(runtime: OpenTelemetryRuntime) -> None:
    lock_available_when_logged = []

    def record_lock_availability(*_args: object, **_kwargs: object) -> None:
        def lock_is_available() -> bool:
            acquired = runtime._lock.acquire(blocking=False)
            if acquired:
                runtime._lock.release()
            return acquired

        with ThreadPoolExecutor(max_workers=1) as executor:
            lock_available_when_logged.append(executor.submit(lock_is_available).result(timeout=5))

    with (
        patch.object(opentelemetry.logger, "info", side_effect=record_lock_availability),
        patch.object(opentelemetry.logger, "warning", side_effect=record_lock_availability),
    ):
        with runtime.observe_create(_free_port()):
            correlation = _correlation()
            runtime.emit_request_event(
                RequestAdmissionEvent.capture(
                    "model_request_started",
                    sequence=1,
                    correlation=correlation,
                    request_resource_key=RequestResourceKey("provider", "model", RequestDomain.CHAT),
                    request_attempt_id="orphan",
                )
            )
            runtime._active_model_requests = MagicMock()
            runtime._active_model_requests.add.side_effect = RuntimeError("reconciliation failed")
            with runtime.observe_create(_free_port()):
                pass

    assert lock_available_when_logged == [True, True, True]


def test_runtime_rebinds_idle_listener_and_preserves_metrics(runtime: OpenTelemetryRuntime) -> None:
    first_port = _free_port()
    second_port = _free_port()
    with runtime.observe_create(first_port):
        runtime.emit_scheduler_event(
            SchedulerAdmissionEvent.capture(
                "row_group_checkpointed",
                sequence=1,
                correlation=_correlation(),
                diagnostics={"surviving_rows": 1, "dropped_rows": 0},
            )
        )
    first_url = runtime.metrics_url

    with runtime.observe_create(second_port):
        runtime.emit_scheduler_event(
            SchedulerAdmissionEvent.capture(
                "row_group_checkpointed",
                sequence=2,
                correlation=_correlation(),
                diagnostics={"surviving_rows": 2, "dropped_rows": 0},
            )
        )

    assert first_url == f"http://127.0.0.1:{first_port}/metrics"
    assert runtime.metrics_url == f"http://127.0.0.1:{second_port}/metrics"
    assert 'record_result="generated"' in _scrape(runtime)
    assert " 3.0" in _scrape(runtime)
    with pytest.raises(URLError):
        urlopen(first_url, timeout=0.2)


def test_runtime_stops_rebound_listener_outside_runtime_lock(runtime: OpenTelemetryRuntime) -> None:
    first_port = _free_port()
    with runtime.observe_create(first_port):
        pass

    lock_was_available = False
    original_stop_server = opentelemetry._stop_server

    def stop_server(server: object, thread: object) -> None:
        nonlocal lock_was_available

        def lock_is_available() -> bool:
            acquired = runtime._lock.acquire(blocking=False)
            if acquired:
                runtime._lock.release()
            return acquired

        with ThreadPoolExecutor(max_workers=1) as executor:
            lock_was_available = executor.submit(lock_is_available).result(timeout=5)
        original_stop_server(server, thread)  # type: ignore[arg-type]

    with patch.object(opentelemetry, "_stop_server", side_effect=stop_server) as stop_server_mock:
        with runtime.observe_create(_free_port()):
            pass

    stop_server_mock.assert_called_once()
    assert lock_was_available is True


def test_runtime_reuses_active_listener_for_conflicting_port(
    runtime: OpenTelemetryRuntime,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING)
    first_port = _free_port()
    with runtime.observe_create(first_port):
        with runtime.observe_create(_free_port()):
            assert runtime.metrics_url == f"http://127.0.0.1:{first_port}/metrics"

    assert "reusing it instead of requested port" in caplog.text


def test_runtime_port_failure_does_not_escape(runtime: OpenTelemetryRuntime) -> None:
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        with runtime.observe_create(int(occupied.getsockname()[1])):
            assert runtime.metrics_url is None


def test_runtime_initialization_failure_cleans_up_partial_provider(
    runtime: OpenTelemetryRuntime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = MagicMock()
    provider.get_meter.return_value.create_histogram.side_effect = RuntimeError("initialization failed")
    monkeypatch.setattr("opentelemetry.sdk.metrics.MeterProvider", lambda **_kwargs: provider)

    with runtime.observe_create(_free_port()):
        assert runtime.metrics_url is None

    provider.shutdown.assert_called_once_with()
    assert runtime._initialized is False
    assert runtime._meter_provider is None
    assert runtime._log_handler is None


def test_metric_log_failure_does_not_echo_record(
    runtime: OpenTelemetryRuntime,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with runtime.observe_create(_free_port()):
        assert runtime._log_handler is not None
        runtime._log_records = MagicMock()
        runtime._log_records.add.side_effect = RuntimeError("recording failed")
        record = logging.LogRecord(
            "data_designer.test",
            logging.ERROR,
            __file__,
            1,
            "SECRET-LOG-HANDLER-SENTINEL",
            (),
            None,
        )
        runtime._log_handler.emit(record)

    assert "SECRET-LOG-HANDLER-SENTINEL" not in capsys.readouterr().err


def test_runtime_shutdown_is_idempotent_and_releases_port(runtime: OpenTelemetryRuntime) -> None:
    port = _free_port()
    with runtime.observe_create(port):
        pass

    runtime.shutdown()
    runtime.shutdown()
    assert runtime.metrics_url is None
    with socket.socket() as rebound:
        rebound.bind(("127.0.0.1", port))


def test_runtime_shutdown_during_create_skips_duration_without_warning(
    runtime: OpenTelemetryRuntime,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING)
    with runtime.observe_create(_free_port()):
        runtime.shutdown()

    assert "Failed to record OpenTelemetry create duration" not in caplog.text


def test_disabled_runtime_does_not_import_otel_sdk_in_fresh_process() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "from data_designer.integrations.opentelemetry import OpenTelemetryRuntime; "
                "runtime = OpenTelemetryRuntime(); "
                "context = runtime.observe_create(None); context.__enter__(); context.__exit__(None, None, None); "
                "assert 'opentelemetry.sdk' not in sys.modules; "
                "assert 'opentelemetry.exporter.prometheus' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def _data_designer(
    artifact_path: Path,
    managed_assets_path: Path,
    model_providers: list[ModelProvider],
    runtime: OpenTelemetryRuntime,
    port: int,
) -> DataDesigner:
    managed_assets_path.mkdir(parents=True, exist_ok=True)
    designer = DataDesigner(
        artifact_path=artifact_path,
        model_providers=model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=managed_assets_path,
    )
    designer._open_telemetry = runtime
    designer.set_run_config(RunConfig(otel_metrics_port=port, display_tui=False))
    return designer


def test_data_designer_create_exports_and_accumulates_end_to_end(
    runtime: OpenTelemetryRuntime,
    tmp_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_sampler_only_config_builder: DataDesignerConfigBuilder,
) -> None:
    port = _free_port()
    designer = _data_designer(
        tmp_path / "artifacts",
        tmp_path / "managed-assets",
        stub_model_providers,
        runtime,
        port,
    )

    designer.create(stub_sampler_only_config_builder, dataset_name="first", num_records=3)
    designer.create(stub_sampler_only_config_builder, dataset_name="second", num_records=2)

    metrics = _scrape(runtime)
    generated_line = next(
        line
        for line in metrics.splitlines()
        if line.startswith("data_designer_dataset_records_total{") and 'record_result="generated"' in line
    )
    completed_line = next(
        line
        for line in metrics.splitlines()
        if line.startswith("data_designer_scheduler_events_total{") and 'event_name="job.completed"' in line
    )
    create_line = next(
        line for line in metrics.splitlines() if line.startswith("data_designer_create_duration_seconds_count")
    )
    assert generated_line.endswith(" 5.0")
    assert completed_line.endswith(" 2.0")
    assert create_line.endswith(" 2.0")


def test_data_designer_model_and_log_sinks_export_once_end_to_end(
    runtime: OpenTelemetryRuntime,
    tmp_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_model_configs: list[ModelConfig],
) -> None:
    port = _free_port()
    designer = _data_designer(
        tmp_path / "artifacts",
        tmp_path / "managed-assets",
        stub_model_providers,
        runtime,
        port,
    )
    model_config = stub_model_configs[0].model_copy(update={"skip_health_check": True})
    config_builder = DataDesignerConfigBuilder(model_configs=[model_config])
    config_builder.add_column(LLMTextColumnConfig(name="text", prompt="Generate text", model_alias="stub-model"))
    completion = AsyncMock(return_value=make_stub_completion_response(content="generated"))

    with patch.object(OpenAICompatibleClient, "acompletion", completion):
        designer.create(
            config_builder,
            dataset_name="model",
            num_records=1,
            on_batch_complete=lambda _path: logging.getLogger("data_designer.test").critical("counted once"),
        )

    metrics = _scrape(runtime)
    request_line = next(
        line for line in metrics.splitlines() if line.startswith("gen_ai_client_operation_duration_seconds_count{")
    )
    log_line = next(
        line
        for line in metrics.splitlines()
        if line.startswith("data_designer_log_records_total{") and 'log_severity="fatal"' in line
    )
    assert 'gen_ai_operation_name="chat"' in request_line
    assert 'gen_ai_provider_name="provider-1"' in request_line
    assert request_line.endswith(" 1.0")
    assert log_line.endswith(" 1.0")
    completion.assert_awaited_once()


def test_concurrent_data_designer_creates_share_one_exporter(
    runtime: OpenTelemetryRuntime,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stub_model_providers: list[ModelProvider],
    stub_sampler_only_config_builder: DataDesignerConfigBuilder,
) -> None:
    port = _free_port()
    managed_assets_path = tmp_path / "managed-assets"
    designers = [
        _data_designer(tmp_path / f"artifacts-{index}", managed_assets_path, stub_model_providers, runtime, port)
        for index in range(2)
    ]
    barrier = Barrier(2)
    observe_create = runtime.observe_create

    @contextlib.contextmanager
    def synchronized_observe_create(configured_port: int | None) -> Iterator[None]:
        with observe_create(configured_port):
            barrier.wait(timeout=10)
            yield

    monkeypatch.setattr(runtime, "observe_create", synchronized_observe_create)

    def create(index: int) -> None:
        designers[index].create(
            stub_sampler_only_config_builder,
            dataset_name=f"concurrent-{index}",
            num_records=1,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(create, range(2)))

    metrics = _scrape(runtime)
    generated_line = next(
        line
        for line in metrics.splitlines()
        if line.startswith("data_designer_dataset_records_total{") and 'record_result="generated"' in line
    )
    completed_line = next(
        line
        for line in metrics.splitlines()
        if line.startswith("data_designer_scheduler_events_total{") and 'event_name="job.completed"' in line
    )
    assert generated_line.endswith(" 2.0")
    assert completed_line.endswith(" 2.0")


_FORK_PRELUDE = r"""
import multiprocessing
import os
import re
import socket
import sys
import threading
import time
import urllib.request

from data_designer.integrations import opentelemetry

runtime = opentelemetry.get_open_telemetry_runtime()
ctx = multiprocessing.get_context("fork")
# Registered after the runtime's hooks, so it runs before them and records when a fork began,
# before the fork waits on the runtime.
fork_started_at = []
os.register_at_fork(before=lambda: fork_started_at.append(time.monotonic()))


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def create_count(url):
    with urllib.request.urlopen(url, timeout=5) as response:
        text = response.read().decode()
    match = re.search(r"^data_designer_create_duration_seconds_count\{[^}]*\} (\S+)$", text, re.MULTILINE)
    return float(match.group(1))


def finish(child):
    child.join(timeout=20)
    if child.is_alive():
        child.kill()
        child.join()
        sys.exit("forked child hung")
    if child.exitcode != 0:
        sys.exit(f"forked child exited with code {child.exitcode}")


# Set resume from a timer, for a fork that may wait until the blocked thread resumes. Call the
# returned function after the fork; it fails if the timer fired before the fork began, since the
# test then did not fork during the blocked window.
def resume_from_timer(resume):
    resumed_at = []
    timer = threading.Timer(1.0, lambda: (resumed_at.append(time.monotonic()), resume.set()))
    timer_started_at = time.monotonic()
    timer.start()

    def check():
        timer.join()
        forks = [started_at for started_at in fork_started_at if started_at >= timer_started_at]
        assert forks and forks[0] < resumed_at[0], "timer resumed the blocked thread before the fork"

    return check


def observe_create_and_check(port):
    with runtime.observe_create(port):
        pass
    url = f"http://127.0.0.1:{port}/metrics"
    assert runtime.metrics_url == url
    assert create_count(url) == 1.0, "forked child inherited the parent's metric values"
    runtime.shutdown()


with runtime.observe_create(free_port()):
    pass
parent_url = runtime.metrics_url
"""

# Binding a server in the forked child resolves its address (getaddrinfo, getfqdn), which can
# hang on macOS after forking a multi-threaded process.
requires_safe_fork = pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods() or sys.platform == "darwin",
    reason="requires a fork start method that is safe with resolver calls",
)


def _run_probe(script: str) -> None:
    """Run ``script`` in a fresh interpreter, avoiding whatever earlier tests left in the pytest worker."""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, f"probe failed:\n{result.stderr}"


def _run_fork_probe(body: str) -> None:
    """Run ``body`` in a fresh interpreter after a parent run that bound a metrics server."""
    _run_probe(_FORK_PRELUDE + textwrap.dedent(body))


# Stages a lock cycle that deadlocks if the runtime takes a logging lock while holding the lock
# its metric log handler needs: an emitter holds the handler lock and records the log, then
# dictConfig() takes logging's module lock and waits for the handler lock. STAGE_LIFECYCLE_STEP
# must hold the runtime inside a lifecycle step until `go` is set, then take logging's module lock.
_LOG_LOCK_CYCLE_PRELUDE = r"""
import contextvars
import logging
import logging.config
import os
import socket
import sys
import threading
import time

paused, go = threading.Event(), threading.Event()
"""

_LOG_LOCK_CYCLE_BODY = r"""
runtime = opentelemetry.get_open_telemetry_runtime()
with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
with runtime.observe_create(port):
    run_context = contextvars.copy_context()

data_designer_logger = logging.getLogger("data_designer")
recording = threading.Event()
original_record_log = runtime.record_log


def signalling_record_log(level):
    recording.set()
    original_record_log(level)


runtime.record_log = signalling_record_log
threads = [STAGE_LIFECYCLE_STEP()]
assert paused.wait(20)
threads.append(threading.Thread(target=run_context.run, args=(data_designer_logger.warning, "probe"), daemon=True))
threads[-1].start()
assert recording.wait(20)
threads.append(
    threading.Thread(
        target=logging.config.dictConfig, args=({"version": 1, "disable_existing_loggers": False},), daemon=True
    )
)
threads[-1].start()
time.sleep(0.5)
go.set()
for thread in threads:
    thread.join(10)
if any(thread.is_alive() for thread in threads):
    sys.stderr.write("deadlock between the runtime, a metric log emission and dictConfig()\n")
    sys.stderr.flush()
    os._exit(1)
runtime.shutdown()
"""


@requires_safe_fork
def test_forked_child_starts_fresh_metrics_runtime() -> None:
    _run_fork_probe(
        """
        child = ctx.Process(target=observe_create_and_check, args=(free_port(),))
        child.start()
        finish(child)
        assert runtime.metrics_url == parent_url
        assert create_count(parent_url) == 1.0
        runtime.shutdown()
        """
    )


@requires_safe_fork
def test_forked_child_does_not_keep_parent_metrics_port_bound() -> None:
    _run_fork_probe(
        """
        port = runtime._port
        release = ctx.Event()
        child = ctx.Process(target=release.wait, args=(20,))
        child.start()
        try:
            runtime.shutdown()
            with runtime.observe_create(port):
                pass
            assert runtime.metrics_url == parent_url, "parent could not rebind its port while a forked child lived"
        finally:
            release.set()
        finish(child)
        runtime.shutdown()
        """
    )


@requires_safe_fork
def test_forked_child_does_not_inherit_held_metrics_sdk_locks() -> None:
    _run_fork_probe(
        """
        histogram = runtime._create_duration
        locks = []
        for storage in histogram._measurement_consumer._reader_storages.values():
            for match in storage._instrument_view_instrument_matches.get(histogram, []):
                locks.append(match._lock)
                locks.extend(aggregation._lock for aggregation in match._attributes_aggregation.values())
        assert locks, "OpenTelemetry SDK internals changed; update this probe"
        for lock in locks:
            lock.acquire()
        try:
            child = ctx.Process(target=observe_create_and_check, args=(free_port(),))
            child.start()
            finish(child)
        finally:
            for lock in locks:
                lock.release()
        runtime.shutdown()
        """
    )


@requires_safe_fork
def test_forked_child_closes_socket_of_server_being_stopped() -> None:
    _run_fork_probe(
        """
        import threading

        port = runtime._port
        stopping, resume = threading.Event(), threading.Event()
        original_stop_server = opentelemetry._stop_server


        def blocking_stop_server(server, thread):
            stopping.set()
            resume.wait(20)
            original_stop_server(server, thread)


        # Fork after shutdown() has detached the server but before it closed the socket.
        opentelemetry._stop_server = blocking_stop_server
        shutdown_thread = threading.Thread(target=runtime.shutdown)
        shutdown_thread.start()
        assert stopping.wait(20)
        release = ctx.Event()
        child = ctx.Process(target=release.wait, args=(20,))
        child.start()
        resume.set()
        shutdown_thread.join(20)
        opentelemetry._stop_server = original_stop_server
        try:
            with runtime.observe_create(port):
                pass
            assert runtime.metrics_url == parent_url, "parent could not rebind its port while a forked child lived"
        finally:
            release.set()
        finish(child)
        runtime.shutdown()
        """
    )


@requires_safe_fork
def test_forked_child_drops_log_handler_attached_during_initialization() -> None:
    _run_fork_probe(
        """
        import logging
        import threading

        runtime.shutdown()
        data_designer_logger = logging.getLogger("data_designer")
        original_add_handler = data_designer_logger.addHandler
        attached, resume = threading.Event(), threading.Event()


        def blocking_add_handler(handler):
            original_add_handler(handler)
            if isinstance(handler, opentelemetry._MetricLogHandler):
                attached.set()
                resume.wait(20)


        def check_metric_log_handlers(port):
            with runtime.observe_create(port):
                pass
            handlers = [h for h in data_designer_logger.handlers if isinstance(h, opentelemetry._MetricLogHandler)]
            assert len(handlers) == 1, f"forked child has {len(handlers)} metric log handlers"
            runtime.shutdown()


        # Fork after _initialize() attached its log handler but before it recorded it. The fork
        # may wait for initialization to finish, so resume it from a timer instead of after it.
        data_designer_logger.addHandler = blocking_add_handler
        init_thread = threading.Thread(target=runtime._activate_run, args=("run-init", free_port()))
        init_thread.start()
        assert attached.wait(20)
        del data_designer_logger.addHandler
        resume_after_fork = resume_from_timer(resume)
        child = ctx.Process(target=check_metric_log_handlers, args=(free_port(),))
        child.start()
        resume_after_fork()
        try:
            finish(child)
        finally:
            resume.set()
            init_thread.join(20)
        runtime.shutdown()
        """
    )


@requires_safe_fork
def test_forked_child_closes_socket_of_server_being_started() -> None:
    _run_fork_probe(
        """
        import threading

        import prometheus_client

        runtime.shutdown()
        port = free_port()
        original_start_http_server = prometheus_client.start_http_server
        started, resume = threading.Event(), threading.Event()


        def blocking_start_http_server(*args, **kwargs):
            result = original_start_http_server(*args, **kwargs)
            started.set()
            resume.wait(20)
            return result


        # Fork after the server bound its socket but before the runtime recorded it. The fork
        # may wait for startup to finish, so resume startup from a timer instead of after it.
        prometheus_client.start_http_server = blocking_start_http_server
        start_thread = threading.Thread(target=runtime._activate_run, args=("run-start", port))
        start_thread.start()
        assert started.wait(20)
        prometheus_client.start_http_server = original_start_http_server
        resume_after_fork = resume_from_timer(resume)
        release = ctx.Event()
        child = ctx.Process(target=release.wait, args=(20,))
        child.start()
        resume_after_fork()
        start_thread.join(20)
        try:
            runtime.shutdown()
            with runtime.observe_create(port):
                pass
            url = f"http://127.0.0.1:{port}/metrics"
            assert runtime.metrics_url == url, "parent could not rebind its port while a forked child lived"
        finally:
            release.set()
        finish(child)
        runtime.shutdown()
        """
    )


@requires_safe_fork
def test_forked_child_drops_log_handler_removed_during_shutdown() -> None:
    _run_fork_probe(
        """
        import logging
        import threading

        data_designer_logger = logging.getLogger("data_designer")
        original_remove_handler = data_designer_logger.removeHandler
        removing, resume = threading.Event(), threading.Event()


        def blocking_remove_handler(handler):
            if isinstance(handler, opentelemetry._MetricLogHandler):
                removing.set()
                resume.wait(20)
            original_remove_handler(handler)


        def check_metric_log_handlers(port):
            with runtime.observe_create(port):
                pass
            handlers = [h for h in data_designer_logger.handlers if isinstance(h, opentelemetry._MetricLogHandler)]
            assert len(handlers) == 1, f"forked child has {len(handlers)} metric log handlers"
            runtime.shutdown()


        # Fork after shutdown() detached its log handler but before it removed it.
        data_designer_logger.removeHandler = blocking_remove_handler
        shutdown_thread = threading.Thread(target=runtime.shutdown)
        shutdown_thread.start()
        assert removing.wait(20)
        del data_designer_logger.removeHandler
        resume_after_fork = resume_from_timer(resume)
        child = ctx.Process(target=check_metric_log_handlers, args=(free_port(),))
        child.start()
        resume_after_fork()
        try:
            finish(child)
        finally:
            resume.set()
            shutdown_thread.join(20)
        """
    )


def test_shutdown_does_not_deadlock_with_metric_logging_and_dict_config() -> None:
    _run_probe(
        _LOG_LOCK_CYCLE_PRELUDE
        + textwrap.dedent(
            """
            from data_designer.integrations import opentelemetry


            def STAGE_LIFECYCLE_STEP():
                # Hold shutdown() after it detached its log handler, before it removes it.
                data_designer_logger = logging.getLogger("data_designer")
                original_remove_handler = data_designer_logger.removeHandler

                def blocking_remove_handler(handler):
                    if isinstance(handler, opentelemetry._MetricLogHandler):
                        paused.set()
                        go.wait(20)
                    original_remove_handler(handler)

                data_designer_logger.removeHandler = blocking_remove_handler
                thread = threading.Thread(target=opentelemetry.get_open_telemetry_runtime().shutdown, daemon=True)
                thread.start()
                return thread
            """
        )
        + _LOG_LOCK_CYCLE_BODY
    )


@requires_safe_fork
def test_fork_does_not_deadlock_with_metric_logging_and_dict_config() -> None:
    _run_probe(
        _LOG_LOCK_CYCLE_PRELUDE
        + textwrap.dedent(
            """
            import multiprocessing

            staged = threading.Event()


            def pause_between_fork_hooks():
                if staged.is_set():
                    staged.clear()
                    paused.set()
                    go.wait(20)


            # Registered after logging's fork hooks and before the runtime's, so it runs after the
            # runtime's before-fork hook and before logging's, which takes logging's module lock.
            assert "data_designer.integrations.opentelemetry" not in sys.modules
            os.register_at_fork(before=pause_between_fork_hooks)
            from data_designer.integrations import opentelemetry


            def STAGE_LIFECYCLE_STEP():
                child = multiprocessing.get_context("fork").Process(target=int)
                staged.set()
                thread = threading.Thread(target=child.start, daemon=True)
                thread.start()
                return thread
            """
        )
        + _LOG_LOCK_CYCLE_BODY
    )


def test_metric_log_record_does_not_wait_for_metrics_server_startup() -> None:
    _run_probe(
        textwrap.dedent(
            """
            import contextvars
            import logging
            import socket
            import threading
            import time

            import prometheus_client

            from data_designer.integrations import opentelemetry

            runtime = opentelemetry.get_open_telemetry_runtime()


            def free_port():
                with socket.socket() as sock:
                    sock.bind(("127.0.0.1", 0))
                    return sock.getsockname()[1]


            with runtime.observe_create(free_port()):
                run_context = contextvars.copy_context()
            original_start_http_server = prometheus_client.start_http_server
            starting, resume = threading.Event(), threading.Event()


            def blocking_start_http_server(*args, **kwargs):
                starting.set()
                resume.wait(20)
                return original_start_http_server(*args, **kwargs)


            # A log record emitted while another thread starts the metrics server must not wait
            # for the bind, which can include a slow reverse DNS lookup.
            runtime.shutdown()
            prometheus_client.start_http_server = blocking_start_http_server
            start_thread = threading.Thread(target=runtime._activate_run, args=("run-start", free_port()))
            start_thread.start()
            assert starting.wait(20)
            started_at = time.monotonic()
            run_context.run(logging.getLogger("data_designer").warning, "probe")
            waited = time.monotonic() - started_at
            resume.set()
            start_thread.join(20)
            assert waited < 1.0, f"log record waited {waited:.1f}s for metrics server startup"
            runtime.shutdown()
            """
        )
    )
