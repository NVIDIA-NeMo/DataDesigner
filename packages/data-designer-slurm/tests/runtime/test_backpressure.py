# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from data_designer.slurm.runtime.backpressure import (
    MAX_WAITING_REQUESTS_ENVIRONMENT,
    RETRY_AFTER_SECONDS_ENVIRONMENT,
    AsgiMessage,
    QueueBackpressureController,
    QueueBackpressureSettings,
    QueueDepthBackpressureMiddleware,
)


def test_backpressure_settings_are_strict_and_support_absent_retry_header() -> None:
    settings = QueueBackpressureSettings.from_environment(
        {
            MAX_WAITING_REQUESTS_ENVIRONMENT: "7",
            RETRY_AFTER_SECONDS_ENVIRONMENT: "",
        }
    )
    assert settings.max_waiting_requests == 7
    assert settings.retry_after_seconds is None
    with pytest.raises(ValueError, match="threshold"):
        QueueBackpressureSettings.from_environment({MAX_WAITING_REQUESTS_ENVIRONMENT: "-1"})


@pytest.mark.asyncio
async def test_middleware_rejects_fresh_overload_and_preserves_health() -> None:
    downstream_calls: list[str] = []
    messages: list[AsgiMessage] = []

    async def app(scope: dict[str, object], receive: object, send: object) -> None:
        del receive, send
        downstream_calls.append(str(scope["path"]))

    async def receive() -> AsgiMessage:
        return {"type": "http.request"}

    async def send(message: AsgiMessage) -> None:
        messages.append(message)

    controller = QueueBackpressureController(
        QueueBackpressureSettings(2, 3),
        reader=lambda: 4,
        start_sampler=False,
    )
    controller.sample_once()
    middleware = QueueDepthBackpressureMiddleware(app, controller)

    await middleware({"type": "http", "path": "/v1/chat/completions"}, receive, send)
    await middleware({"type": "http", "path": "/health"}, receive, send)

    assert messages[0]["status"] == 429
    assert (b"retry-after", b"3") in messages[0]["headers"]
    assert downstream_calls == ["/health"]


@pytest.mark.asyncio
async def test_middleware_fails_open_when_metrics_are_unavailable() -> None:
    called = False

    async def app(scope: dict[str, object], receive: object, send: object) -> None:
        nonlocal called
        del scope, receive, send
        called = True

    async def receive() -> AsgiMessage:
        return {"type": "http.request"}

    async def send(message: AsgiMessage) -> None:
        raise AssertionError(f"unexpected middleware response: {message}")

    controller = QueueBackpressureController(reader=lambda: None, start_sampler=False)
    controller.sample_once()
    middleware = QueueDepthBackpressureMiddleware(app, controller)
    await middleware({"type": "http", "path": "/v1/completions"}, receive, send)

    assert called


def test_sampler_fails_open_after_reader_error_and_recovers() -> None:
    readings: list[Exception | int] = [RuntimeError("metrics unavailable"), 4]

    def read_queue_depth() -> int:
        reading = readings.pop(0)
        if isinstance(reading, Exception):
            raise reading
        return reading

    controller = QueueBackpressureController(
        QueueBackpressureSettings(2, 3),
        reader=read_queue_depth,
        start_sampler=False,
    )

    failed_snapshot = controller.sample_once()
    failed_reject, _ = controller.should_reject()
    recovered_snapshot = controller.sample_once()
    recovered_reject, _ = controller.should_reject()

    assert failed_snapshot.depth is None
    assert not failed_reject
    assert recovered_snapshot.depth == 4
    assert recovered_reject


def test_sampler_preserves_process_control_exceptions() -> None:
    def interrupt() -> int:
        raise KeyboardInterrupt

    controller = QueueBackpressureController(reader=interrupt, start_sampler=False)

    with pytest.raises(KeyboardInterrupt):
        controller.sample_once()
