# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest

from data_designer.slurm.runtime import probes
from data_designer.slurm.runtime.probes import HttpReadinessProber


def test_readiness_probe_does_not_buffer_response_body(monkeypatch: pytest.MonkeyPatch) -> None:
    connection = _FakeConnection()
    monkeypatch.setattr(probes.http.client, "HTTPConnection", lambda *args, **kwargs: connection)

    assert HttpReadinessProber().is_ready("127.0.0.1", 8000, "/health", timeout_seconds=1.0)

    assert connection.requested == ("GET", "/health", {"Connection": "close"})
    assert connection.closed


class _FakeConnection:
    def __init__(self) -> None:
        self.requested: tuple[str, str, dict[str, str]] | None = None
        self.closed = False

    def request(self, method: str, path: str, *, headers: dict[str, str]) -> None:
        self.requested = (method, path, headers)

    def getresponse(self) -> SimpleNamespace:
        return SimpleNamespace(status=200)

    def close(self) -> None:
        self.closed = True
