# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import socket

import pytest

pytest_plugins = [
    "data_designer.config.testing.fixtures",
    "data_designer.engine.testing.fixtures",
]


@pytest.fixture(autouse=True)
def numeric_loopback_names(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep local HTTP servers independent of reverse DNS."""
    original_getfqdn = socket.getfqdn

    def getfqdn(name: str = "") -> str:
        return name if name in ("127.0.0.1", "::1") else original_getfqdn(name)

    monkeypatch.setattr(socket, "getfqdn", getfqdn)
