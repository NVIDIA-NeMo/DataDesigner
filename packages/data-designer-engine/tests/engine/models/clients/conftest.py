# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from data_designer.engine.models.clients.adapters.litellm_bridge import LiteLLMBridgeClient


@pytest.fixture
def mock_router() -> MagicMock:
    return MagicMock()


@pytest.fixture
def bridge_client(mock_router: MagicMock) -> LiteLLMBridgeClient:
    return LiteLLMBridgeClient(provider_name="stub-provider", router=mock_router)


# ---------------------------------------------------------------------------
# Shared mock helpers for native HTTP adapter tests
# ---------------------------------------------------------------------------


def mock_httpx_response(json_data: dict[str, Any], status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.text = json.dumps(json_data)
    resp.headers = {}
    return resp


def make_mock_sync_client(response_json: dict[str, Any], status_code: int = 200) -> MagicMock:
    mock = MagicMock()
    mock.post = MagicMock(return_value=mock_httpx_response(response_json, status_code))
    return mock


def make_mock_async_client(response_json: dict[str, Any], status_code: int = 200) -> MagicMock:
    mock = MagicMock()
    mock.post = AsyncMock(return_value=mock_httpx_response(response_json, status_code))
    return mock
