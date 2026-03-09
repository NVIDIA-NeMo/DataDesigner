# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from data_designer.engine.models.clients.adapters.litellm_bridge import LiteLLMBridgeClient


@pytest.fixture
def mock_router() -> MagicMock:
    return MagicMock()


@pytest.fixture
def bridge_client(mock_router: MagicMock) -> LiteLLMBridgeClient:
    return LiteLLMBridgeClient(provider_name="stub-provider", router=mock_router)
