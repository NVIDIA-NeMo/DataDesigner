# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from data_designer.engine.testing.stubs import StubHuggingFaceSeedReader

pytest_plugins = ["data_designer.config.testing.fixtures"]


@pytest.fixture
def stub_seed_reader() -> StubHuggingFaceSeedReader:
    """Stub seed reader for testing seed dataset functionality."""
    return StubHuggingFaceSeedReader()
