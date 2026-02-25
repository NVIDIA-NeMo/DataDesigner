# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import io

import pytest

import data_designer.lazy_heavy_imports as lazy

pytest_plugins = ["data_designer.config.testing.fixtures"]


@pytest.fixture
def sample_png_bytes() -> bytes:
    """Create a valid 1x1 PNG as raw bytes."""
    img = lazy.Image.new("RGB", (1, 1), color="red")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def minimal_png_base64(sample_png_bytes: bytes) -> str:
    """Return a valid 1x1 PNG as a base64-encoded string."""
    return base64.b64encode(sample_png_bytes).decode()
