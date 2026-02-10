# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.metadata
from unittest.mock import patch

from data_designer.config.version import get_library_version


def test_get_library_version_returns_valid_string() -> None:
    version = get_library_version()
    assert isinstance(version, str)
    assert version != "unknown"
    assert len(version) > 0


def test_get_library_version_returns_unknown_on_missing_package() -> None:
    with patch(
        "data_designer.config.version.importlib.metadata.version",
        side_effect=importlib.metadata.PackageNotFoundError("data-designer-config"),
    ):
        assert get_library_version() == "unknown"
