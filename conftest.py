# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Root-level conftest.py for pytest plugin registration
# This must be at the root level per pytest deprecation warning:
# https://docs.pytest.org/en/stable/deprecations.html#pytest-plugins-in-non-top-level-conftest-files

# Build pytest_plugins list dynamically based on which packages are installed.
# This allows CI to test packages in isolation (e.g., only data-designer-config installed).
from importlib.util import find_spec

pytest_plugins: list[str] = []

if find_spec("data_designer.config"):
    pytest_plugins.append("data_designer.config.testing.fixtures")

if find_spec("data_designer.engine"):
    pytest_plugins.append("data_designer.engine.testing.fixtures")
