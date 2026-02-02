# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Root-level conftest.py for pytest plugin registration
# This must be at the root level per pytest deprecation warning:
# https://docs.pytest.org/en/stable/deprecations.html#pytest-plugins-in-non-top-level-conftest-files

pytest_plugins = ["data_designer.config.testing.fixtures", "data_designer.engine.testing.fixtures"]
