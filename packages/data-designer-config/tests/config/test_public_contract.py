# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from data_designer.config import InvalidConfigError
from data_designer.config.errors import InvalidConfigError as InvalidConfigErrorDefinition


def test_invalid_config_error_is_publicly_exported() -> None:
    assert InvalidConfigError is InvalidConfigErrorDefinition
