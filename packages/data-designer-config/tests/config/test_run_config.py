# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from data_designer.config.run_config import JinjaRenderingEngine, RunConfig, ThrottleConfig


def test_run_config_defaults_to_secure_jinja_renderer() -> None:
    assert JinjaRenderingEngine(RunConfig().jinja_rendering_engine) == JinjaRenderingEngine.SECURE


def test_run_config_accepts_native_renderer() -> None:
    run_config = RunConfig(jinja_rendering_engine=JinjaRenderingEngine.NATIVE)
    assert JinjaRenderingEngine(run_config.jinja_rendering_engine) == JinjaRenderingEngine.NATIVE


def test_run_config_preserves_dropped_columns_by_default() -> None:
    assert RunConfig().preserve_dropped_columns is True


def test_run_config_accepts_disabled_dropped_column_preservation() -> None:
    run_config = RunConfig(preserve_dropped_columns=False)
    assert run_config.preserve_dropped_columns is False


def test_throttle_config_accepts_rampup_seconds() -> None:
    config = ThrottleConfig(rampup_seconds=30.0)
    assert config.rampup_seconds == 30.0


def test_throttle_config_rejects_negative_rampup_seconds() -> None:
    with pytest.raises(ValueError, match="rampup_seconds"):
        ThrottleConfig(rampup_seconds=-1.0)
