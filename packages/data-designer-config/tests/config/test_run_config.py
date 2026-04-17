# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from data_designer.config.run_config import JinjaRenderingEngine, RunConfig


def test_run_config_defaults_to_native_jinja_renderer() -> None:
    assert JinjaRenderingEngine(RunConfig().jinja_rendering_engine) == JinjaRenderingEngine.NATIVE


def test_run_config_accepts_secure_renderer() -> None:
    run_config = RunConfig(jinja_rendering_engine=JinjaRenderingEngine.SECURE)
    assert JinjaRenderingEngine(run_config.jinja_rendering_engine) == JinjaRenderingEngine.SECURE
