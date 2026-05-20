# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from pydantic import ValidationError

import data_designer.config as dd
from data_designer.config.run_config import JinjaRenderingEngine, RequestAdmissionTuningConfig, RunConfig


def test_run_config_defaults_to_secure_jinja_renderer() -> None:
    assert JinjaRenderingEngine(RunConfig().jinja_rendering_engine) == JinjaRenderingEngine.SECURE


def test_run_config_accepts_native_renderer() -> None:
    run_config = RunConfig(jinja_rendering_engine=JinjaRenderingEngine.NATIVE)
    assert JinjaRenderingEngine(run_config.jinja_rendering_engine) == JinjaRenderingEngine.NATIVE


def test_run_config_rejects_removed_throttle_with_targeted_message() -> None:
    with pytest.raises(ValidationError, match="RunConfig.throttle was removed"):
        RunConfig(throttle={"max_concurrent_requests": 1})


def test_request_admission_tuning_config_accepts_canonical_fields() -> None:
    config = RequestAdmissionTuningConfig(
        multiplicative_decrease_factor=0.5,
        additive_increase_step=2,
        increase_after_successes=7,
        cooldown_seconds=1.5,
        startup_ramp_seconds=30.0,
    )

    assert config.multiplicative_decrease_factor == 0.5
    assert config.additive_increase_step == 2
    assert config.increase_after_successes == 7
    assert config.cooldown_seconds == 1.5
    assert config.startup_ramp_seconds == 30.0


def test_request_admission_tuning_config_accepts_throttle_era_field_names() -> None:
    config = RequestAdmissionTuningConfig(
        reduce_factor=0.5,
        additive_increase=2,
        success_window=7,
        cooldown_seconds=1.5,
        rampup_seconds=30.0,
    )

    assert config.multiplicative_decrease_factor == 0.5
    assert config.additive_increase_step == 2
    assert config.increase_after_successes == 7
    assert config.cooldown_seconds == 1.5
    assert config.startup_ramp_seconds == 30.0


def test_request_admission_tuning_config_rejects_duplicate_legacy_and_canonical_fields() -> None:
    with pytest.raises(ValidationError, match="Specify either 'reduce_factor' or 'multiplicative_decrease_factor'"):
        RequestAdmissionTuningConfig(reduce_factor=0.5, multiplicative_decrease_factor=0.75)


def test_run_config_accepts_request_admission_tuning() -> None:
    run_config = RunConfig(request_admission=RequestAdmissionTuningConfig(startup_ramp_seconds=10.0))

    assert run_config.request_admission is not None
    assert run_config.request_admission.startup_ramp_seconds == 10.0


def test_run_config_accepts_request_admission_tuning_dict_with_throttle_era_names() -> None:
    run_config = RunConfig(request_admission={"reduce_factor": 0.5, "success_window": 7, "rampup_seconds": 10.0})

    assert run_config.request_admission is not None
    assert run_config.request_admission.multiplicative_decrease_factor == 0.5
    assert run_config.request_admission.increase_after_successes == 7
    assert run_config.request_admission.startup_ramp_seconds == 10.0


def test_request_admission_tuning_config_is_exported_from_config_package() -> None:
    assert dd.RequestAdmissionTuningConfig is RequestAdmissionTuningConfig
