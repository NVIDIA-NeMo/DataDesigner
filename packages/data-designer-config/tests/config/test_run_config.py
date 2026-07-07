# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from pydantic import ValidationError

import data_designer.config as dd
from data_designer.config.run_config import (
    JinjaRenderingEngine,
    RequestAdmissionTuningConfig,
    RunConfig,
    ThrottleConfig,
)


def test_run_config_defaults_to_secure_jinja_renderer() -> None:
    assert JinjaRenderingEngine(RunConfig().jinja_rendering_engine) == JinjaRenderingEngine.SECURE


def test_run_config_accepts_native_renderer() -> None:
    run_config = RunConfig(jinja_rendering_engine=JinjaRenderingEngine.NATIVE)
    assert JinjaRenderingEngine(run_config.jinja_rendering_engine) == JinjaRenderingEngine.NATIVE


def test_run_config_defaults_to_display_tui_disabled() -> None:
    assert RunConfig().display_tui is False


def test_run_config_accepts_display_tui() -> None:
    assert RunConfig(display_tui=False).display_tui is False


def test_run_config_does_not_write_scheduler_events_by_default() -> None:
    assert RunConfig().write_scheduler_events is False


def test_run_config_accepts_scheduler_event_writes() -> None:
    assert RunConfig(write_scheduler_events=True).write_scheduler_events is True


def test_run_config_progress_bar_shim_translates_to_display_tui() -> None:
    with pytest.warns(DeprecationWarning, match="RunConfig.progress_bar.*RunConfig.display_tui") as caught:
        run_config = RunConfig(progress_bar=False)

    assert run_config.display_tui is False
    assert caught[0].filename == __file__


def test_run_config_progress_bar_property_getter_warns() -> None:
    run_config = RunConfig(display_tui=False)

    with pytest.warns(DeprecationWarning, match="RunConfig.progress_bar.*RunConfig.display_tui"):
        assert run_config.progress_bar is False


def test_run_config_progress_bar_property_setter_warns() -> None:
    run_config = RunConfig(display_tui=False)

    with pytest.warns(DeprecationWarning, match="RunConfig.progress_bar.*RunConfig.display_tui"):
        run_config.progress_bar = True

    assert run_config.display_tui is True


def test_run_config_model_copy_progress_bar_shim_translates_to_display_tui() -> None:
    run_config = RunConfig(display_tui=True)

    with pytest.warns(DeprecationWarning, match="RunConfig.progress_bar.*RunConfig.display_tui"):
        copied = run_config.model_copy(update={"progress_bar": False})

    assert copied.display_tui is False


def test_run_config_model_copy_display_tui_wins_over_progress_bar_shim() -> None:
    run_config = RunConfig(display_tui=True)

    with pytest.warns(DeprecationWarning, match="RunConfig.progress_bar.*RunConfig.display_tui"):
        copied = run_config.model_copy(update={"progress_bar": False, "display_tui": True})

    assert copied.display_tui is True


def test_run_config_preserves_dropped_columns_by_default() -> None:
    assert RunConfig().preserve_dropped_columns is True


def test_run_config_accepts_disabled_dropped_column_preservation() -> None:
    run_config = RunConfig(preserve_dropped_columns=False)
    assert run_config.preserve_dropped_columns is False


def test_run_config_defaults_max_in_flight_tasks_to_1024() -> None:
    assert RunConfig().max_in_flight_tasks == 1024


def test_run_config_accepts_custom_max_in_flight_tasks() -> None:
    run_config = RunConfig(max_in_flight_tasks=2048)

    assert run_config.max_in_flight_tasks == 2048


def test_run_config_rejects_invalid_max_in_flight_tasks() -> None:
    with pytest.raises(ValidationError, match="max_in_flight_tasks"):
        RunConfig(max_in_flight_tasks=0)


def test_run_config_throttle_shim_rejects_unknown_legacy_fields() -> None:
    with pytest.raises(ValidationError, match="max_concurrent_requests"):
        RunConfig(throttle={"max_concurrent_requests": 1})


def test_run_config_throttle_shim_translates_to_request_admission() -> None:
    with pytest.warns(DeprecationWarning, match="RunConfig.throttle.*RequestAdmissionTuningConfig"):
        run_config = RunConfig(
            throttle=ThrottleConfig(
                reduce_factor=0.5,
                additive_increase=2,
                success_window=7,
                cooldown_seconds=1.5,
                ceiling_overshoot=0.2,
                rampup_seconds=30.0,
            )
        )

    assert run_config.request_admission is not None
    assert run_config.request_admission.multiplicative_decrease_factor == 0.5
    assert run_config.request_admission.additive_increase_step == 2
    assert run_config.request_admission.successes_until_increase == 7
    assert run_config.request_admission.cooldown_seconds == 1.5
    assert run_config.request_admission.startup_ramp_seconds == 30.0


def test_run_config_throttle_shim_accepts_legacy_dict() -> None:
    with pytest.warns(DeprecationWarning, match="RunConfig.throttle.*RequestAdmissionTuningConfig"):
        run_config = RunConfig(
            throttle={
                "reduce_factor": 0.5,
                "additive_increase": 2,
                "success_window": 7,
                "cooldown_seconds": 1.5,
                "rampup_seconds": 30.0,
            }
        )

    assert run_config.request_admission is not None
    assert run_config.request_admission.multiplicative_decrease_factor == 0.5
    assert run_config.request_admission.additive_increase_step == 2
    assert run_config.request_admission.successes_until_increase == 7
    assert run_config.request_admission.cooldown_seconds == 1.5
    assert run_config.request_admission.startup_ramp_seconds == 30.0


def test_run_config_rejects_throttle_and_request_admission_together() -> None:
    with pytest.raises(ValidationError, match="Specify either RunConfig.throttle or RunConfig.request_admission"):
        RunConfig(throttle=ThrottleConfig(), request_admission=RequestAdmissionTuningConfig())


def test_request_admission_tuning_config_accepts_canonical_fields() -> None:
    config = RequestAdmissionTuningConfig(
        multiplicative_decrease_factor=0.5,
        additive_increase_step=2,
        successes_until_increase=7,
        cooldown_seconds=1.5,
        startup_ramp_seconds=30.0,
    )

    assert config.multiplicative_decrease_factor == 0.5
    assert config.additive_increase_step == 2
    assert config.successes_until_increase == 7
    assert config.cooldown_seconds == 1.5
    assert config.startup_ramp_seconds == 30.0


def test_request_admission_tuning_config_rejects_throttle_era_field_names() -> None:
    with pytest.raises(ValidationError, match="success_window"):
        RequestAdmissionTuningConfig(success_window=7)


def test_run_config_accepts_request_admission_tuning() -> None:
    run_config = RunConfig(request_admission=RequestAdmissionTuningConfig(startup_ramp_seconds=10.0))

    assert run_config.request_admission is not None
    assert run_config.request_admission.startup_ramp_seconds == 10.0


def test_run_config_accepts_request_admission_tuning_dict() -> None:
    run_config = RunConfig(
        request_admission={
            "multiplicative_decrease_factor": 0.5,
            "successes_until_increase": 7,
            "startup_ramp_seconds": 10.0,
        }
    )

    assert run_config.request_admission is not None
    assert run_config.request_admission.multiplicative_decrease_factor == 0.5
    assert run_config.request_admission.successes_until_increase == 7
    assert run_config.request_admission.startup_ramp_seconds == 10.0


def test_request_admission_tuning_config_is_exported_from_config_package() -> None:
    assert dd.RequestAdmissionTuningConfig is RequestAdmissionTuningConfig


def test_deprecated_throttle_config_is_exported_from_config_package() -> None:
    assert dd.ThrottleConfig is ThrottleConfig
    namespace: dict[str, object] = {}
    exec("from data_designer.config import ThrottleConfig", namespace)
    assert namespace["ThrottleConfig"] is ThrottleConfig


def test_throttle_config_accepts_rampup_seconds() -> None:
    config = ThrottleConfig(rampup_seconds=30.0)
    assert config.rampup_seconds == 30.0


def test_throttle_config_rejects_negative_rampup_seconds() -> None:
    with pytest.raises(ValueError, match="rampup_seconds"):
        ThrottleConfig(rampup_seconds=-1.0)
