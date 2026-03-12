# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import patch

import pytest

import data_designer.cli.runtime as runtime_mod


def test_ensure_cli_default_model_settings_runs_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI bootstrap only attempts default setup once per process."""
    monkeypatch.setattr(runtime_mod, "_default_model_settings_checked", False)

    with patch("data_designer.cli.runtime.resolve_seed_default_model_settings") as mock_resolve:
        runtime_mod.ensure_cli_default_model_settings()
        runtime_mod.ensure_cli_default_model_settings()

    mock_resolve.assert_called_once_with()
    assert runtime_mod._default_model_settings_checked is True


def test_ensure_cli_default_model_settings_warns_and_continues(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI bootstrap prints an actionable warning when setup fails."""
    monkeypatch.setattr(runtime_mod, "_default_model_settings_checked", False)

    with (
        patch("data_designer.cli.runtime.print_warning") as mock_print_warning,
        patch("data_designer.cli.runtime.resolve_seed_default_model_settings", side_effect=RuntimeError("boom")),
    ):
        runtime_mod.ensure_cli_default_model_settings()

    mock_print_warning.assert_called_once()
    warning = mock_print_warning.call_args[0][0]
    assert "Could not initialize default model providers and model configs automatically." in warning
    assert "The command will continue." in warning
    assert "data-designer config providers" in warning
    assert "data-designer config models" in warning
    assert runtime_mod._default_model_settings_checked is True
