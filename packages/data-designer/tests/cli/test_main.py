# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import Mock, call, patch

from data_designer.cli.main import main


@patch("data_designer.cli.main.app")
@patch("data_designer.cli.main.ensure_cli_default_model_settings")
def test_main_bootstraps_before_running_app(mock_bootstrap: Mock, mock_app: Mock) -> None:
    """The CLI entrypoint bootstraps defaults before invoking Typer."""
    call_order = Mock()
    call_order.attach_mock(mock_bootstrap, "bootstrap")
    call_order.attach_mock(mock_app, "app")

    main()

    assert call_order.mock_calls == [call.bootstrap(), call.app()]
