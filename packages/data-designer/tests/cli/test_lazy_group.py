# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import MagicMock, patch

import click

from data_designer.cli.lazy_group import _wrap_command_with_cli_bootstrap


@patch("data_designer.cli.lazy_group.ensure_cli_default_model_settings")
def test_wrap_command_with_cli_bootstrap_is_idempotent(mock_bootstrap: MagicMock) -> None:
    """Wrapping the same command twice should only add one bootstrap layer."""
    callback = MagicMock()
    command = click.Command("test", callback=callback)

    _wrap_command_with_cli_bootstrap(command)
    _wrap_command_with_cli_bootstrap(command)

    assert command.callback is not None
    command.callback()

    mock_bootstrap.assert_called_once_with()
    callback.assert_called_once_with()
