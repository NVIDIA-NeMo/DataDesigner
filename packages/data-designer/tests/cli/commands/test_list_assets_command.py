# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import MagicMock, patch

from data_designer.cli.commands.list_assets import list_assets_command
from data_designer.cli.controllers.list_assets_controller import ListAssetsController
from data_designer.config.utils.constants import DATA_DESIGNER_HOME


@patch("data_designer.cli.commands.list_assets.ListAssetsController")
def test_list_assets_command_delegates_to_controller(mock_controller_cls: MagicMock) -> None:
    """Command creates controller with DATA_DESIGNER_HOME and delegates."""
    mock_controller = MagicMock(spec=ListAssetsController)
    mock_controller_cls.return_value = mock_controller

    list_assets_command(output_format=MagicMock(value="text"))

    mock_controller_cls.assert_called_once_with(DATA_DESIGNER_HOME)
    mock_controller.list_assets.assert_called_once_with("text")


@patch("data_designer.cli.commands.list_assets.ListAssetsController")
def test_list_assets_command_passes_json_format(mock_controller_cls: MagicMock) -> None:
    """Command forwards the json format value to the controller."""
    mock_controller = MagicMock(spec=ListAssetsController)
    mock_controller_cls.return_value = mock_controller

    list_assets_command(output_format=MagicMock(value="json"))

    mock_controller.list_assets.assert_called_once_with("json")
