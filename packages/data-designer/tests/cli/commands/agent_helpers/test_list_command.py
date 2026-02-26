# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import MagicMock, patch

from data_designer.cli.commands.agent_helpers.list import (
    column_types_command,
    model_aliases_command,
    persona_datasets_command,
    processor_types_command,
    sampler_types_command,
    validator_types_command,
)
from data_designer.cli.controllers.list_controller import ListController
from data_designer.config.utils.constants import DATA_DESIGNER_HOME

_PATCH_TARGET = "data_designer.cli.commands.agent_helpers.list.ListController"


# ---------------------------------------------------------------------------
# model-aliases
# ---------------------------------------------------------------------------


@patch(_PATCH_TARGET)
def test_model_aliases_delegates_text(mock_cls: MagicMock) -> None:
    mock_ctrl = MagicMock(spec=ListController)
    mock_cls.return_value = mock_ctrl

    model_aliases_command()

    mock_cls.assert_called_once_with(DATA_DESIGNER_HOME)
    mock_ctrl.list_model_aliases.assert_called_once_with()


# ---------------------------------------------------------------------------
# persona-datasets
# ---------------------------------------------------------------------------


@patch(_PATCH_TARGET)
def test_persona_datasets_delegates(mock_cls: MagicMock) -> None:
    mock_ctrl = MagicMock(spec=ListController)
    mock_cls.return_value = mock_ctrl

    persona_datasets_command()

    mock_cls.assert_called_once_with(DATA_DESIGNER_HOME)
    mock_ctrl.list_persona_datasets.assert_called_once_with()


# ---------------------------------------------------------------------------
# columns
# ---------------------------------------------------------------------------


@patch(_PATCH_TARGET)
def test_column_types_delegates(mock_cls: MagicMock) -> None:
    mock_ctrl = MagicMock(spec=ListController)
    mock_cls.return_value = mock_ctrl

    column_types_command()

    mock_cls.assert_called_once_with(DATA_DESIGNER_HOME)
    mock_ctrl.list_column_types.assert_called_once_with()


# ---------------------------------------------------------------------------
# samplers
# ---------------------------------------------------------------------------


@patch(_PATCH_TARGET)
def test_sampler_types_delegates(mock_cls: MagicMock) -> None:
    mock_ctrl = MagicMock(spec=ListController)
    mock_cls.return_value = mock_ctrl

    sampler_types_command()

    mock_cls.assert_called_once_with(DATA_DESIGNER_HOME)
    mock_ctrl.list_sampler_types.assert_called_once_with()


# ---------------------------------------------------------------------------
# validators
# ---------------------------------------------------------------------------


@patch(_PATCH_TARGET)
def test_validator_types_delegates(mock_cls: MagicMock) -> None:
    mock_ctrl = MagicMock(spec=ListController)
    mock_cls.return_value = mock_ctrl

    validator_types_command()

    mock_cls.assert_called_once_with(DATA_DESIGNER_HOME)
    mock_ctrl.list_validator_types.assert_called_once_with()


# ---------------------------------------------------------------------------
# processors
# ---------------------------------------------------------------------------


@patch(_PATCH_TARGET)
def test_processor_types_delegates(mock_cls: MagicMock) -> None:
    mock_ctrl = MagicMock(spec=ListController)
    mock_cls.return_value = mock_ctrl

    processor_types_command()

    mock_cls.assert_called_once_with(DATA_DESIGNER_HOME)
    mock_ctrl.list_processor_types.assert_called_once_with()
