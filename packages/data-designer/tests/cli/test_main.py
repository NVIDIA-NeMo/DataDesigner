# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.metadata
from unittest.mock import Mock, call, patch

from pydantic import BaseModel
from typer.testing import CliRunner

from data_designer.cli.main import app, main
from data_designer.config.utils.constants import DEFAULT_NUM_RECORDS
from data_designer.recipes.recipe import DataDesignerRecipe

runner = CliRunner()


class DemoRecipeConfig(BaseModel):
    """Demo recipe config."""


@patch("data_designer.cli.main.app")
@patch("data_designer.cli.main.ensure_cli_default_model_settings")
def test_main_bootstraps_before_running_app(mock_bootstrap: Mock, mock_app: Mock) -> None:
    """The CLI entrypoint bootstraps defaults before invoking Typer."""
    call_order = Mock()
    call_order.attach_mock(mock_bootstrap, "bootstrap")
    call_order.attach_mock(mock_app, "app")

    with patch("sys.argv", ["data-designer"]):
        main()

    assert call_order.mock_calls == [call.bootstrap(), call.app()]


@patch("data_designer.cli.main.app")
@patch("data_designer.cli.main.ensure_cli_default_model_settings")
def test_main_skips_bootstrap_for_version(mock_bootstrap: Mock, mock_app: Mock) -> None:
    """The CLI entrypoint avoids default setup for the fast version path."""
    with patch("sys.argv", ["data-designer", "--version"]):
        main()

    mock_bootstrap.assert_not_called()
    mock_app.assert_called_once_with()


@patch("data_designer.cli.main.app")
@patch("data_designer.cli.main.ensure_cli_default_model_settings")
def test_main_skips_bootstrap_when_version_follows_another_flag(mock_bootstrap: Mock, mock_app: Mock) -> None:
    """The CLI entrypoint detects eager version requests even after another root flag."""
    with patch("sys.argv", ["data-designer", "--help", "--version"]):
        main()

    mock_bootstrap.assert_not_called()
    mock_app.assert_called_once_with()


def test_app_version_prints_installed_data_designer_version() -> None:
    with patch("data_designer.cli.main.importlib.metadata.version", return_value="0.6.0") as mock_version:
        result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.output == "0.6.0\n"
    mock_version.assert_called_once_with("data-designer")


def test_app_version_errors_when_package_version_is_missing() -> None:
    with patch(
        "data_designer.cli.main.importlib.metadata.version",
        side_effect=importlib.metadata.PackageNotFoundError("data-designer"),
    ):
        result = runner.invoke(app, ["--version"])

    assert result.exit_code == 1
    assert "Unable to resolve installed data-designer package version." in result.output


@patch("data_designer.cli.commands.create.GenerationController")
def test_app_dispatches_lazy_create_command(mock_controller_cls: Mock) -> None:
    """The Typer app dispatches lazy-loaded commands through the resolved callback."""
    mock_controller = Mock()
    mock_controller_cls.return_value = mock_controller

    result = runner.invoke(app, ["create", "config.yaml"])

    assert result.exit_code == 0
    mock_controller.run_create.assert_called_once_with(
        config_source="config.yaml",
        num_records=DEFAULT_NUM_RECORDS,
        dataset_name="dataset",
        artifact_path=None,
    )


@patch("data_designer.cli.commands.recipes.DataDesigner")
@patch("data_designer.cli.commands.recipes.RecipeRegistry")
def test_app_dispatches_lazy_run_recipe_command(mock_registry_cls: Mock, mock_data_designer_cls: Mock) -> None:
    """The Typer app dispatches the recipe command through the lazy command loader."""
    mock_config_builder = Mock()
    recipe = DataDesignerRecipe(
        name="demo",
        description="Demo recipe",
        config_model=DemoRecipeConfig,
        build_config=Mock(return_value=mock_config_builder),
    )
    mock_registry_cls.return_value.get_recipe.return_value = recipe
    mock_results = Mock()
    mock_results.load_dataset.return_value = [object()]
    mock_data_designer_cls.return_value.create.return_value = mock_results

    result = runner.invoke(app, ["run-recipe", "demo", "--num-records", "1"])

    assert result.exit_code == 0
    mock_registry_cls.return_value.get_recipe.assert_called_once_with("demo")
    mock_data_designer_cls.return_value.create.assert_called_once_with(
        mock_config_builder,
        num_records=1,
        dataset_name="dataset",
    )
