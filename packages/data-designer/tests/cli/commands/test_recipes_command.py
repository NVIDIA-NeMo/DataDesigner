# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from pydantic import BaseModel

from data_designer.cli.commands.recipes import run_recipe_command
from data_designer.recipes.recipe import DataDesignerRecipe


class DemoRecipeConfig(BaseModel):
    """Demo recipe config."""

    value: str = "default"


def test_run_recipe_command_builds_and_creates_recipe() -> None:
    mock_config_builder = MagicMock()
    build_config = MagicMock(return_value=mock_config_builder)
    postprocess = MagicMock()
    recipe = DataDesignerRecipe(
        name="demo",
        description="Demo recipe",
        config_model=DemoRecipeConfig,
        build_config=build_config,
        postprocess=postprocess,
    )
    mock_results = MagicMock()
    mock_results.load_dataset.return_value = [object(), object()]

    with (
        patch("data_designer.cli.commands.recipes.RecipeRegistry") as mock_registry_cls,
        patch("data_designer.cli.commands.recipes.DataDesigner") as mock_data_designer_cls,
    ):
        mock_registry_cls.return_value.get_recipe.return_value = recipe
        mock_data_designer_cls.return_value.create.return_value = mock_results

        run_recipe_command(
            recipe_name="demo",
            config_path=None,
            mode="create",
            num_records=2,
            dataset_name="demo-dataset",
            artifact_path=Path("/tmp/artifacts"),
        )

    mock_registry_cls.return_value.get_recipe.assert_called_once_with("demo")
    build_config.assert_called_once()
    mock_data_designer_cls.assert_called_once_with(artifact_path=Path("/tmp/artifacts"))
    mock_data_designer_cls.return_value.create.assert_called_once_with(
        mock_config_builder,
        num_records=2,
        dataset_name="demo-dataset",
    )
    postprocess.assert_called_once()
