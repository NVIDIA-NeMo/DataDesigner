# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from data_designer.recipes.recipe import DataDesignerRecipe
from data_designer.recipes.registry import RecipeLoadError, RecipeRegistry


class EmptyRecipeConfig(BaseModel):
    """Empty recipe config for registry tests."""


class FakeEntryPoint:
    """Minimal entry point stub."""

    def __init__(self, name: str, loaded: Any):
        self.name = name
        self._loaded = loaded

    def load(self) -> Any:
        return self._loaded


def _recipe(name: str = "demo") -> DataDesignerRecipe:
    return DataDesignerRecipe(
        name=name,
        description="Demo recipe",
        config_model=EmptyRecipeConfig,
        build_config=lambda config: config,  # type: ignore[arg-type, return-value]
    )


def test_list_recipes_loads_recipe_entry_points() -> None:
    recipe = _recipe()
    entry_point = FakeEntryPoint("demo-entry-point", recipe)

    with patch("data_designer.recipes.registry.importlib.metadata.entry_points", return_value=[entry_point]):
        recipes = RecipeRegistry().list_recipes()

    assert len(recipes) == 1
    assert recipes[0].entry_point_name == "demo-entry-point"
    assert recipes[0].recipe is recipe


def test_get_recipe_matches_entry_point_or_recipe_name() -> None:
    recipe = _recipe(name="demo-recipe")
    entry_point = FakeEntryPoint("demo-entry-point", recipe)

    with patch("data_designer.recipes.registry.importlib.metadata.entry_points", return_value=[entry_point]):
        registry = RecipeRegistry()
        assert registry.get_recipe("demo-entry-point") is recipe
        assert registry.get_recipe("demo-recipe") is recipe


def test_list_recipes_rejects_non_recipe_entry_points() -> None:
    entry_point = FakeEntryPoint("bad", object())

    with patch("data_designer.recipes.registry.importlib.metadata.entry_points", return_value=[entry_point]):
        with pytest.raises(RecipeLoadError, match="expected DataDesignerRecipe"):
            RecipeRegistry().list_recipes()
