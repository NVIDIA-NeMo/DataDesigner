# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass

from data_designer.recipes.recipe import DataDesignerRecipe

RECIPE_ENTRY_POINT_GROUP = "data_designer.recipes"


class RecipeLoadError(Exception):
    """Raised when a Data Designer recipe entry point cannot be loaded."""


@dataclass(frozen=True)
class RecipeInfo:
    """Metadata for an installed recipe."""

    entry_point_name: str
    recipe: DataDesignerRecipe


class RecipeRegistry:
    """Discover and load Data Designer recipes from Python entry points."""

    def list_recipes(self) -> list[RecipeInfo]:
        """Return all installed recipes sorted by entry point name."""
        recipes: list[RecipeInfo] = []
        for entry_point in importlib.metadata.entry_points(group=RECIPE_ENTRY_POINT_GROUP):
            recipe = self._load_entry_point(entry_point)
            recipes.append(RecipeInfo(entry_point_name=entry_point.name, recipe=recipe))
        return sorted(recipes, key=lambda item: item.entry_point_name)

    def get_recipe(self, recipe_name: str) -> DataDesignerRecipe:
        """Load a recipe by entry point name or recipe ``name``.

        Args:
            recipe_name: Entry point name or ``DataDesignerRecipe.name``.

        Returns:
            The requested recipe.

        Raises:
            RecipeLoadError: If no matching recipe is installed.
        """
        for item in self.list_recipes():
            if recipe_name in (item.entry_point_name, item.recipe.name):
                return item.recipe
        raise RecipeLoadError(f"No installed Data Designer recipe named {recipe_name!r}.")

    def _load_entry_point(self, entry_point: importlib.metadata.EntryPoint) -> DataDesignerRecipe:
        try:
            loaded = entry_point.load()
        except Exception as exc:
            raise RecipeLoadError(f"Failed to load recipe entry point {entry_point.name!r}: {exc}") from exc

        if not isinstance(loaded, DataDesignerRecipe):
            raise RecipeLoadError(
                f"Recipe entry point {entry_point.name!r} returned {type(loaded).__name__}, "
                "expected DataDesignerRecipe."
            )
        return loaded
