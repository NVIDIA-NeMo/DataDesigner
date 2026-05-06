# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import click
import typer
from pydantic import BaseModel, ValidationError

from data_designer.cli.ui import console, print_error, print_header, print_success
from data_designer.config.errors import InvalidConfigError
from data_designer.config.utils.constants import DEFAULT_NUM_RECORDS
from data_designer.config.utils.io_helpers import smart_load_yaml
from data_designer.interface import DataDesigner
from data_designer.recipes.recipe import DataDesignerRecipe
from data_designer.recipes.registry import RecipeLoadError, RecipeRegistry


def list_command() -> None:
    """List installed Data Designer recipes."""
    try:
        recipes = RecipeRegistry().list_recipes()
    except RecipeLoadError as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from exc

    print_header("Installed Data Designer Recipes")
    if not recipes:
        console.print("  No recipes found.")
        return

    for item in recipes:
        console.print(f"  [bold]{item.entry_point_name}[/bold] — {item.recipe.description}")


def show_command(
    recipe_name: str = typer.Argument(..., help="Installed recipe name."),
) -> None:
    """Show a recipe's metadata and config schema."""
    try:
        recipe = RecipeRegistry().get_recipe(recipe_name)
    except RecipeLoadError as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from exc

    print_header(f"Recipe: {recipe.name}")
    console.print(f"  Description: {recipe.description}")
    console.print("  Config schema:")
    console.print_json(data=recipe.config_model.model_json_schema())


def run_recipe_command(
    recipe_name: str = typer.Argument(..., help="Installed recipe name."),
    config_path: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="YAML/JSON recipe configuration file. Omit for recipes with an empty config model.",
    ),
    mode: str = typer.Option(
        "create",
        "--mode",
        click_type=click.Choice(["create", "preview", "validate"], case_sensitive=False),
        help="Execution mode.",
    ),
    num_records: int = typer.Option(
        DEFAULT_NUM_RECORDS,
        "--num-records",
        "-n",
        help="Number of records to generate.",
        min=1,
    ),
    dataset_name: str = typer.Option(
        "dataset",
        "--dataset-name",
        "-d",
        help="Name for the generated dataset folder when --mode=create.",
    ),
    artifact_path: Path | None = typer.Option(
        None,
        "--artifact-path",
        "-o",
        help="Path where generated artifacts will be stored. Defaults to ./artifacts.",
    ),
) -> None:
    """Run an installed Data Designer recipe."""
    try:
        recipe = RecipeRegistry().get_recipe(recipe_name)
        recipe_config = _load_recipe_config(recipe, config_path)
        config_builder = recipe.build_config(recipe_config)
    except RecipeLoadError as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        print_error(f"Failed to build recipe {recipe_name!r}: {exc}")
        raise typer.Exit(code=1) from exc

    resolved_artifact_path = artifact_path or Path.cwd() / "artifacts"

    print_header("Data Designer Recipe")
    console.print(f"  Recipe: [bold]{recipe.name}[/bold]")
    console.print(f"  Mode: [bold]{mode}[/bold]")
    console.print(f"  Records: [bold]{num_records}[/bold]")
    console.print(f"  Artifact path: [bold]{resolved_artifact_path}[/bold]")
    if config_path is not None:
        console.print(f"  Config: [bold]{config_path}[/bold]")
    console.print()

    data_designer = DataDesigner(artifact_path=resolved_artifact_path)
    try:
        if mode == "validate":
            data_designer.validate(config_builder)
            print_success("Recipe configuration is valid")
            return

        if mode == "preview":
            results = data_designer.preview(config_builder, num_records=num_records)
            results.display_sample_record(index=0)
            print_success(f"Recipe preview complete — {len(results.dataset)} record(s) generated")
            return

        results = data_designer.create(config_builder, num_records=num_records, dataset_name=dataset_name)
        if recipe.postprocess is not None:
            recipe.postprocess(results, recipe_config)
        print_success(f"Recipe create complete — {len(results.load_dataset())} record(s) generated")
    except InvalidConfigError as exc:
        print_error(f"Recipe configuration is invalid: {exc}")
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        print_error(f"Recipe execution failed: {exc}")
        raise typer.Exit(code=1) from exc


def _load_recipe_config(recipe: DataDesignerRecipe, config_path: Path | None) -> BaseModel:
    """Load and validate a recipe config file."""
    raw_config = {}
    if config_path is not None:
        try:
            raw_config = smart_load_yaml(config_path)
        except Exception as exc:
            raise RecipeLoadError(f"Failed to load recipe config {config_path}: {exc}") from exc

    if raw_config is None:
        raw_config = {}
    if not isinstance(raw_config, dict):
        raise RecipeLoadError(f"Recipe config for {recipe.name!r} must be a mapping, got {type(raw_config).__name__}.")

    try:
        return recipe.config_model.model_validate(raw_config)
    except ValidationError as exc:
        raise RecipeLoadError(f"Invalid config for recipe {recipe.name!r}: {exc}") from exc
