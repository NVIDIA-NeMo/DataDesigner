# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Annotated

import click
import typer

from data_designer.cli.ui import console, print_error, print_header
from data_designer.cli.utils.config_loader import ConfigLoadError
from data_designer.cli.utils.recipe_loader import get_recipe_details, get_recipe_help_text, list_recipes

OutputFormat = click.Choice(["text", "json"], case_sensitive=False)


def list_command(
    output: Annotated[
        str,
        typer.Option(
            "--output",
            "-o",
            click_type=OutputFormat,
            help="Output format.",
        ),
    ] = "text",
) -> None:
    """List installed Data Designer recipes."""
    recipes = list_recipes()
    if output == "json":
        console.print_json(
            data={
                "schema_version": "data-designer.recipes.list.v1",
                "recipes": [recipe.to_dict() for recipe in recipes],
            }
        )
        return

    print_header("Installed Data Designer Recipes")
    if not recipes:
        console.print("  No recipes found.")
        return

    for recipe in recipes:
        package = recipe.package or "unknown package"
        version = f" {recipe.version}" if recipe.version is not None else ""
        console.print(f"  [bold]{recipe.name}[/bold] ({package}{version})")


def show_command(
    recipe_name: Annotated[str, typer.Argument(help="Installed recipe name.")],
    output: Annotated[
        str,
        typer.Option(
            "--output",
            "-o",
            click_type=OutputFormat,
            help="Output format.",
        ),
    ] = "text",
) -> None:
    """Show metadata for an installed Data Designer recipe."""
    try:
        details = get_recipe_details(recipe_name)
    except ConfigLoadError as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from exc

    if output == "json":
        console.print_json(
            data={
                "schema_version": "data-designer.recipes.show.v1",
                "recipe": details.to_dict(),
            }
        )
        return

    print_header(f"Recipe: {details.summary.name}")
    console.print(f"  Entry point: [bold]{details.summary.entry_point}[/bold]")
    if details.summary.package is not None:
        version = f" {details.summary.version}" if details.summary.version is not None else ""
        console.print(f"  Package: [bold]{details.summary.package}{version}[/bold]")
    if details.description is not None:
        console.print(f"  Description: {details.description}")

    if not details.arguments:
        console.print("  Structured argument metadata: unavailable")
        return

    console.print("  Arguments:")
    for argument in details.arguments:
        flags = ", ".join(argument["flags"]) if argument["flags"] else argument["name"]
        required = " required" if argument["required"] else ""
        default = "" if argument["default"] is None else f" default={argument['default']!r}"
        help_text = "" if argument["help"] is None else f" — {argument['help']}"
        console.print(f"    [bold]{flags}[/bold]{required}{default}{help_text}")


def help_command(
    recipe_name: Annotated[str, typer.Argument(help="Installed recipe name.")],
) -> None:
    """Show recipe-specific workflow argument help."""
    try:
        console.print(get_recipe_help_text(recipe_name), markup=False)
    except ConfigLoadError as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from exc
