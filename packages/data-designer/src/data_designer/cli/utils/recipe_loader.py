# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import importlib.metadata
import io
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click
import typer

from data_designer.cli.utils.config_loader import (
    ConfigLoadError,
    WorkflowHelpRequested,
    call_config_builder_function,
    load_config_builder,
)
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.script_params import DataDesignerScriptParams

RECIPE_ENTRY_POINT_GROUP = "data_designer.recipes"


@dataclass(frozen=True)
class RecipeSummary:
    """Summary of an installed Data Designer recipe."""

    name: str
    entry_point: str
    package: str | None
    version: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return {
            "name": self.name,
            "entry_point": self.entry_point,
            "package": self.package,
            "version": self.version,
        }


@dataclass(frozen=True)
class RecipeDetails:
    """Inspectable metadata for an installed Data Designer recipe."""

    summary: RecipeSummary
    description: str | None
    help_text: str | None
    arguments: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return {
            **self.summary.to_dict(),
            "description": self.description,
            "supports_structured_help": self.help_text is not None,
            "arguments": self.arguments,
        }


def load_recipe_config_builder(
    recipe_name: str,
    script_params: DataDesignerScriptParams | None = None,
) -> DataDesignerConfigBuilder:
    """Load a Data Designer recipe from an installed entry point.

    Recipe entry points may expose either a config builder callable or a string
    path to a local config source. Callables follow the same
    ``load_config_builder(params)`` protocol as Python config files.
    """
    entry_point = _get_recipe_entry_point(recipe_name)
    try:
        loaded = entry_point.load()
    except Exception as exc:
        raise ConfigLoadError(f"Failed to load recipe {recipe_name!r}: {exc}") from exc

    if isinstance(loaded, str | Path):
        return load_config_builder(str(loaded), script_params=script_params)

    if isinstance(loaded, typer.Typer):
        return _call_typer_recipe_app(loaded, recipe_name, script_params)

    if callable(loaded):
        return call_config_builder_function(loaded, f"recipe {recipe_name!r}", script_params)

    raise ConfigLoadError(
        f"Recipe {recipe_name!r} loaded {type(loaded).__name__}, "
        "expected a config builder callable, Typer app, or config path."
    )


def list_recipes() -> list[RecipeSummary]:
    """List installed Data Designer recipe entry points."""
    return [_entry_point_summary(entry_point) for entry_point in _recipe_entry_points()]


def get_recipe_details(recipe_name: str) -> RecipeDetails:
    """Return inspectable details for an installed recipe."""
    entry_point = _get_recipe_entry_point(recipe_name)
    summary = _entry_point_summary(entry_point)
    typer_app = load_recipe_typer_app(recipe_name)
    if typer_app is not None:
        command = typer.main.get_command(typer_app)
        return RecipeDetails(
            summary=summary,
            description=command.help,
            help_text=_click_help_text(command, recipe_name),
            arguments=_click_arguments(command),
        )

    parser = load_recipe_arg_parser(recipe_name)

    if parser is None:
        return RecipeDetails(
            summary=summary,
            description=None,
            help_text=None,
            arguments=[],
        )

    return RecipeDetails(
        summary=summary,
        description=parser.description,
        help_text=parser.format_help(),
        arguments=_argparse_arguments(parser),
    )


def get_recipe_help_text(recipe_name: str) -> str:
    """Return recipe-specific help text for an installed recipe."""
    details = get_recipe_details(recipe_name)
    if details.help_text is None:
        raise ConfigLoadError(
            f"Recipe {recipe_name!r} does not expose structured help. "
            "Recipe modules can define build_arg_parser() to support inspection."
        )
    return details.help_text


def load_recipe_typer_app(recipe_name: str) -> typer.Typer | None:
    """Load an optional Typer app exposed by a recipe module."""
    entry_point = _get_recipe_entry_point(recipe_name)
    loaded = _load_recipe_entry_point(entry_point, recipe_name)
    if isinstance(loaded, typer.Typer):
        return loaded

    module = _loaded_recipe_module(loaded)
    if module is None:
        return None

    build_typer_app = getattr(module, "build_typer_app", None)
    if build_typer_app is None:
        return None
    if not callable(build_typer_app):
        raise ConfigLoadError(f"Recipe {recipe_name!r} defines build_typer_app, but it is not callable.")

    app = build_typer_app()
    if not isinstance(app, typer.Typer):
        raise ConfigLoadError(
            f"Recipe {recipe_name!r} build_typer_app() returned {type(app).__name__}, expected Typer."
        )
    return app


def load_recipe_arg_parser(recipe_name: str) -> argparse.ArgumentParser | None:
    """Load an optional argparse parser exposed by a recipe module."""
    entry_point = _get_recipe_entry_point(recipe_name)
    loaded = _load_recipe_entry_point(entry_point, recipe_name)
    module = _loaded_recipe_module(loaded)
    if module is None:
        return None

    build_arg_parser = getattr(module, "build_arg_parser", None)
    if build_arg_parser is None:
        return None
    if not callable(build_arg_parser):
        raise ConfigLoadError(f"Recipe {recipe_name!r} defines build_arg_parser, but it is not callable.")

    parser = build_arg_parser()
    if not isinstance(parser, argparse.ArgumentParser):
        raise ConfigLoadError(
            f"Recipe {recipe_name!r} build_arg_parser() returned {type(parser).__name__}, expected ArgumentParser."
        )
    return parser


def _loaded_recipe_module(loaded: Any) -> Any | None:
    module_name = getattr(loaded, "__module__", None)
    if module_name is None:
        return None
    return sys.modules.get(module_name)


def _call_typer_recipe_app(
    app: typer.Typer,
    recipe_name: str,
    script_params: DataDesignerScriptParams | None,
) -> DataDesignerConfigBuilder:
    params = script_params or DataDesignerScriptParams()
    command = typer.main.get_command(app)
    try:
        config_builder = command.main(
            args=list(params.argv),
            prog_name=f"data-designer preview/create --recipe {recipe_name} --",
            standalone_mode=False,
        )
    except click.exceptions.Exit as exc:
        if exc.exit_code == 0:
            raise WorkflowHelpRequested from exc
        raise ConfigLoadError(f"Recipe {recipe_name!r} exited with code {exc.exit_code}") from exc
    except click.ClickException as exc:
        raise ConfigLoadError(f"Error parsing recipe {recipe_name!r} arguments: {exc.format_message()}") from exc

    if config_builder == 0 and any(arg in {"--help", "-h"} for arg in params.argv):
        raise WorkflowHelpRequested

    if not isinstance(config_builder, DataDesignerConfigBuilder):
        raise ConfigLoadError(
            f"Recipe {recipe_name!r} returned {type(config_builder).__name__}, expected DataDesignerConfigBuilder"
        )
    return config_builder


def _get_recipe_entry_point(recipe_name: str) -> importlib.metadata.EntryPoint:
    recipes = _recipe_entry_points()
    for entry_point in recipes:
        if entry_point.name == recipe_name:
            return entry_point

    available = ", ".join(sorted(entry_point.name for entry_point in recipes)) or "none"
    raise ConfigLoadError(
        f"No installed Data Designer recipe named {recipe_name!r}. "
        f"Expected an entry point in {RECIPE_ENTRY_POINT_GROUP!r}. Available recipes: {available}."
    )


def _recipe_entry_points() -> list[importlib.metadata.EntryPoint]:
    return sorted(
        importlib.metadata.entry_points(group=RECIPE_ENTRY_POINT_GROUP), key=lambda entry_point: entry_point.name
    )


def _load_recipe_entry_point(entry_point: importlib.metadata.EntryPoint, recipe_name: str) -> Any:
    try:
        return entry_point.load()
    except Exception as exc:
        raise ConfigLoadError(f"Failed to load recipe {recipe_name!r}: {exc}") from exc


def _entry_point_summary(entry_point: importlib.metadata.EntryPoint) -> RecipeSummary:
    distribution = getattr(entry_point, "dist", None)
    return RecipeSummary(
        name=entry_point.name,
        entry_point=getattr(entry_point, "value", ""),
        package=_distribution_metadata(distribution, "Name"),
        version=_distribution_metadata(distribution, "Version"),
    )


def _distribution_metadata(distribution: Any, key: str) -> str | None:
    if distribution is None:
        return None
    metadata = getattr(distribution, "metadata", None)
    if metadata is None:
        return None
    value = metadata.get(key)
    return str(value) if value is not None else None


def _click_help_text(command: click.Command, recipe_name: str) -> str:
    stdout = io.StringIO()
    with (
        redirect_stdout(stdout),
        click.Context(command, info_name=f"data-designer preview/create --recipe {recipe_name} --") as context,
    ):
        help_text = command.get_help(context)
    return help_text or stdout.getvalue()


def _click_arguments(command: click.Command) -> list[dict[str, Any]]:
    arguments: list[dict[str, Any]] = []
    for parameter in command.params:
        if getattr(parameter, "hidden", False):
            continue
        flags = []
        if isinstance(parameter, click.Option):
            flags = [*parameter.opts, *parameter.secondary_opts]
        arguments.append(
            {
                "name": parameter.name,
                "flags": flags,
                "required": bool(parameter.required),
                "default": _jsonable_default(parameter.default),
                "choices": _click_choices(parameter.type),
                "nargs": parameter.nargs,
                "help": getattr(parameter, "help", None),
            }
        )
    return arguments


def _click_choices(parameter_type: click.ParamType) -> list[str] | None:
    if isinstance(parameter_type, click.Choice):
        return list(parameter_type.choices)
    return None


def _argparse_arguments(parser: argparse.ArgumentParser) -> list[dict[str, Any]]:
    arguments: list[dict[str, Any]] = []
    for action in parser._actions:
        if action.help == argparse.SUPPRESS:
            continue
        if isinstance(action, argparse._HelpAction):
            continue
        arguments.append(
            {
                "name": action.dest,
                "flags": list(action.option_strings),
                "required": bool(getattr(action, "required", False)),
                "default": _jsonable_default(action.default),
                "choices": list(action.choices) if action.choices is not None else None,
                "nargs": action.nargs,
                "help": action.help,
            }
        )
    return arguments


def _jsonable_default(value: Any) -> Any:
    if value is argparse.SUPPRESS:
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple | set):
        return list(value)
    return value
