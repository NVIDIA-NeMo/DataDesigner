# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import ModuleType
from typing import Annotated, Any
from unittest.mock import MagicMock, patch

import click
import pytest
import typer

from data_designer.cli.utils.config_loader import ConfigLoadError
from data_designer.cli.utils.recipe_loader import get_recipe_details, get_recipe_help_text, load_recipe_config_builder
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.script_params import DataDesignerScriptParams


class FakeEntryPoint:
    """Minimal entry point stub."""

    def __init__(self, name: str, loaded: Any):
        self.name = name
        self._loaded = loaded

    def load(self) -> Any:
        return self._loaded


def build_arg_parser() -> argparse.ArgumentParser:
    """Build a parser used by recipe inspection tests."""
    parser = argparse.ArgumentParser(
        prog="demo",
        description="Demo recipe.",
    )
    parser.add_argument("--seed-path", required=True, help="Seed dataset path.")
    parser.add_argument("--variant", default="compact", choices=["compact", "verbose"], help="Prompt variant.")
    return parser


def demo_recipe(params: DataDesignerScriptParams) -> DataDesignerConfigBuilder:
    """Demo recipe callable."""
    builder = DataDesignerConfigBuilder()
    builder._test_argv = params.argv
    return builder


def make_typer_recipe_module(module_name: str) -> tuple[ModuleType, Any]:
    """Create a temporary module exposing a Typer recipe app."""
    module = ModuleType(module_name)

    def recipe_command(
        seed_path: Annotated[Path, typer.Option("--seed-path", help="Seed dataset path.")],
        variant: Annotated[
            str,
            typer.Option(
                "--variant",
                help="Prompt variant.",
                click_type=click.Choice(["compact", "verbose"]),
            ),
        ] = "compact",
    ) -> DataDesignerConfigBuilder:
        builder = DataDesignerConfigBuilder()
        builder._test_seed_path = seed_path
        builder._test_variant = variant
        return builder

    def build_typer_app() -> typer.Typer:
        app = typer.Typer(add_completion=False, help="Demo Typer recipe.")
        app.command(name=None, help="Build the demo recipe.")(recipe_command)
        return app

    def recipe(params: DataDesignerScriptParams) -> DataDesignerConfigBuilder:
        command = typer.main.get_command(build_typer_app())
        return command.main(args=list(params.argv), standalone_mode=False)

    recipe.__module__ = module_name
    recipe_command.__module__ = module_name
    build_typer_app.__module__ = module_name
    module.recipe = recipe
    module.recipe_command = recipe_command
    module.build_typer_app = build_typer_app
    sys.modules[module_name] = module
    return module, recipe


def test_load_recipe_config_builder_loads_callable_recipe() -> None:
    """Test that a recipe entry point can expose a config builder callable."""
    entry_point = FakeEntryPoint("demo", demo_recipe)

    with patch("data_designer.cli.utils.recipe_loader.importlib.metadata.entry_points", return_value=[entry_point]):
        result = load_recipe_config_builder(
            "demo",
            script_params=DataDesignerScriptParams(argv=("--seed-path", "seed.jsonl")),
        )

    assert isinstance(result, DataDesignerConfigBuilder)
    assert result._test_argv == ("--seed-path", "seed.jsonl")


def test_get_recipe_details_reads_argparse_metadata() -> None:
    """Test that recipe details include parser-derived argument metadata."""
    entry_point = FakeEntryPoint("demo", demo_recipe)

    with patch("data_designer.cli.utils.recipe_loader.importlib.metadata.entry_points", return_value=[entry_point]):
        details = get_recipe_details("demo")

    assert details.description == "Demo recipe."
    assert details.help_text is not None
    assert "Seed dataset path." in details.help_text
    assert details.arguments[0]["name"] == "seed_path"
    assert details.arguments[0]["flags"] == ["--seed-path"]
    assert details.arguments[0]["required"] is True
    assert details.arguments[1]["choices"] == ["compact", "verbose"]


def test_get_recipe_details_reads_typer_metadata() -> None:
    """Test that recipe details can include Typer-derived argument metadata."""
    module, recipe = make_typer_recipe_module("test_typer_recipe_details")
    entry_point = FakeEntryPoint("demo", recipe)

    try:
        with patch("data_designer.cli.utils.recipe_loader.importlib.metadata.entry_points", return_value=[entry_point]):
            details = get_recipe_details("demo")
    finally:
        sys.modules.pop(module.__name__, None)

    assert details.description == "Build the demo recipe."
    assert details.help_text is not None
    assert "Seed dataset path." in details.help_text
    assert details.arguments[0]["name"] == "seed_path"
    assert details.arguments[0]["flags"] == ["--seed-path"]
    assert details.arguments[0]["required"] is True
    assert details.arguments[1]["choices"] == ["compact", "verbose"]


def test_get_recipe_help_text_returns_argparse_help() -> None:
    """Test that recipe help returns parser-formatted help."""
    entry_point = FakeEntryPoint("demo", demo_recipe)

    with patch("data_designer.cli.utils.recipe_loader.importlib.metadata.entry_points", return_value=[entry_point]):
        help_text = get_recipe_help_text("demo")

    assert "usage: demo" in help_text
    assert "--seed-path" in help_text


def test_load_recipe_config_builder_loads_typer_app_entry_point() -> None:
    """Test that a recipe entry point can expose a Typer app directly."""
    module, _ = make_typer_recipe_module("test_typer_recipe_entry_point")
    entry_point = FakeEntryPoint("demo", module.build_typer_app())

    try:
        with patch("data_designer.cli.utils.recipe_loader.importlib.metadata.entry_points", return_value=[entry_point]):
            result = load_recipe_config_builder(
                "demo",
                script_params=DataDesignerScriptParams(argv=("--seed-path", "seed.jsonl", "--variant", "verbose")),
            )
    finally:
        sys.modules.pop(module.__name__, None)

    assert isinstance(result, DataDesignerConfigBuilder)
    assert result._test_seed_path == Path("seed.jsonl")
    assert result._test_variant == "verbose"


@patch("data_designer.cli.utils.recipe_loader.load_config_builder")
def test_load_recipe_config_builder_loads_config_path(mock_load_config_builder: MagicMock) -> None:
    """Test that a recipe entry point can expose a config source path."""
    mock_builder = MagicMock(spec=DataDesignerConfigBuilder)
    mock_load_config_builder.return_value = mock_builder
    script_params = DataDesignerScriptParams(argv=("--seed-path", "seed.jsonl"))
    entry_point = FakeEntryPoint("demo", Path("workflow.py"))

    with patch("data_designer.cli.utils.recipe_loader.importlib.metadata.entry_points", return_value=[entry_point]):
        result = load_recipe_config_builder("demo", script_params=script_params)

    mock_load_config_builder.assert_called_once_with("workflow.py", script_params=script_params)
    assert result is mock_builder


def test_load_recipe_config_builder_errors_for_missing_recipe() -> None:
    """Test that an unknown recipe produces a clear load error."""
    with patch("data_designer.cli.utils.recipe_loader.importlib.metadata.entry_points", return_value=[]):
        with pytest.raises(ConfigLoadError, match="No installed Data Designer recipe named 'missing'"):
            load_recipe_config_builder("missing")


def test_load_recipe_config_builder_rejects_invalid_entry_point_object() -> None:
    """Test that recipes must expose a callable or config source path."""
    entry_point = FakeEntryPoint("bad", object())

    with patch("data_designer.cli.utils.recipe_loader.importlib.metadata.entry_points", return_value=[entry_point]):
        with pytest.raises(ConfigLoadError, match="expected a config builder callable, Typer app, or config path"):
            load_recipe_config_builder("bad")
