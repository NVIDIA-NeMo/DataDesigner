# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.metadata
from types import SimpleNamespace
from unittest.mock import Mock, patch

import click
import pytest
import typer
from typer.testing import CliRunner

from data_designer.cli.lazy_group import create_lazy_typer_group

ENTRY_POINT_GROUP = "test.data_designer.cli"
runner = CliRunner()


def _normalized_output(output: str) -> str:
    return " ".join(line.strip(" │") for line in click.unstyle(output).splitlines())


def _app() -> typer.Typer:
    app = typer.Typer(cls=create_lazy_typer_group({}, entry_point_group=ENTRY_POINT_GROUP))

    @app.callback()
    def callback() -> None:
        pass

    @app.command()
    def base() -> None:
        click.echo("base")

    return app


def _command(output: str = "extension") -> click.Command:
    @click.command()
    def run() -> None:
        click.echo(output)

    return click.Group(name="extension", commands={"run": run})


def _nested_command() -> click.Command:
    @click.command()
    @click.option("--value", required=True)
    @click.pass_context
    def init(ctx: click.Context, value: str) -> None:
        click.echo(value)
        ctx.exit(3)

    profile = click.Group(name="profile", commands={"init": init})
    return click.Group(name="extension", commands={"profile": profile})


def _entry_point(
    name: str = "slurm",
    *,
    distribution_name: str = "data-designer-slurm",
    requirements: list[str] | None = None,
    loaded: object | None = None,
    load_error: Exception | None = None,
) -> SimpleNamespace:
    distribution = SimpleNamespace(
        metadata={"Name": distribution_name, "Summary": f"{distribution_name} summary"},
        requires=requirements if requirements is not None else ["data-designer>=0"],
        version="1.0.0",
    )
    load = (
        Mock(side_effect=load_error)
        if load_error is not None
        else Mock(return_value=loaded or Mock(return_value=_command()))
    )
    return SimpleNamespace(
        name=name,
        value=f"{distribution_name}.cli:create_cli",
        group=ENTRY_POINT_GROUP,
        dist=distribution,
        load=load,
    )


def test_no_extensions_preserves_help_and_built_in_commands() -> None:
    app = _app()
    with patch.object(importlib.metadata, "entry_points", return_value=[]):
        help_result = runner.invoke(app, ["--help"])
        command_result = runner.invoke(app, ["base"])

    assert help_result.exit_code == 0
    assert "base" in help_result.output
    assert "slurm" not in help_result.output
    assert command_result.exit_code == 0
    assert command_result.output == "base\n"


def test_root_help_lists_extension_without_loading_it() -> None:
    entry_point = _entry_point()
    with patch.object(importlib.metadata, "entry_points", return_value=[entry_point]):
        result = runner.invoke(_app(), ["--help"])

    assert result.exit_code == 0
    assert "slurm" in result.output
    assert "data-designer-slurm summary" in result.output
    entry_point.load.assert_not_called()


def test_root_help_uses_fallback_when_distribution_summary_is_missing() -> None:
    entry_point = _entry_point()
    entry_point.dist.metadata.pop("Summary")
    with patch.object(importlib.metadata, "entry_points", return_value=[entry_point]):
        result = runner.invoke(_app(), ["--help"])

    assert result.exit_code == 0
    assert "CLI extension provided by data-designer-slurm" in result.output
    entry_point.load.assert_not_called()


def test_built_in_command_does_not_load_extension() -> None:
    entry_point = _entry_point()
    with patch.object(importlib.metadata, "entry_points", return_value=[entry_point]):
        result = runner.invoke(_app(), ["base"])

    assert result.exit_code == 0
    assert result.output == "base\n"
    entry_point.load.assert_not_called()


def test_selected_extension_loads_and_dispatches() -> None:
    factory = Mock(return_value=_command("selected"))
    entry_point = _entry_point(loaded=factory)
    with patch.object(importlib.metadata, "entry_points", return_value=[entry_point]):
        result = runner.invoke(_app(), ["slurm", "run"])

    assert result.exit_code == 0
    assert result.output == "selected\n"
    entry_point.load.assert_called_once_with()
    factory.assert_called_once_with()


def test_selected_extension_dispatches_nested_command_and_preserves_exit_code() -> None:
    factory = Mock(return_value=_nested_command())
    entry_point = _entry_point(loaded=factory)
    with patch.object(importlib.metadata, "entry_points", return_value=[entry_point]):
        result = runner.invoke(_app(), ["slurm", "profile", "init", "--value", "created"])

    assert result.exit_code == 3
    assert result.output == "created\n"
    entry_point.load.assert_called_once_with()
    factory.assert_called_once_with()


def test_selecting_one_extension_does_not_load_another() -> None:
    alpha = _entry_point("alpha", distribution_name="alpha-extension")
    beta = _entry_point("beta", distribution_name="beta-extension")
    with patch.object(importlib.metadata, "entry_points", return_value=[beta, alpha]):
        result = runner.invoke(_app(), ["alpha", "run"])

    assert result.exit_code == 0
    alpha.load.assert_called_once_with()
    beta.load.assert_not_called()


def test_broken_extension_fails_only_when_selected() -> None:
    entry_point = _entry_point(load_error=ImportError("missing dependency"))
    app = _app()
    with patch.object(importlib.metadata, "entry_points", return_value=[entry_point]):
        help_result = runner.invoke(app, ["--help"])
        built_in_result = runner.invoke(app, ["base"])
        command_result = runner.invoke(app, ["slurm"])

    assert help_result.exit_code == 0
    assert built_in_result.exit_code == 0
    assert built_in_result.output == "base\n"
    assert command_result.exit_code == 1
    assert "Failed to load CLI extension 'slurm'" in command_result.output
    assert "data-designer-slurm 1.0.0" in command_result.output
    assert "missing dependency" in command_result.output


def test_extension_target_must_be_callable() -> None:
    entry_point = _entry_point(loaded=object())
    with patch.object(importlib.metadata, "entry_points", return_value=[entry_point]):
        result = runner.invoke(_app(), ["slurm"])

    assert result.exit_code == 1
    assert "must load a zero-argument callable" in result.output


def test_extension_factory_must_return_click_command() -> None:
    entry_point = _entry_point(loaded=Mock(return_value=object()))
    with patch.object(importlib.metadata, "entry_points", return_value=[entry_point]):
        result = runner.invoke(_app(), ["slurm"])

    assert result.exit_code == 1
    assert "returned object" in result.output
    assert "click.Command" in result.output


def test_extension_factory_error_has_distribution_context() -> None:
    entry_point = _entry_point(loaded=Mock(side_effect=RuntimeError("factory failed")))
    with patch.object(importlib.metadata, "entry_points", return_value=[entry_point]):
        result = runner.invoke(_app(), ["slurm"])

    assert result.exit_code == 1
    assert "Failed to create CLI extension 'slurm'" in result.output
    assert "data-designer-slurm 1.0.0" in result.output
    assert "factory failed" in result.output


def test_extension_without_owning_distribution_fails_before_import() -> None:
    entry_point = _entry_point()
    entry_point.dist = None
    with patch.object(importlib.metadata, "entry_points", return_value=[entry_point]):
        result = runner.invoke(_app(), ["slurm"])

    assert result.exit_code == 1
    assert "has no owning distribution" in result.output
    entry_point.load.assert_not_called()


def test_invalid_dependency_metadata_fails_before_import() -> None:
    entry_point = _entry_point(requirements=["data-designer=>1"])
    with patch.object(importlib.metadata, "entry_points", return_value=[entry_point]):
        result = runner.invoke(_app(), ["slurm"])

    assert result.exit_code == 1
    assert "has invalid dependency metadata" in result.output
    entry_point.load.assert_not_called()


def test_missing_base_distribution_version_fails_before_import() -> None:
    entry_point = _entry_point()
    with (
        patch.object(importlib.metadata, "entry_points", return_value=[entry_point]),
        patch.object(
            importlib.metadata,
            "version",
            side_effect=importlib.metadata.PackageNotFoundError("data-designer"),
        ),
    ):
        result = runner.invoke(_app(), ["slurm"])

    assert result.exit_code == 1
    assert "Unable to resolve the installed data-designer version" in result.output
    entry_point.load.assert_not_called()


def test_duplicate_extensions_fail_deterministically_without_loading() -> None:
    alpha = _entry_point(distribution_name="alpha-extension")
    beta = _entry_point(distribution_name="beta-extension")
    outputs = []

    for entry_points in ([beta, alpha], [alpha, beta]):
        with patch.object(importlib.metadata, "entry_points", return_value=entry_points):
            result = runner.invoke(_app(), ["slurm"])
        assert result.exit_code == 1
        outputs.append(result.output)

    assert outputs[0] == outputs[1]
    assert "CLI command 'slurm' is provided by multiple extensions" in outputs[0]
    assert outputs[0].index("alpha-extension") < outputs[0].index("beta-extension")
    alpha.load.assert_not_called()
    beta.load.assert_not_called()


def test_extension_cannot_replace_built_in_command() -> None:
    entry_point = _entry_point("base")
    with patch.object(importlib.metadata, "entry_points", return_value=[entry_point]):
        help_result = runner.invoke(_app(), ["--help"])
        command_result = runner.invoke(_app(), ["base"])

    assert help_result.exit_code == 0
    assert "Unavailable" in help_result.output
    assert command_result.exit_code == 1
    assert "conflicts with a built-in command" in command_result.output
    entry_point.load.assert_not_called()


@pytest.mark.parametrize(
    ("requirements", "message"),
    [
        ([], "must declare a dependency on data-designer"),
        (["data-designer<0"], "is incompatible with data-designer"),
        (["data-designer>=0", "data-designer<0"], "is incompatible with data-designer"),
    ],
)
def test_incompatible_extension_fails_before_import(requirements: list[str], message: str) -> None:
    entry_point = _entry_point(requirements=requirements)
    with patch.object(importlib.metadata, "entry_points", return_value=[entry_point]):
        result = runner.invoke(_app(), ["slurm"])

    assert result.exit_code == 1
    assert message in _normalized_output(result.output)
    entry_point.load.assert_not_called()


def test_invalid_extension_name_is_ignored_without_breaking_built_in_commands() -> None:
    entry_point = _entry_point("bad name")
    app = _app()
    with patch.object(importlib.metadata, "entry_points", return_value=[entry_point]):
        help_result = runner.invoke(app, ["--help"])
        command_result = runner.invoke(app, ["base"])

    assert help_result.exit_code == 0
    assert "Ignoring invalid CLI extension command name 'bad name'" in help_result.output
    assert "data-designer-slurm 1.0.0" in help_result.output
    assert command_result.exit_code == 0
    assert command_result.output.endswith("base\n")
    entry_point.load.assert_not_called()


def test_discovery_error_warns_without_breaking_built_in_commands() -> None:
    app = _app()
    with patch.object(importlib.metadata, "entry_points", side_effect=RuntimeError("corrupt metadata")):
        help_result = runner.invoke(app, ["--help"])
        command_result = runner.invoke(app, ["base"])

    assert help_result.exit_code == 0
    assert f"Failed to discover CLI extensions from {ENTRY_POINT_GROUP!r}: corrupt metadata" in help_result.output
    assert command_result.exit_code == 0
    assert command_result.output.endswith("base\n")
