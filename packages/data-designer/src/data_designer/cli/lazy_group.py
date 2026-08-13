# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import importlib.metadata
from collections import defaultdict
from typing import Any

import click
import typer
from typer.core import TyperCommand, TyperGroup

CLI_EXTENSION_ENTRY_POINT_GROUP = "data_designer.cli"
_DATA_DESIGNER_DISTRIBUTION = "data-designer"


def _distribution_name(entry_point: importlib.metadata.EntryPoint) -> str:
    distribution = getattr(entry_point, "dist", None)
    if distribution is None:
        return "unknown-distribution"
    return distribution.metadata.get("Name") or "unknown-distribution"


def _distribution_version(entry_point: importlib.metadata.EntryPoint) -> str:
    distribution = getattr(entry_point, "dist", None)
    if distribution is None:
        return "unknown-version"
    return distribution.version


def _entry_point_label(entry_point: importlib.metadata.EntryPoint) -> str:
    return f"{_distribution_name(entry_point)} {_distribution_version(entry_point)} ({entry_point.value})"


def _entry_point_help(entry_point: importlib.metadata.EntryPoint) -> str:
    distribution = getattr(entry_point, "dist", None)
    if distribution is not None:
        summary = distribution.metadata.get("Summary")
        if summary:
            return summary
    return f"CLI extension provided by {_distribution_name(entry_point)}"


def _entry_point_sort_key(entry_point: importlib.metadata.EntryPoint) -> tuple[str, str, str, str, str]:
    distribution_name = _distribution_name(entry_point)
    return (
        entry_point.name,
        distribution_name.casefold(),
        distribution_name,
        _distribution_version(entry_point),
        entry_point.value,
    )


def _validate_entry_point_compatibility(entry_point: importlib.metadata.EntryPoint) -> None:
    packaging_requirements = importlib.import_module("packaging.requirements")
    packaging_utils = importlib.import_module("packaging.utils")

    distribution = getattr(entry_point, "dist", None)
    label = _entry_point_label(entry_point)
    if distribution is None:
        raise click.ClickException(f"CLI extension {entry_point.name!r} has no owning distribution: {label}.")

    try:
        requirements = [packaging_requirements.Requirement(value) for value in distribution.requires or []]
    except packaging_requirements.InvalidRequirement as e:
        raise click.ClickException(
            f"CLI extension {entry_point.name!r} from {label} has invalid dependency metadata: {e}."
        ) from None

    data_designer_requirements = [
        requirement
        for requirement in requirements
        if packaging_utils.canonicalize_name(requirement.name) == _DATA_DESIGNER_DISTRIBUTION
        and (requirement.marker is None or requirement.marker.evaluate())
    ]
    if not data_designer_requirements:
        raise click.ClickException(
            f"CLI extension {entry_point.name!r} from {label} must declare a dependency on data-designer."
        )

    try:
        installed_version = importlib.metadata.version(_DATA_DESIGNER_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError:
        raise click.ClickException("Unable to resolve the installed data-designer version.") from None

    if not all(installed_version in requirement.specifier for requirement in data_designer_requirements):
        expected = " and ".join(str(requirement) for requirement in data_designer_requirements)
        raise click.ClickException(
            f"CLI extension {entry_point.name!r} from {label} is incompatible with "
            f"data-designer {installed_version}; requires {expected}."
        )


class _LazyCommand(click.Command):
    """A click.Command stub that defers module loading until invocation.

    Stores only the command name and help text so that group-level ``--help``
    can list the command without importing its module.  The real Click command
    (produced by Typer from the decorated function) is resolved lazily on first
    ``make_context`` or ``invoke`` call.
    """

    def __init__(
        self,
        name: str,
        module_path: str,
        attr_name: str,
        *,
        rich_help_panel: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self._module_path = module_path
        self._attr_name = attr_name
        self._resolved: click.Command | None = None
        self.rich_help_panel = rich_help_panel

    def _resolve(self) -> click.Command:
        if self._resolved is not None:
            return self._resolved
        module = importlib.import_module(self._module_path)
        func = getattr(module, self._attr_name)
        temp_app = typer.Typer()
        temp_app.command(name=self.name)(func)
        click_cmd = typer.main.get_command(temp_app)
        # Typer returns a Group when there are multiple commands, but a single
        # Command when there is only one.  Handle both cases.
        if hasattr(click_cmd, "commands"):
            self._resolved = click_cmd.commands[self.name]
        else:
            self._resolved = click_cmd
        return self._resolved

    def make_context(
        self,
        info_name: str,
        args: list[str],
        parent: click.Context | None = None,
        **extra: Any,
    ) -> click.Context:
        return self._resolve().make_context(info_name, args, parent, **extra)


class _LazyEntryPointCommand(click.Command):
    def __init__(self, entry_point: importlib.metadata.EntryPoint) -> None:
        super().__init__(name=entry_point.name, help=_entry_point_help(entry_point))
        self._entry_point = entry_point
        self._resolved: click.Command | None = None
        self.rich_help_panel = "Extensions"

    def _resolve(self) -> click.Command:
        if self._resolved is not None:
            return self._resolved

        _validate_entry_point_compatibility(self._entry_point)
        label = _entry_point_label(self._entry_point)
        try:
            factory = self._entry_point.load()
        except Exception as e:
            raise click.ClickException(f"Failed to load CLI extension {self.name!r} from {label}: {e}.") from None
        if not callable(factory):
            raise click.ClickException(f"CLI extension {self.name!r} from {label} must load a zero-argument callable.")

        try:
            command = factory()
        except Exception as e:
            raise click.ClickException(f"Failed to create CLI extension {self.name!r} from {label}: {e}.") from None
        if not isinstance(command, (click.Command, TyperCommand, TyperGroup)):
            raise click.ClickException(
                f"CLI extension {self.name!r} from {label} returned {type(command).__name__}, expected click.Command."
            )

        command.name = self.name
        self._resolved = command
        return command

    def make_context(
        self,
        info_name: str,
        args: list[str],
        parent: click.Context | None = None,
        **extra: Any,
    ) -> click.Context:
        return self._resolve().make_context(info_name, args, parent, **extra)


class _UnavailableCommand(click.Command):
    def __init__(self, name: str, message: str) -> None:
        super().__init__(name=name, help=f"Unavailable: {message}")
        self._message = message
        self.rich_help_panel = "Extensions"

    def make_context(
        self,
        info_name: str,
        args: list[str],
        parent: click.Context | None = None,
        **extra: Any,
    ) -> click.Context:
        raise click.ClickException(self._message)


def create_lazy_typer_group(
    lazy_subcommands: dict[str, dict[str, str]],
    *,
    entry_point_group: str | None = None,
) -> type[TyperGroup]:
    """Factory that returns a ``TyperGroup`` subclass with lazy-loaded commands.

    ``list_commands`` includes lazy command names so that ``--help`` works
    without importing any command module.  ``get_command`` returns a lightweight
    ``_LazyCommand`` stub for lazy entries; the real Typer/Click command is only
    built when the stub is invoked.

    Args:
        lazy_subcommands: Mapping of command names to metadata dicts with keys:
            - ``module``: Dotted module path (e.g. ``data_designer.cli.commands.preview``)
            - ``attr``:   Function attribute name in the module (e.g. ``preview_command``)
            - ``help``:   (optional) Short help text for group listing
            - ``rich_help_panel``: (optional) Rich help panel name
        entry_point_group: Optional entry-point group for lazy top-level command
            extensions. Entry-point targets must be zero-argument callables that
            return a ``click.Command``.

    Returns:
        A ``TyperGroup`` subclass.
    """

    class LazyTyperGroup(TyperGroup):
        _extension_entry_points: dict[str, list[importlib.metadata.EntryPoint]] | None = None

        def _discover_extension_entry_points(self) -> dict[str, list[importlib.metadata.EntryPoint]]:
            if self._extension_entry_points is not None:
                return self._extension_entry_points
            if entry_point_group is None:
                self._extension_entry_points = {}
                return self._extension_entry_points

            try:
                entry_points = sorted(
                    importlib.metadata.entry_points(group=entry_point_group),
                    key=_entry_point_sort_key,
                )
            except Exception as e:
                click.echo(f"Warning: Failed to discover CLI extensions from {entry_point_group!r}: {e}.", err=True)
                self._extension_entry_points = {}
                return self._extension_entry_points

            discovered: defaultdict[str, list[importlib.metadata.EntryPoint]] = defaultdict(list)
            for entry_point in entry_points:
                if (
                    not entry_point.name
                    or entry_point.name.startswith("-")
                    or any(character.isspace() for character in entry_point.name)
                ):
                    click.echo(
                        f"Warning: Ignoring invalid CLI extension command name {entry_point.name!r} "
                        f"from {_entry_point_label(entry_point)}.",
                        err=True,
                    )
                    continue
                discovered[entry_point.name].append(entry_point)
            self._extension_entry_points = dict(discovered)
            return self._extension_entry_points

        def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
            if not args and self.no_args_is_help and not ctx.resilient_parsing:
                click.echo(ctx.get_help(), color=ctx.color)
                ctx.exit(0)
            return super().parse_args(ctx, args)

        def list_commands(self, ctx: click.Context) -> list[str]:
            eager = super().list_commands(ctx)
            lazy_names = [name for name in lazy_subcommands if name not in eager]
            built_in_names = set(eager) | set(lazy_names)
            extension_names = [name for name in self._discover_extension_entry_points() if name not in built_in_names]
            return eager + sorted(lazy_names) + sorted(extension_names)

        def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
            cmd = super().get_command(ctx, cmd_name)
            extension_entry_points = self._discover_extension_entry_points().get(cmd_name, [])
            if extension_entry_points and (cmd is not None or cmd_name in lazy_subcommands):
                providers = ", ".join(_entry_point_label(entry_point) for entry_point in extension_entry_points)
                return _UnavailableCommand(
                    cmd_name,
                    f"CLI extension command {cmd_name!r} from {providers} conflicts with a built-in command.",
                )
            if len(extension_entry_points) > 1:
                providers = ", ".join(_entry_point_label(entry_point) for entry_point in extension_entry_points)
                return _UnavailableCommand(
                    cmd_name,
                    f"CLI command {cmd_name!r} is provided by multiple extensions: {providers}.",
                )
            if cmd is not None:
                return cmd
            if cmd_name in lazy_subcommands:
                info = lazy_subcommands[cmd_name]
                return _LazyCommand(
                    name=cmd_name,
                    module_path=info["module"],
                    attr_name=info["attr"],
                    help=info.get("help"),
                    rich_help_panel=info.get("rich_help_panel"),
                )
            if extension_entry_points:
                return _LazyEntryPointCommand(extension_entry_points[0])
            return None

    return LazyTyperGroup
