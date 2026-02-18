# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
from typing import Any

import click
import typer
from typer.core import TyperGroup


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

    def invoke(self, ctx: click.Context) -> Any:
        return self._resolve().invoke(ctx)


def create_lazy_typer_group(
    lazy_subcommands: dict[str, dict[str, str]],
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

    Returns:
        A ``TyperGroup`` subclass.
    """

    class LazyTyperGroup(TyperGroup):
        def list_commands(self, ctx: click.Context) -> list[str]:
            eager = super().list_commands(ctx)
            lazy_names = [name for name in lazy_subcommands if name not in eager]
            return eager + sorted(lazy_names)

        def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
            cmd = super().get_command(ctx, cmd_name)
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
            return None

    return LazyTyperGroup
