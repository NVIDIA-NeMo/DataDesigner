# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import click
import typer

from data_designer.cli.controllers.plugin_catalog_controller import PluginCatalogController
from data_designer.config.utils.constants import DATA_DESIGNER_HOME


def list_command(
    ctx: typer.Context,
    tap: str | None = typer.Option(
        None,
        "--tap",
        help="Plugin tap alias to read. Can also be provided before the subcommand.",
    ),
    refresh: bool = typer.Option(
        False,
        "--refresh",
        help="Fetch the tap catalog even when a fresh cache entry exists.",
    ),
    include_incompatible: bool = typer.Option(
        False,
        "--include-incompatible",
        help="Show plugins that do not satisfy the local Python or Data Designer version.",
    ),
) -> None:
    """List discoverable Data Designer plugins from a tap catalog."""
    controller = PluginCatalogController(DATA_DESIGNER_HOME)
    controller.run_list(
        tap_alias=_resolve_tap_alias(ctx, tap),
        refresh=refresh,
        include_incompatible=include_incompatible,
    )


def search_command(
    ctx: typer.Context,
    query: str = typer.Argument(help="Keyword, plugin type, package name, source, maintainer, or tag to search for."),
    tap: str | None = typer.Option(
        None,
        "--tap",
        help="Plugin tap alias to search. Can also be provided before the subcommand.",
    ),
    refresh: bool = typer.Option(
        False,
        "--refresh",
        help="Fetch the tap catalog even when a fresh cache entry exists.",
    ),
    include_incompatible: bool = typer.Option(
        False,
        "--include-incompatible",
        help="Search plugins that do not satisfy the local Python or Data Designer version.",
    ),
) -> None:
    """Search discoverable Data Designer plugins from a tap catalog."""
    controller = PluginCatalogController(DATA_DESIGNER_HOME)
    controller.run_search(
        query,
        tap_alias=_resolve_tap_alias(ctx, tap),
        refresh=refresh,
        include_incompatible=include_incompatible,
    )


def info_command(
    ctx: typer.Context,
    plugin_name: str = typer.Argument(help="Plugin name from the tap catalog."),
    tap: str | None = typer.Option(
        None,
        "--tap",
        help="Plugin tap alias to read. Can also be provided before the subcommand.",
    ),
    refresh: bool = typer.Option(
        False,
        "--refresh",
        help="Fetch the tap catalog even when a fresh cache entry exists.",
    ),
) -> None:
    """Show metadata, compatibility, docs, and install plan for one plugin."""
    controller = PluginCatalogController(DATA_DESIGNER_HOME)
    controller.run_info(
        plugin_name,
        tap_alias=_resolve_tap_alias(ctx, tap),
        refresh=refresh,
    )


def install_command(
    ctx: typer.Context,
    plugin_name: str = typer.Argument(help="Plugin name from the tap catalog."),
    tap: str | None = typer.Option(
        None,
        "--tap",
        help="Plugin tap alias to install from. Can also be provided before the subcommand.",
    ),
    refresh: bool = typer.Option(
        False,
        "--refresh",
        help="Fetch the tap catalog even when a fresh cache entry exists.",
    ),
    manager: str = typer.Option(
        "auto",
        "--manager",
        click_type=click.Choice(["auto", "uv", "pip"]),
        help="Package manager to use for installation.",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Install without an interactive confirmation prompt.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print the install plan without mutating the current environment.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Allow installation even when catalog compatibility checks fail.",
    ),
) -> None:
    """Install one Data Designer plugin package, then verify runtime discovery."""
    controller = PluginCatalogController(DATA_DESIGNER_HOME)
    controller.run_install(
        plugin_name,
        tap_alias=_resolve_tap_alias(ctx, tap),
        refresh=refresh,
        manager=manager,
        yes=yes,
        dry_run=dry_run,
        force=force,
    )


def installed_command() -> None:
    """List installed Data Designer plugins discovered from runtime entry points."""
    controller = PluginCatalogController(DATA_DESIGNER_HOME)
    controller.run_installed()


def taps_list_command() -> None:
    """List configured plugin taps."""
    controller = PluginCatalogController(DATA_DESIGNER_HOME)
    controller.run_taps_list()


def taps_add_command(
    alias: str = typer.Argument(help="Local alias for the plugin tap."),
    url: str = typer.Argument(help="Tap repository URL, catalog URL, local catalog file, or local tap directory."),
    trusted: bool = typer.Option(
        False,
        "--trusted",
        help="Mark the tap as trusted for install-plan display and confirmations.",
    ),
    cache_ttl_seconds: int = typer.Option(
        24 * 60 * 60,
        "--cache-ttl-seconds",
        min=0,
        help="Seconds before cached catalog metadata is refreshed. Use 0 to always refresh.",
    ),
) -> None:
    """Add a plugin tap alias."""
    controller = PluginCatalogController(DATA_DESIGNER_HOME)
    controller.run_taps_add(
        alias=alias,
        url=url,
        trusted=trusted,
        cache_ttl_seconds=cache_ttl_seconds,
    )


def taps_remove_command(
    alias: str = typer.Argument(help="Plugin tap alias to remove."),
) -> None:
    """Remove a plugin tap alias."""
    controller = PluginCatalogController(DATA_DESIGNER_HOME)
    controller.run_taps_remove(alias=alias)


def _resolve_tap_alias(ctx: typer.Context, tap_alias: str | None) -> str | None:
    if tap_alias is not None:
        return tap_alias

    parent = ctx.parent
    while parent is not None:
        candidate = parent.params.get("tap") if parent.params else None
        if isinstance(candidate, str) and candidate:
            return candidate
        parent = parent.parent
    return None
