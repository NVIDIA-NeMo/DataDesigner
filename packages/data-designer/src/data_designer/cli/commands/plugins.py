# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import click
import typer

from data_designer.cli.controllers.plugin_catalog_controller import PluginCatalogController
from data_designer.cli.ui import print_info
from data_designer.config.utils.constants import DATA_DESIGNER_HOME


def list_command(
    ctx: typer.Context,
    catalog: str | None = typer.Option(
        None,
        "--catalog",
        help="Plugin catalog alias to read. Can also be provided before the subcommand.",
    ),
    refresh: bool = typer.Option(
        False,
        "--refresh",
        help="Fetch the catalog even when a fresh cache entry exists.",
    ),
    include_incompatible: bool = typer.Option(
        False,
        "--include-incompatible",
        help="Show plugins that do not satisfy the local Python or Data Designer version.",
    ),
) -> None:
    """List discoverable Data Designer plugins from a catalog."""
    controller = PluginCatalogController(DATA_DESIGNER_HOME)
    controller.run_list(
        catalog_alias=_resolve_catalog_alias(ctx, catalog),
        refresh=refresh,
        include_incompatible=include_incompatible,
    )


def search_command(
    ctx: typer.Context,
    query: str = typer.Argument(
        help="Keyword, plugin type, package name, requirement, docs URL, or entry point to search for."
    ),
    catalog: str | None = typer.Option(
        None,
        "--catalog",
        help="Plugin catalog alias to search. Can also be provided before the subcommand.",
    ),
    refresh: bool = typer.Option(
        False,
        "--refresh",
        help="Fetch the catalog even when a fresh cache entry exists.",
    ),
    include_incompatible: bool = typer.Option(
        False,
        "--include-incompatible",
        help="Search plugins that do not satisfy the local Python or Data Designer version.",
    ),
) -> None:
    """Search discoverable Data Designer plugins from a catalog."""
    controller = PluginCatalogController(DATA_DESIGNER_HOME)
    controller.run_search(
        query,
        catalog_alias=_resolve_catalog_alias(ctx, catalog),
        refresh=refresh,
        include_incompatible=include_incompatible,
    )


def info_command(
    ctx: typer.Context,
    plugin_name: str = typer.Argument(help="Runtime plugin name or package name from the catalog."),
    catalog: str | None = typer.Option(
        None,
        "--catalog",
        help="Plugin catalog alias to read. Can also be provided before the subcommand.",
    ),
    refresh: bool = typer.Option(
        False,
        "--refresh",
        help="Fetch the catalog even when a fresh cache entry exists.",
    ),
) -> None:
    """Show metadata, compatibility, docs, and install plan for one plugin package."""
    controller = PluginCatalogController(DATA_DESIGNER_HOME)
    controller.run_info(
        plugin_name,
        catalog_alias=_resolve_catalog_alias(ctx, catalog),
        refresh=refresh,
    )


def install_command(
    ctx: typer.Context,
    plugin_name: str = typer.Argument(help="Runtime plugin name or package name from the catalog."),
    catalog: str | None = typer.Option(
        None,
        "--catalog",
        help="Plugin catalog alias to install from. Can also be provided before the subcommand.",
    ),
    refresh: bool = typer.Option(
        False,
        "--refresh",
        help="Fetch the catalog even when a fresh cache entry exists.",
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
        help="Allow installation when only incompatible catalog entries are available.",
    ),
) -> None:
    """Install one Data Designer plugin package, then verify runtime discovery."""
    controller = PluginCatalogController(DATA_DESIGNER_HOME)
    controller.run_install(
        plugin_name,
        catalog_alias=_resolve_catalog_alias(ctx, catalog),
        refresh=refresh,
        manager=manager,
        yes=yes,
        dry_run=dry_run,
        force=force,
    )


def installed_command(ctx: typer.Context) -> None:
    """List installed Data Designer plugin entry points."""
    _warn_if_parent_catalog_unused(ctx, "installed plugins are discovered from the current Python environment")
    controller = PluginCatalogController(DATA_DESIGNER_HOME)
    controller.run_installed()


def catalogs_list_command(ctx: typer.Context) -> None:
    """List configured plugin catalogs."""
    _warn_if_parent_catalog_unused(ctx, "catalog management commands operate on aliases directly")
    controller = PluginCatalogController(DATA_DESIGNER_HOME)
    controller.run_catalogs_list()


def catalogs_add_command(
    ctx: typer.Context,
    alias: str = typer.Argument(help="Local alias for the plugin catalog."),
    url: str = typer.Argument(
        help="Catalog repository URL, catalog URL, local catalog file, or local catalog directory."
    ),
    trusted: bool = typer.Option(
        False,
        "--trusted",
        help="Mark the catalog as trusted for install-plan display and confirmations.",
    ),
    cache_ttl_seconds: int = typer.Option(
        24 * 60 * 60,
        "--cache-ttl-seconds",
        min=0,
        help="Seconds before cached catalog metadata is refreshed. Use 0 to always refresh.",
    ),
) -> None:
    """Add a plugin catalog alias."""
    _warn_if_parent_catalog_unused(ctx, "catalog management commands operate on aliases directly")
    controller = PluginCatalogController(DATA_DESIGNER_HOME)
    controller.run_catalogs_add(
        alias=alias,
        url=url,
        trusted=trusted,
        cache_ttl_seconds=cache_ttl_seconds,
    )


def catalogs_remove_command(
    ctx: typer.Context,
    alias: str = typer.Argument(help="Plugin catalog alias to remove."),
) -> None:
    """Remove a plugin catalog alias."""
    _warn_if_parent_catalog_unused(ctx, "catalog management commands operate on aliases directly")
    controller = PluginCatalogController(DATA_DESIGNER_HOME)
    controller.run_catalogs_remove(alias=alias)


def _resolve_catalog_alias(ctx: typer.Context, catalog_alias: str | None) -> str | None:
    if catalog_alias is not None:
        return catalog_alias

    return _parent_catalog_alias(ctx)


def _parent_catalog_alias(ctx: typer.Context) -> str | None:
    """Return --catalog from the plugins parent command when present."""

    parent = ctx.parent
    while parent is not None:
        candidate = parent.params.get("catalog") if parent.params else None
        if isinstance(candidate, str) and candidate:
            return candidate
        parent = parent.parent
    return None


def _warn_if_parent_catalog_unused(ctx: typer.Context, reason: str) -> None:
    catalog_alias = _parent_catalog_alias(ctx)
    if catalog_alias is not None:
        print_info(f"Ignoring --catalog {catalog_alias!r}; {reason}.")
