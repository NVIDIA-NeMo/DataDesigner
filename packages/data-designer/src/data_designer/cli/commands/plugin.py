# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import click
import typer

from data_designer.cli.controllers.plugin_catalog_controller import PluginCatalogController
from data_designer.cli.ui import print_warning
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
        help="Show catalog packages that do not satisfy the local Python or Data Designer version.",
    ),
) -> None:
    """List installable Data Designer plugin packages from a catalog."""
    controller = PluginCatalogController(DATA_DESIGNER_HOME)
    controller.run_list(
        catalog_alias=_resolve_catalog_alias(ctx, catalog),
        refresh=refresh,
        include_incompatible=include_incompatible,
    )


def search_command(
    ctx: typer.Context,
    query: str = typer.Argument(
        help="Keyword, package name or alias, description, runtime plugin name, or runtime plugin type to search for."
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
        help="Search catalog packages that do not satisfy the local Python or Data Designer version.",
    ),
) -> None:
    """Search installable Data Designer plugin packages from a catalog."""
    controller = PluginCatalogController(DATA_DESIGNER_HOME)
    controller.run_search(
        query,
        catalog_alias=_resolve_catalog_alias(ctx, catalog),
        refresh=refresh,
        include_incompatible=include_incompatible,
    )


def info_command(
    ctx: typer.Context,
    package: str = typer.Argument(
        help="Plugin package name or package alias from the catalog.",
        metavar="PACKAGE",
    ),
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
    """Show metadata, compatibility, docs, and install strategy for one plugin package."""
    controller = PluginCatalogController(DATA_DESIGNER_HOME)
    controller.run_info(
        package,
        catalog_alias=_resolve_catalog_alias(ctx, catalog),
        refresh=refresh,
    )


def install_command(
    ctx: typer.Context,
    package: str = typer.Argument(
        help="Plugin package name or package alias from the catalog. May include a version specifier.",
        metavar="PACKAGE",
    ),
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
        help=(
            "Package manager to use. auto prefers uv; uv adds to the active project when one is detected; "
            "pip mutates the environment."
        ),
    ),
    version: str | None = typer.Option(
        None,
        "--version",
        help="Exact plugin package version to install.",
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
        help=(
            "Print the install plan without mutating the current environment. Exits 1 if compatibility would block "
            "install."
        ),
    ),
) -> None:
    """Install one Data Designer plugin package, then verify declared runtime entry points."""
    controller = PluginCatalogController(DATA_DESIGNER_HOME)
    controller.run_install(
        package,
        catalog_alias=_resolve_catalog_alias(ctx, catalog),
        refresh=refresh,
        manager=manager,
        version=version,
        yes=yes,
        dry_run=dry_run,
    )


def uninstall_command(
    ctx: typer.Context,
    package: str = typer.Argument(
        help="Plugin package name or package alias from the catalog.",
        metavar="PACKAGE",
    ),
    catalog: str | None = typer.Option(
        None,
        "--catalog",
        help="Plugin catalog alias to uninstall from. Can also be provided before the subcommand.",
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
        help=(
            "Package manager to use. auto prefers uv; uv removes from the active project and environment when a "
            "project is detected; pip mutates the environment."
        ),
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Uninstall without an interactive confirmation prompt.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print the uninstall plan without mutating the current environment.",
    ),
) -> None:
    """Uninstall one Data Designer plugin package, then verify declared runtime entry points are removed."""
    controller = PluginCatalogController(DATA_DESIGNER_HOME)
    controller.run_uninstall(
        package,
        catalog_alias=_resolve_catalog_alias(ctx, catalog),
        refresh=refresh,
        manager=manager,
        yes=yes,
        dry_run=dry_run,
    )


def installed_command(ctx: typer.Context) -> None:
    """List installed Data Designer plugin packages with runtime plugin metadata."""
    _warn_if_parent_catalog_unused(ctx, "installed plugin packages are discovered from the current Python environment")
    controller = PluginCatalogController(DATA_DESIGNER_HOME)
    controller.run_installed()


def catalog_list_command(ctx: typer.Context) -> None:
    """List configured plugin catalogs."""
    _warn_if_parent_catalog_unused(ctx, "catalog management commands operate on aliases directly")
    controller = PluginCatalogController(DATA_DESIGNER_HOME)
    controller.run_catalog_list()


def catalog_add_command(
    ctx: typer.Context,
    alias: str = typer.Argument(help="Local alias for the plugin catalog."),
    url: str = typer.Argument(
        help="Catalog repository URL, catalog URL, local catalog file, or local catalog directory."
    ),
) -> None:
    """Add a plugin catalog alias."""
    _warn_if_parent_catalog_unused(ctx, "catalog management commands operate on aliases directly")
    controller = PluginCatalogController(DATA_DESIGNER_HOME)
    controller.run_catalog_add(
        alias=alias,
        url=url,
    )


def catalog_remove_command(
    ctx: typer.Context,
    alias: str = typer.Argument(help="Plugin catalog alias to remove."),
) -> None:
    """Remove a plugin catalog alias."""
    _warn_if_parent_catalog_unused(ctx, "catalog management commands operate on aliases directly")
    controller = PluginCatalogController(DATA_DESIGNER_HOME)
    controller.run_catalog_remove(alias=alias)


def _resolve_catalog_alias(ctx: typer.Context, catalog_alias: str | None) -> str | None:
    if catalog_alias is not None:
        return catalog_alias

    return _parent_catalog_alias(ctx)


def _parent_catalog_alias(ctx: typer.Context) -> str | None:
    """Return --catalog from the plugin parent command when present."""

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
        print_warning(f"Ignoring --catalog {catalog_alias!r}; {reason}.")
