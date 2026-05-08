# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import shlex
from pathlib import Path

import typer
from pydantic import ValidationError
from rich.table import Table

from data_designer.cli.plugin_catalog import (
    DEFAULT_PLUGIN_CATALOG_ALIAS,
    PLUGIN_CATALOG_ALIAS_PATTERN,
    CompatibilityResult,
    InstalledPluginInfo,
    PluginCatalogConfig,
    PluginCatalogEntry,
    PluginCatalogError,
)
from data_designer.cli.repositories.plugin_catalog_repository import PluginCatalogRepository
from data_designer.cli.services.plugin_catalog_service import PluginCatalogService
from data_designer.cli.services.plugin_install_service import PluginInstallService
from data_designer.cli.ui import (
    confirm_action,
    console,
    display_config_preview,
    print_error,
    print_header,
    print_info,
    print_success,
    print_warning,
)
from data_designer.config.utils.constants import NordColor


class PluginCatalogController:
    """Controller for plugin catalog browsing, alias management, and package workflows.

    Catalog browsing and environment mutation intentionally use separate services so
    read-only catalog operations stay decoupled from package-manager execution.
    """

    def __init__(self, config_dir: Path) -> None:
        self.config_dir = config_dir
        self.catalog_repository = PluginCatalogRepository(config_dir)
        self.catalog_service = PluginCatalogService(self.catalog_repository)
        self.install_service = PluginInstallService()

    def run_list(
        self,
        *,
        catalog_alias: str | None = None,
        refresh: bool = False,
        include_incompatible: bool = False,
    ) -> None:
        """List plugin packages from a catalog."""
        catalog = self._get_catalog_or_exit(catalog_alias)
        entries = self._list_entries_or_exit(catalog.alias, refresh=refresh, include_incompatible=include_incompatible)

        print_header("Data Designer Plugin Packages")
        print_info(f"Catalog: {catalog.alias} ({catalog.url})")
        console.print()

        if not entries:
            self._display_empty_list_state(catalog.alias, include_incompatible=include_incompatible)
            return

        self._display_catalog_entries(entries)

    def run_search(
        self,
        query: str,
        *,
        catalog_alias: str | None = None,
        refresh: bool = False,
        include_incompatible: bool = False,
    ) -> None:
        """Search plugin packages from a catalog."""
        catalog = self._get_catalog_or_exit(catalog_alias)
        entries = self._search_entries_or_exit(
            query,
            catalog.alias,
            refresh=refresh,
            include_incompatible=include_incompatible,
        )

        print_header("Data Designer Plugin Package Search")
        print_info(f"Catalog: {catalog.alias} ({catalog.url})")
        print_info(f"Query: {query}")
        console.print()

        if not entries:
            self._display_empty_search_state(
                query,
                catalog.alias,
                include_incompatible=include_incompatible,
            )
            return

        self._display_catalog_entries(entries)

    def run_info(
        self,
        package_name: str,
        *,
        catalog_alias: str | None = None,
        refresh: bool = False,
    ) -> None:
        """Show full metadata for one plugin package."""
        catalog = self._get_catalog_or_exit(catalog_alias)
        package_entries = self._get_package_entries_or_exit(
            package_name,
            catalog.alias,
            refresh=refresh,
            include_incompatible=True,
        )
        entry = package_entries[0]
        compatibility = self.catalog_service.evaluate_compatibility(entry)

        print_header(f"Plugin Package: {entry.package.name}")
        print_info(f"Catalog: {catalog.alias} ({catalog.url})")
        console.print(f"  Runtime plugins: [bold]{_format_runtime_plugins(package_entries)}[/bold]")
        self._display_compatibility(compatibility)

        try:
            plan = self.install_service.build_install_plan(entry, catalog)
            console.print(f"  Requirement: [bold]{entry.install.requirement}[/bold]")
            if entry.install.index_url is not None:
                console.print(f"  Index URL: [bold]{entry.install.index_url}[/bold]")
            console.print(f"  Install command: [bold]{shlex.join(plan.command)}[/bold]")
        except ValueError as e:
            print_warning(str(e))

        console.print()
        display_config_preview(
            {
                "package": {
                    "name": entry.package.name,
                    "description": entry.description,
                },
                "install": entry.install.model_dump(mode="json", exclude_none=True),
                "compatibility": (
                    entry.compatibility.model_dump(mode="json", exclude_none=True)
                    if entry.compatibility is not None
                    else None
                ),
                "docs": entry.docs.model_dump(mode="json", exclude_none=True) if entry.docs is not None else None,
                "plugins": [_runtime_plugin_metadata(plugin) for plugin in package_entries],
            },
            "Plugin Metadata",
        )

    def run_install(
        self,
        package_name: str,
        *,
        catalog_alias: str | None = None,
        refresh: bool = False,
        manager: str = "auto",
        yes: bool = False,
        dry_run: bool = False,
        force: bool = False,
    ) -> None:
        """Install one plugin package from the catalog."""
        catalog = self._get_catalog_or_exit(catalog_alias)
        package_entries = self._get_package_entries_or_exit(
            package_name,
            catalog.alias,
            refresh=refresh,
            include_incompatible=True,
        )
        entry = package_entries[0]
        compatibility = self.catalog_service.evaluate_compatibility(entry)

        if not compatibility.is_compatible and not force and not dry_run:
            print_error(f"Plugin package {entry.package.name!r} is not compatible with this environment")
            for reason in compatibility.reasons:
                console.print(f"  - {reason}")
            raise typer.Exit(code=1)

        try:
            plan = self.install_service.build_install_plan(entry, catalog, manager=manager)
        except ValueError as e:
            print_error(f"Failed to build plugin install plan: {e}")
            raise typer.Exit(code=1)

        print_header("Install Data Designer Plugin Package")
        console.print(f"  Package: [bold]{entry.package.name}[/bold]")
        console.print(f"  Catalog: [bold]{catalog.alias}[/bold] ({catalog.url})")
        console.print(f"  Requirement: [bold]{entry.install.requirement}[/bold]")
        if entry.install.index_url is not None:
            console.print(f"  Index URL: [bold]{entry.install.index_url}[/bold]")
        console.print(f"  Command: [bold]{shlex.join(plan.command)}[/bold]")
        self._display_compatibility(compatibility)

        if not catalog.trusted:
            print_warning(
                "This catalog is not marked trusted. Plugin package installation executes Python package code from "
                "the requirement above."
            )

        if dry_run:
            if not compatibility.is_compatible and not force:
                print_warning(
                    "Dry run complete; no changes made. A real install would be blocked unless you pass --force."
                )
            else:
                print_info("Dry run complete; no changes made")
            return

        if not yes and not confirm_action("Install this package into the current Python environment?", default=False):
            print_info("No changes made")
            return

        try:
            self.install_service.install(plan)
        except RuntimeError as e:
            print_error(str(e))
            raise typer.Exit(code=1)

        if self.install_service.verify_entry_points(package_entries):
            print_success(f"Plugin package {entry.package.name!r} installed and registered")
        else:
            print_warning(
                f"Plugin package {entry.package.name!r} was installed, but Data Designer did not discover every "
                "declared package entry point. Restart the shell or check the package entry point metadata."
            )

    def run_uninstall(
        self,
        package_name: str,
        *,
        catalog_alias: str | None = None,
        refresh: bool = False,
        manager: str = "auto",
        yes: bool = False,
        dry_run: bool = False,
    ) -> None:
        """Uninstall one plugin package resolved from the catalog."""
        catalog = self._get_catalog_or_exit(catalog_alias)
        package_entries = self._get_package_entries_or_exit(
            package_name,
            catalog.alias,
            refresh=refresh,
            include_incompatible=True,
        )
        entry = package_entries[0]

        try:
            plan = self.install_service.build_uninstall_plan(entry, catalog, manager=manager)
        except ValueError as e:
            print_error(f"Failed to build plugin uninstall plan: {e}")
            raise typer.Exit(code=1)

        print_header("Uninstall Data Designer Plugin Package")
        console.print(f"  Package: [bold]{entry.package.name}[/bold]")
        console.print(f"  Catalog: [bold]{catalog.alias}[/bold] ({catalog.url})")
        console.print(f"  Command: [bold]{shlex.join(plan.command)}[/bold]")

        if dry_run:
            print_info("Dry run complete; no changes made")
            return

        if not yes and not confirm_action("Uninstall this package from the current Python environment?", default=False):
            print_info("No changes made")
            return

        try:
            self.install_service.uninstall(plan)
        except RuntimeError as e:
            print_error(str(e))
            raise typer.Exit(code=1)

        if self.install_service.verify_entry_points_removed(package_entries):
            print_success(f"Plugin package {entry.package.name!r} uninstalled and no longer registered")
        else:
            print_warning(
                f"Plugin package {entry.package.name!r} was uninstalled, but Data Designer still discovers one or "
                "more declared package entry points. Restart the shell or check the package environment."
            )

    def run_installed(self) -> None:
        """List installed runtime plugin entry points without importing plugin modules."""
        print_header("Installed Data Designer Runtime Plugins")
        installed_plugins = self.catalog_service.list_installed_plugins()
        if not installed_plugins:
            print_warning("No installed Data Designer runtime plugins were discovered")
            return
        self._display_installed_plugins(installed_plugins)

    def run_catalogs_list(self) -> None:
        """List configured plugin catalogs."""
        print_header("Data Designer Plugin Catalogs")
        try:
            catalogs = self.catalog_service.list_catalogs()
        except (PluginCatalogError, OSError) as e:
            print_error(f"Failed to list plugin catalogs: {e}")
            raise typer.Exit(code=1)

        table = Table(title="Plugin Catalogs", border_style=NordColor.NORD8.value)
        table.add_column("Alias", style=NordColor.NORD14.value, no_wrap=True)
        table.add_column("URL", style=NordColor.NORD4.value)
        table.add_column("Trusted", style=NordColor.NORD13.value, justify="center")
        table.add_column("Cache TTL", style=NordColor.NORD9.value, justify="right")

        for catalog in catalogs:
            table.add_row(
                catalog.alias,
                catalog.url,
                "yes" if catalog.trusted else "no",
                f"{catalog.cache_ttl_seconds}s",
            )
        console.print(table)

    def run_catalogs_add(
        self,
        *,
        alias: str,
        url: str,
        trusted: bool,
        cache_ttl_seconds: int,
    ) -> None:
        """Add a plugin catalog alias."""
        try:
            catalog = self.catalog_service.add_catalog(
                alias,
                url,
                trusted=trusted,
                cache_ttl_seconds=cache_ttl_seconds,
            )
        except ValidationError as e:
            if any(tuple(error["loc"]) == ("alias",) for error in e.errors()):
                print_error(f"Invalid catalog alias {alias!r}: must match `{PLUGIN_CATALOG_ALIAS_PATTERN}`")
            else:
                print_error(f"Invalid plugin catalog configuration: {e}")
            raise typer.Exit(code=1)
        except (PluginCatalogError, OSError, ValueError) as e:
            print_error(f"Failed to add plugin catalog: {e}")
            raise typer.Exit(code=1)

        print_success(f"Plugin catalog {catalog.alias!r} added")
        print_info(f"Catalog: {catalog.url}")

    def run_catalogs_remove(self, *, alias: str) -> None:
        """Remove a plugin catalog alias."""
        try:
            self.catalog_service.remove_catalog(alias)
        except (PluginCatalogError, OSError, ValueError) as e:
            print_error(f"Failed to remove plugin catalog: {e}")
            raise typer.Exit(code=1)
        print_success(f"Plugin catalog {alias!r} removed")

    def _get_catalog_or_exit(self, catalog_alias: str | None) -> PluginCatalogConfig:
        try:
            return self.catalog_service.get_catalog(catalog_alias or DEFAULT_PLUGIN_CATALOG_ALIAS)
        except (PluginCatalogError, OSError, ValueError) as e:
            print_error(str(e))
            raise typer.Exit(code=1)

    def _list_entries_or_exit(
        self,
        catalog_alias: str,
        *,
        refresh: bool,
        include_incompatible: bool,
    ) -> list[PluginCatalogEntry]:
        try:
            return self.catalog_service.list_entries(
                catalog_alias,
                refresh=refresh,
                include_incompatible=include_incompatible,
            )
        except (PluginCatalogError, OSError, ValueError) as e:
            print_error(f"Failed to load plugin catalog: {e}")
            raise typer.Exit(code=1)

    def _search_entries_or_exit(
        self,
        query: str,
        catalog_alias: str,
        *,
        refresh: bool,
        include_incompatible: bool,
    ) -> list[PluginCatalogEntry]:
        try:
            return self.catalog_service.search_entries(
                query,
                catalog_alias,
                refresh=refresh,
                include_incompatible=include_incompatible,
            )
        except (PluginCatalogError, OSError, ValueError) as e:
            print_error(f"Failed to search plugin catalog: {e}")
            raise typer.Exit(code=1)

    def _get_package_entries_or_exit(
        self,
        package_name: str,
        catalog_alias: str,
        *,
        refresh: bool,
        include_incompatible: bool,
    ) -> list[PluginCatalogEntry]:
        try:
            package_entries = self.catalog_service.get_package_entries(
                package_name,
                catalog_alias,
                refresh=refresh,
                include_incompatible=include_incompatible,
            )
        except (PluginCatalogError, OSError, ValueError) as e:
            print_error(f"Failed to load plugin package metadata: {e}")
            raise typer.Exit(code=1)
        if not package_entries:
            print_error(f"Plugin package or alias {package_name!r} was not found in catalog {catalog_alias!r}")
            raise typer.Exit(code=1)
        return package_entries

    def _display_empty_list_state(self, catalog_alias: str, *, include_incompatible: bool) -> None:
        if include_incompatible:
            print_warning("No plugin packages found")
            return

        all_entries = self._list_entries_or_exit(catalog_alias, refresh=False, include_incompatible=True)
        if all_entries:
            print_warning("No compatible plugin packages found")
            print_info("Incompatible catalog packages are hidden. Use --include-incompatible to show them.")
            return

        print_warning("No plugin packages found")

    def _display_empty_search_state(
        self,
        query: str,
        catalog_alias: str,
        *,
        include_incompatible: bool,
    ) -> None:
        if include_incompatible:
            print_warning("No matching plugin packages found")
            return

        all_matches = self._search_entries_or_exit(
            query,
            catalog_alias,
            refresh=False,
            include_incompatible=True,
        )
        if all_matches:
            print_warning("No compatible plugin packages matched")
            print_info("Matching incompatible catalog packages are hidden. Use --include-incompatible to show them.")
            return

        print_warning("No matching plugin packages found")

    def _display_catalog_entries(self, entries: list[PluginCatalogEntry]) -> None:
        table = Table(title="Catalog Plugin Packages", border_style=NordColor.NORD8.value)
        table.add_column("Package", style=NordColor.NORD14.value, no_wrap=True)
        table.add_column("Runtime Plugins", style=NordColor.NORD9.value)
        table.add_column("Compatible", style=NordColor.NORD13.value, no_wrap=True)
        table.add_column("Docs", style=NordColor.NORD7.value)

        for package_entries in self.catalog_service.group_entries_by_package(entries).values():
            entry = package_entries[0]
            compatibility = self.catalog_service.evaluate_compatibility(entry)
            docs_url = entry.docs.url if entry.docs is not None and entry.docs.url is not None else ""
            table.add_row(
                entry.package.name,
                _format_runtime_plugins(package_entries),
                "yes" if compatibility.is_compatible else "no",
                docs_url,
            )
        console.print(table)

    @staticmethod
    def _display_installed_plugins(installed_plugins: list[InstalledPluginInfo]) -> None:
        table = Table(title="Installed Runtime Plugins", border_style=NordColor.NORD8.value)
        table.add_column("Runtime Plugin", style=NordColor.NORD14.value, no_wrap=True)
        table.add_column("Entry Point", style=NordColor.NORD4.value)

        for plugin in installed_plugins:
            table.add_row(
                plugin.name,
                plugin.entry_point_value,
            )
        console.print(table)

    @staticmethod
    def _display_compatibility(compatibility: CompatibilityResult) -> None:
        if compatibility.is_compatible:
            console.print("  Compatibility: [bold green]compatible[/bold green]")
            return

        console.print("  Compatibility: [bold yellow]not compatible[/bold yellow]")
        for reason in compatibility.reasons:
            console.print(f"    - {reason}")


def _format_runtime_plugins(entries: list[PluginCatalogEntry]) -> str:
    return ", ".join(f"{entry.name} ({entry.plugin_type.value})" for entry in entries)


def _runtime_plugin_metadata(entry: PluginCatalogEntry) -> dict[str, object]:
    return {
        "name": entry.name,
        "plugin_type": entry.plugin_type.value,
        "entry_point": entry.entry_point.model_dump(mode="json", exclude_none=True),
    }
