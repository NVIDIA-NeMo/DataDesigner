# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import shlex
from pathlib import Path

import typer
from rich.table import Table

from data_designer.cli.plugin_catalog import (
    DEFAULT_PLUGIN_TAP_ALIAS,
    CompatibilityResult,
    InstalledPluginInfo,
    PluginCatalogEntry,
    PluginTapConfig,
)
from data_designer.cli.repositories.plugin_tap_repository import PluginTapRepository
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
    """Controller for plugin catalog, tap, and install workflows."""

    def __init__(self, config_dir: Path) -> None:
        self.config_dir = config_dir
        self.tap_repository = PluginTapRepository(config_dir)
        self.catalog_service = PluginCatalogService(self.tap_repository)
        self.install_service = PluginInstallService()

    def run_list(
        self,
        *,
        tap_alias: str | None = None,
        refresh: bool = False,
        include_incompatible: bool = False,
    ) -> None:
        """List plugins from a tap catalog."""
        tap = self._get_tap_or_exit(tap_alias)
        entries = self._list_entries_or_exit(tap.alias, refresh=refresh, include_incompatible=include_incompatible)

        print_header("Data Designer Plugins")
        print_info(f"Tap: {tap.alias} ({tap.url})")
        console.print()

        if not entries:
            print_warning("No plugins found")
            return

        self._display_catalog_entries(entries)

    def run_search(
        self,
        query: str,
        *,
        tap_alias: str | None = None,
        refresh: bool = False,
        include_incompatible: bool = False,
    ) -> None:
        """Search plugins from a tap catalog."""
        tap = self._get_tap_or_exit(tap_alias)
        try:
            entries = self.catalog_service.search_entries(
                query,
                tap.alias,
                refresh=refresh,
                include_incompatible=include_incompatible,
            )
        except Exception as e:
            print_error(f"Failed to search plugin catalog: {e}")
            raise typer.Exit(code=1)

        print_header("Data Designer Plugin Search")
        print_info(f"Tap: {tap.alias} ({tap.url})")
        print_info(f"Query: {query}")
        console.print()

        if not entries:
            print_warning("No matching plugins found")
            return

        self._display_catalog_entries(entries)

    def run_info(
        self,
        plugin_name: str,
        *,
        tap_alias: str | None = None,
        refresh: bool = False,
    ) -> None:
        """Show full metadata for one plugin."""
        tap = self._get_tap_or_exit(tap_alias)
        entry = self._get_entry_or_exit(plugin_name, tap.alias, refresh=refresh)
        compatibility = self.catalog_service.evaluate_compatibility(entry)

        print_header(f"Plugin: {entry.name}")
        print_info(f"Tap: {tap.alias} ({tap.url})")
        self._display_compatibility(compatibility)

        try:
            plan = self.install_service.build_install_plan(entry, tap)
            console.print(f"  Install command: [bold]{shlex.join(plan.command)}[/bold]")
        except ValueError as e:
            print_warning(str(e))

        console.print()
        display_config_preview(entry.model_dump(mode="json", exclude_none=True), "Plugin Metadata")

    def run_install(
        self,
        plugin_name: str,
        *,
        tap_alias: str | None = None,
        refresh: bool = False,
        manager: str = "auto",
        yes: bool = False,
        dry_run: bool = False,
        force: bool = False,
    ) -> None:
        """Install one plugin from a catalog entry."""
        tap = self._get_tap_or_exit(tap_alias)
        entry = self._get_entry_or_exit(plugin_name, tap.alias, refresh=refresh, include_incompatible=force)
        compatibility = self.catalog_service.evaluate_compatibility(entry)

        if not compatibility.is_compatible and not force:
            print_error(f"Plugin {entry.name!r} is not compatible with this environment")
            for reason in compatibility.reasons:
                console.print(f"  - {reason}")
            raise typer.Exit(code=1)

        try:
            plan = self.install_service.build_install_plan(entry, tap, manager=manager)
        except ValueError as e:
            print_error(f"Failed to build plugin install plan: {e}")
            raise typer.Exit(code=1)

        print_header("Install Data Designer Plugin")
        console.print(f"  Plugin: [bold]{entry.name}[/bold]")
        console.print(f"  Tap: [bold]{tap.alias}[/bold] ({tap.url})")
        console.print(f"  Source: [bold]{plan.source_description}[/bold]")
        console.print(f"  Command: [bold]{shlex.join(plan.command)}[/bold]")
        self._display_compatibility(compatibility)

        if not tap.trusted:
            print_warning(
                "This tap is not marked trusted. Plugin installation executes Python package code from the source above."
            )

        if dry_run:
            print_info("Dry run complete; no changes made")
            return

        if not yes and not confirm_action("Install this plugin into the current Python environment?", default=False):
            print_info("No changes made")
            return

        try:
            self.install_service.install(plan)
        except RuntimeError as e:
            print_error(str(e))
            raise typer.Exit(code=1)

        if self.install_service.verify_entry_point(entry):
            print_success(f"Plugin {entry.name!r} installed and discovered")
        else:
            print_warning(
                f"Plugin {entry.name!r} was installed, but Data Designer did not discover its entry point. "
                "Restart the shell or check the package entry point metadata."
            )

    def run_installed(self) -> None:
        """List plugins currently discoverable through runtime entry points."""
        print_header("Installed Data Designer Plugins")
        installed_plugins = self.catalog_service.list_installed_plugins()
        if not installed_plugins:
            print_warning("No installed Data Designer plugins were discovered")
            return
        self._display_installed_plugins(installed_plugins)

    def run_taps_list(self) -> None:
        """List configured plugin taps."""
        print_header("Data Designer Plugin Taps")
        taps = self.catalog_service.list_taps()
        table = Table(title="Plugin Taps", border_style=NordColor.NORD8.value)
        table.add_column("Alias", style=NordColor.NORD14.value, no_wrap=True)
        table.add_column("URL", style=NordColor.NORD4.value)
        table.add_column("Trusted", style=NordColor.NORD13.value, justify="center")
        table.add_column("Cache TTL", style=NordColor.NORD9.value, justify="right")

        for tap in taps:
            table.add_row(
                tap.alias,
                tap.url,
                "yes" if tap.trusted else "no",
                f"{tap.cache_ttl_seconds}s",
            )
        console.print(table)

    def run_taps_add(
        self,
        *,
        alias: str,
        url: str,
        trusted: bool,
        cache_ttl_seconds: int,
    ) -> None:
        """Add a plugin tap alias."""
        try:
            tap = self.catalog_service.add_tap(
                alias,
                url,
                trusted=trusted,
                cache_ttl_seconds=cache_ttl_seconds,
            )
        except Exception as e:
            print_error(f"Failed to add plugin tap: {e}")
            raise typer.Exit(code=1)

        print_success(f"Plugin tap {tap.alias!r} added")
        print_info(f"Catalog: {tap.url}")

    def run_taps_remove(self, *, alias: str) -> None:
        """Remove a plugin tap alias."""
        try:
            self.catalog_service.remove_tap(alias)
        except Exception as e:
            print_error(f"Failed to remove plugin tap: {e}")
            raise typer.Exit(code=1)
        print_success(f"Plugin tap {alias!r} removed")

    def _get_tap_or_exit(self, tap_alias: str | None) -> PluginTapConfig:
        try:
            return self.catalog_service.get_tap(tap_alias or DEFAULT_PLUGIN_TAP_ALIAS)
        except ValueError as e:
            print_error(str(e))
            raise typer.Exit(code=1)

    def _list_entries_or_exit(
        self,
        tap_alias: str,
        *,
        refresh: bool,
        include_incompatible: bool,
    ) -> list[PluginCatalogEntry]:
        try:
            return self.catalog_service.list_entries(
                tap_alias,
                refresh=refresh,
                include_incompatible=include_incompatible,
            )
        except Exception as e:
            print_error(f"Failed to load plugin catalog: {e}")
            raise typer.Exit(code=1)

    def _get_entry_or_exit(
        self,
        plugin_name: str,
        tap_alias: str,
        *,
        refresh: bool,
        include_incompatible: bool = True,
    ) -> PluginCatalogEntry:
        try:
            return self.catalog_service.get_entry(
                plugin_name,
                tap_alias,
                refresh=refresh,
                include_incompatible=include_incompatible,
            )
        except Exception as e:
            print_error(str(e))
            raise typer.Exit(code=1)

    def _display_catalog_entries(self, entries: list[PluginCatalogEntry]) -> None:
        table = Table(title="Catalog Plugins", border_style=NordColor.NORD8.value)
        table.add_column("Name", style=NordColor.NORD14.value, no_wrap=True)
        table.add_column("Type", style=NordColor.NORD9.value, no_wrap=True)
        table.add_column("Package", style=NordColor.NORD4.value)
        table.add_column("Version", style=NordColor.NORD15.value, no_wrap=True)
        table.add_column("Compatible", style=NordColor.NORD13.value, no_wrap=True)
        table.add_column("Docs", style=NordColor.NORD7.value)

        for entry in entries:
            compatibility = self.catalog_service.evaluate_compatibility(entry)
            docs_url = entry.docs.url if entry.docs is not None and entry.docs.url is not None else ""
            table.add_row(
                entry.name,
                entry.plugin_type.value,
                entry.package.name,
                entry.package.version or "",
                "yes" if compatibility.is_compatible else "no",
                docs_url,
            )
        console.print(table)

    @staticmethod
    def _display_installed_plugins(installed_plugins: list[InstalledPluginInfo]) -> None:
        table = Table(title="Installed Plugins", border_style=NordColor.NORD8.value)
        table.add_column("Name", style=NordColor.NORD14.value, no_wrap=True)
        table.add_column("Type", style=NordColor.NORD9.value, no_wrap=True)
        table.add_column("Config", style=NordColor.NORD4.value)
        table.add_column("Implementation", style=NordColor.NORD7.value)

        for plugin in installed_plugins:
            table.add_row(
                plugin.name,
                plugin.plugin_type.value,
                plugin.config_qualified_name,
                plugin.impl_qualified_name,
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
