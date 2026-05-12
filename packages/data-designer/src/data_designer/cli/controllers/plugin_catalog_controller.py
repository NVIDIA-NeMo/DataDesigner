# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import shlex
from pathlib import Path

import typer
from packaging.utils import canonicalize_name
from pydantic import ValidationError
from rich.markup import escape
from rich.style import Style
from rich.table import Table
from rich.text import Text

from data_designer.cli.plugin_catalog import (
    DATA_DESIGNER_PLUGIN_PACKAGE_PREFIX,
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

NARROW_CATALOG_LAYOUT_WIDTH = 100
CATALOG_TABLE_ROW_LEADING = 1
CHECKMARK = "✓"
X_MARK = "x"


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
        console.print()

        if not entries:
            self._display_empty_list_state(catalog.alias, include_incompatible=include_incompatible)
            _print_catalog_reference(catalog)
            return

        self._display_catalog_entries(entries)
        _print_catalog_reference(catalog)

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
        print_info(f"Query: {_escape_markup(query)}")
        console.print()

        if not entries:
            self._display_empty_search_state(
                query,
                catalog.alias,
                include_incompatible=include_incompatible,
            )
            _print_catalog_reference(catalog)
            return

        self._display_catalog_entries(entries)
        _print_catalog_reference(catalog)

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
            command_name="info",
        )
        entry = package_entries[0]
        compatibility = self.catalog_service.evaluate_compatibility(entry)

        print_header(f"Plugin Package: {_escape_markup(entry.package.name)}")
        print_info(f"Catalog: {_escape_markup(catalog.alias)} ({_escape_markup(catalog.url)})")
        console.print(f"  Runtime plugins: [bold]{_escape_markup(_format_runtime_plugins(package_entries))}[/bold]")
        self._display_compatibility(compatibility)

        try:
            plan = self.install_service.build_install_plan(entry, catalog)
            console.print(f"  Requirement: [bold]{_escape_markup(entry.install.requirement)}[/bold]")
            if entry.install.index_url is not None:
                console.print(f"  Index URL: [bold]{_escape_markup(entry.install.index_url)}[/bold]")
            console.print(
                f"  Install target: [bold]{_escape_markup(_target_description(plan.install_mode, plan.project_root))}[/bold]"
            )
            if plan.data_designer_protection is not None:
                console.print(f"  Data Designer: [bold]{_escape_markup(plan.data_designer_protection)}[/bold]")
            console.print(f"  Install command: [bold]{_escape_markup(shlex.join(plan.command))}[/bold]")
            if plan.source_warning is not None:
                print_warning(plan.source_warning)
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
    ) -> None:
        """Install one plugin package from the catalog."""
        catalog = self._get_catalog_or_exit(catalog_alias)
        package_entries = self._get_package_entries_or_exit(
            package_name,
            catalog.alias,
            refresh=refresh,
            include_incompatible=True,
            command_name="install",
        )
        entry = package_entries[0]
        compatibility = self.catalog_service.evaluate_compatibility(entry)

        if not compatibility.is_compatible and not dry_run:
            print_error(f"Plugin package {entry.package.name!r} is not compatible with this environment")
            for reason in compatibility.reasons:
                console.print(Text.assemble("  - ", reason))
            raise typer.Exit(code=1)

        try:
            plan = self.install_service.build_install_plan(entry, catalog, manager=manager)
        except ValueError as e:
            print_error(f"Failed to build plugin install plan: {e}")
            raise typer.Exit(code=1)

        print_header("Install Data Designer Plugin Package")
        console.print(f"  Package: [bold]{_escape_markup(entry.package.name)}[/bold]")
        console.print(f"  Catalog: [bold]{_escape_markup(catalog.alias)}[/bold] ({_escape_markup(catalog.url)})")
        console.print(f"  Requirement: [bold]{_escape_markup(entry.install.requirement)}[/bold]")
        if entry.install.index_url is not None:
            console.print(f"  Index URL: [bold]{_escape_markup(entry.install.index_url)}[/bold]")
        console.print(
            f"  Install target: [bold]{_escape_markup(_target_description(plan.install_mode, plan.project_root))}[/bold]"
        )
        if plan.data_designer_protection is not None:
            console.print(f"  Data Designer: [bold]{_escape_markup(plan.data_designer_protection)}[/bold]")
        console.print(f"  Command: [bold]{_escape_markup(shlex.join(plan.command))}[/bold]")
        self._display_compatibility(compatibility)

        if plan.source_warning is not None:
            print_warning(plan.source_warning)

        if dry_run:
            if not compatibility.is_compatible:
                print_warning(
                    "Dry run complete; no changes made. Install would be blocked because compatibility checks failed."
                )
                raise typer.Exit(code=1)
            else:
                print_info("Dry run complete; no changes made")
            return

        if not yes and not confirm_action(
            f"Install this package into the {_target_description(plan.install_mode, plan.project_root)}?",
            default=False,
        ):
            print_info("No changes made")
            return

        try:
            self.install_service.install(plan)
        except RuntimeError as e:
            print_error(str(e))
            raise typer.Exit(code=1)

        if self.install_service.verify_entry_points(package_entries):
            print_success(f"Plugin package {entry.package.name!r} installed and runtime entry points loaded")
        else:
            print_warning(
                f"Plugin package {entry.package.name!r} was installed, but Data Designer could not load every "
                "declared runtime entry point. Restart the shell or check the package code and entry point metadata."
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
            command_name="uninstall",
        )
        entry = package_entries[0]

        try:
            plan = self.install_service.build_uninstall_plan(entry, catalog, manager=manager)
        except ValueError as e:
            print_error(f"Failed to build plugin uninstall plan: {e}")
            raise typer.Exit(code=1)

        print_header("Uninstall Data Designer Plugin Package")
        console.print(f"  Package: [bold]{_escape_markup(entry.package.name)}[/bold]")
        console.print(f"  Catalog: [bold]{_escape_markup(catalog.alias)}[/bold] ({_escape_markup(catalog.url)})")
        console.print(
            f"  Uninstall target: [bold]{_escape_markup(_target_description(plan.uninstall_mode, plan.project_root))}[/bold]"
        )
        _display_commands(plan.commands or [plan.command])

        if dry_run:
            print_info("Dry run complete; no changes made")
            return

        if not yes and not confirm_action(
            f"Uninstall this package from the {_target_description(plan.uninstall_mode, plan.project_root)}?",
            default=False,
        ):
            print_info("No changes made")
            return

        try:
            self.install_service.uninstall(plan)
        except RuntimeError as e:
            print_error(str(e))
            raise typer.Exit(code=1)

        if self.install_service.verify_entry_points_removed(package_entries):
            print_success(f"Plugin package {entry.package.name!r} uninstalled and runtime entry points removed")
        else:
            print_warning(
                f"Plugin package {entry.package.name!r} was uninstalled, but Data Designer still discovers one or "
                "more declared runtime entry points. Restart the shell or check the package environment."
            )

    def run_installed(self) -> None:
        """List installed runtime plugins without importing plugin modules."""
        print_header("Installed Data Designer Runtime Plugins")
        installed_plugins = self.catalog_service.list_installed_plugins()
        if not installed_plugins:
            print_warning("No installed Data Designer runtime plugins were discovered")
            return
        self._display_installed_plugins(installed_plugins)

    def run_catalog_list(self) -> None:
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

        for catalog in catalogs:
            table.add_row(
                _escape_markup(catalog.alias),
                _escape_markup(catalog.url),
            )
        console.print(table)

    def run_catalog_add(
        self,
        *,
        alias: str,
        url: str,
    ) -> None:
        """Add a plugin catalog alias."""
        try:
            catalog = self.catalog_service.add_catalog(
                alias,
                url,
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

    def run_catalog_remove(self, *, alias: str) -> None:
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
        command_name: str,
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
            self._display_runtime_plugin_recovery_hint(
                package_name,
                catalog_alias,
                refresh=refresh,
                include_incompatible=include_incompatible,
                command_name=command_name,
            )
            raise typer.Exit(code=1)
        return package_entries

    def _display_runtime_plugin_recovery_hint(
        self,
        package_name: str,
        catalog_alias: str,
        *,
        refresh: bool,
        include_incompatible: bool,
        command_name: str,
    ) -> None:
        try:
            runtime_entries = self.catalog_service.get_runtime_plugin_entries(
                package_name,
                catalog_alias,
                refresh=refresh,
                include_incompatible=include_incompatible,
            )
        except (PluginCatalogError, OSError, ValueError):
            return

        if not runtime_entries:
            return

        entry = runtime_entries[0]
        package_alias = _package_alias(entry.package.name) or entry.package.name
        print_info(f"{package_name!r} is a runtime plugin exposed by plugin package {entry.package.name!r}.")
        command = _plugin_package_command(command_name, package_alias, catalog_alias)
        print_info(f"Use the package instead: {shlex.join(command)}")

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
        suggestions = self._suggest_entries(query, catalog_alias, include_incompatible=include_incompatible)
        if suggestions:
            package_names = [
                package_entries[0].package.name
                for package_entries in self.catalog_service.group_entries_by_package(suggestions).values()
            ]
            print_info(f"Closest package matches: {', '.join(package_names)}")
        print_info("Try fewer terms, a package alias, or a runtime plugin name.")

    def _suggest_entries(
        self,
        query: str,
        catalog_alias: str,
        *,
        include_incompatible: bool,
    ) -> list[PluginCatalogEntry]:
        try:
            return self.catalog_service.suggest_entries(
                query,
                catalog_alias,
                refresh=False,
                include_incompatible=include_incompatible,
            )
        except (PluginCatalogError, OSError, ValueError):
            return []

    def _display_catalog_entries(self, entries: list[PluginCatalogEntry]) -> None:
        installed_plugins = self.catalog_service.list_installed_plugins()
        if _console_width() < NARROW_CATALOG_LAYOUT_WIDTH:
            self._display_catalog_entries_vertical(entries, installed_plugins)
            return

        table = Table(
            title="Catalog Plugin Packages",
            border_style=NordColor.NORD8.value,
            leading=CATALOG_TABLE_ROW_LEADING,
        )
        table.add_column("Package", style=NordColor.NORD14.value, no_wrap=True)
        table.add_column("Description", style=NordColor.NORD4.value)
        table.add_column("Runtime Plugins", style=NordColor.NORD9.value)
        table.add_column("Compatible", style=NordColor.NORD13.value, justify="center", no_wrap=True)
        table.add_column("Installed", style=NordColor.NORD14.value, justify="center", no_wrap=True)
        table.add_column("Docs", style=NordColor.NORD7.value)

        for package_entries in self.catalog_service.group_entries_by_package(entries).values():
            entry = package_entries[0]
            compatibility = self.catalog_service.evaluate_compatibility(entry)
            docs_url = entry.docs.url if entry.docs is not None and entry.docs.url is not None else ""
            table.add_row(
                _escape_markup(entry.package.name),
                _escape_markup(entry.description),
                _escape_markup(_format_runtime_plugins(package_entries)),
                _format_compatibility_marker(compatibility),
                _format_installed_marker(package_entries, installed_plugins),
                _format_docs_link(docs_url),
            )
        console.print(table)

    def _display_catalog_entries_vertical(
        self,
        entries: list[PluginCatalogEntry],
        installed_plugins: list[InstalledPluginInfo],
    ) -> None:
        for index, package_entries in enumerate(self.catalog_service.group_entries_by_package(entries).values()):
            entry = package_entries[0]
            compatibility = self.catalog_service.evaluate_compatibility(entry)
            docs_url = entry.docs.url if entry.docs is not None and entry.docs.url is not None else ""
            if index:
                console.print()
            console.print(Text(entry.package.name, style=f"bold {NordColor.NORD14.value}"))
            console.print(f"  Description: {_escape_markup(entry.description)}")
            console.print(f"  Runtime plugins: {_escape_markup(_format_runtime_plugins(package_entries))}")
            console.print(f"  Compatible: {_format_compatibility_marker(compatibility)}")
            console.print(f"  Installed: {_format_installed_marker(package_entries, installed_plugins)}")
            if docs_url:
                console.print(f"  Docs: {_escape_markup(docs_url)}")

    @staticmethod
    def _display_installed_plugins(installed_plugins: list[InstalledPluginInfo]) -> None:
        table = Table(title="Installed Runtime Plugins", border_style=NordColor.NORD8.value)
        table.add_column("Runtime Plugin", style=NordColor.NORD14.value, no_wrap=True)
        table.add_column("Package", style=NordColor.NORD9.value, no_wrap=True)
        table.add_column("Version", style=NordColor.NORD13.value, no_wrap=True)
        table.add_column("Entry Point", style=NordColor.NORD4.value)

        for plugin in installed_plugins:
            table.add_row(
                _escape_markup(plugin.name),
                _escape_markup(plugin.package_name or ""),
                _escape_markup(plugin.package_version or ""),
                _escape_markup(plugin.entry_point_value),
            )
        console.print(table)

    @staticmethod
    def _display_compatibility(compatibility: CompatibilityResult) -> None:
        if compatibility.is_compatible:
            console.print("  Compatibility: [bold green]compatible[/bold green]")
            return

        console.print("  Compatibility: [bold yellow]not compatible[/bold yellow]")
        for reason in compatibility.reasons:
            console.print(Text.assemble("    - ", reason))


def _display_commands(commands: list[list[str]]) -> None:
    if len(commands) == 1:
        console.print(f"  Command: [bold]{_escape_markup(shlex.join(commands[0]))}[/bold]")
        return

    console.print("  Commands:")
    for command in commands:
        console.print(f"    [bold]{_escape_markup(shlex.join(command))}[/bold]")


def _print_catalog_reference(catalog: PluginCatalogConfig) -> None:
    console.print()
    catalog_link = Text.assemble(
        "  🗂️  Catalog: ",
        (catalog.alias, Style(color=NordColor.NORD14.value, bold=True, link=catalog.url)),
    )
    console.print(catalog_link)
    console.print()


def _target_description(mode: str, project_root: str | None) -> str:
    if mode == "uv-project" and project_root is not None:
        return f"current uv project ({project_root})"
    return "current Python environment"


def _format_runtime_plugins(entries: list[PluginCatalogEntry]) -> str:
    return ", ".join(f"{entry.name} ({entry.plugin_type.value})" for entry in entries)


def _escape_markup(value: object) -> str:
    return escape(str(value))


def _format_checkmark(value: bool) -> str:
    return CHECKMARK if value else ""


def _format_compatibility_marker(compatibility: CompatibilityResult) -> str:
    return CHECKMARK if compatibility.is_compatible else X_MARK


def _format_installed_marker(
    package_entries: list[PluginCatalogEntry],
    installed_plugins: list[InstalledPluginInfo],
) -> str:
    return _format_checkmark(_package_entries_are_installed(package_entries, installed_plugins))


def _package_entries_are_installed(
    package_entries: list[PluginCatalogEntry],
    installed_plugins: list[InstalledPluginInfo],
) -> bool:
    installed_entry_points = {(plugin.name, plugin.entry_point_value) for plugin in installed_plugins}
    return bool(package_entries) and all(
        (entry.entry_point.name, entry.entry_point.value) in installed_entry_points for entry in package_entries
    )


def _package_alias(package_name: str) -> str | None:
    canonical_package_name = canonicalize_name(package_name)
    if not canonical_package_name.startswith(DATA_DESIGNER_PLUGIN_PACKAGE_PREFIX):
        return None
    return canonical_package_name.removeprefix(DATA_DESIGNER_PLUGIN_PACKAGE_PREFIX)


def _plugin_package_command(command_name: str, package_alias: str, catalog_alias: str) -> list[str]:
    command = ["data-designer", "plugin"]
    if catalog_alias != DEFAULT_PLUGIN_CATALOG_ALIAS:
        command.extend(["--catalog", catalog_alias])
    command.extend([command_name, package_alias])
    return command


def _console_width() -> int:
    width = getattr(console, "width", None)
    return width if isinstance(width, int) else 120


def _format_docs_link(docs_url: str | None) -> Text:
    if not docs_url:
        return Text("")
    return Text("docs", style=Style(color=NordColor.NORD7.value, link=docs_url))


def _runtime_plugin_metadata(entry: PluginCatalogEntry) -> dict[str, object]:
    return {
        "name": entry.name,
        "plugin_type": entry.plugin_type.value,
        "entry_point": entry.entry_point.model_dump(mode="json", exclude_none=True),
    }
