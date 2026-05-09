# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from data_designer.cli.controllers.plugin_catalog_controller import PluginCatalogController
from data_designer.cli.plugin_catalog import (
    CompatibilityResult,
    InstallPlan,
    PluginCatalogConfig,
    PluginCatalogEntry,
    PluginCatalogError,
    UninstallPlan,
)


@pytest.fixture
def controller(tmp_path: Path) -> PluginCatalogController:
    plugin_controller = PluginCatalogController(tmp_path)
    plugin_controller.catalog_service = MagicMock()
    plugin_controller.install_service = MagicMock()
    return plugin_controller


@patch("data_designer.cli.controllers.plugin_catalog_controller.console")
@patch("data_designer.cli.controllers.plugin_catalog_controller.print_info")
@patch("data_designer.cli.controllers.plugin_catalog_controller.print_warning")
def test_run_list_mentions_hidden_incompatible_packages_when_visible_list_is_empty(
    mock_print_warning: MagicMock,
    mock_print_info: MagicMock,
    mock_console: MagicMock,
    controller: PluginCatalogController,
) -> None:
    entry = _entry()
    catalog = _catalog(trusted=True)
    controller.catalog_service.get_catalog.return_value = catalog
    controller.catalog_service.list_entries.side_effect = [[], [entry]]

    controller.run_list(catalog_alias="local")

    assert controller.catalog_service.list_entries.call_args_list == [
        call("local", refresh=False, include_incompatible=False),
        call("local", refresh=False, include_incompatible=True),
    ]
    mock_print_warning.assert_called_once_with("No compatible plugin packages found")
    mock_print_info.assert_any_call(
        "Incompatible catalog packages are hidden. Use --include-incompatible to show them."
    )
    assert mock_console.print.call_count >= 1


@patch("data_designer.cli.controllers.plugin_catalog_controller.console")
@patch("data_designer.cli.controllers.plugin_catalog_controller.print_info")
@patch("data_designer.cli.controllers.plugin_catalog_controller.print_warning")
def test_run_search_mentions_hidden_incompatible_packages_when_visible_matches_are_empty(
    mock_print_warning: MagicMock,
    mock_print_info: MagicMock,
    mock_console: MagicMock,
    controller: PluginCatalogController,
) -> None:
    entry = _entry()
    catalog = _catalog(trusted=True)
    controller.catalog_service.get_catalog.return_value = catalog
    controller.catalog_service.search_entries.side_effect = [[], [entry]]

    controller.run_search("text", catalog_alias="local")

    assert controller.catalog_service.search_entries.call_args_list == [
        call("text", "local", refresh=False, include_incompatible=False),
        call("text", "local", refresh=False, include_incompatible=True),
    ]
    mock_print_warning.assert_called_once_with("No compatible plugin packages matched")
    mock_print_info.assert_any_call(
        "Matching incompatible catalog packages are hidden. Use --include-incompatible to show them."
    )
    assert mock_console.print.call_count >= 1


@patch("data_designer.cli.controllers.plugin_catalog_controller.console")
def test_run_list_renders_package_first_catalog_table(
    mock_console: MagicMock,
    controller: PluginCatalogController,
) -> None:
    package_entries = [
        _entry(name="text-column", plugin_type="column-generator"),
        _entry(name="text-processor", plugin_type="processor"),
    ]
    catalog = _catalog(trusted=True)
    controller.catalog_service.get_catalog.return_value = catalog
    controller.catalog_service.list_entries.return_value = package_entries
    controller.catalog_service.group_entries_by_package.return_value = {
        "data-designer-text-transform": package_entries,
    }
    controller.catalog_service.evaluate_compatibility.return_value = CompatibilityResult(True, [])

    controller.run_list(catalog_alias="local", include_incompatible=True)

    printed_tables = [
        call.args[0] for call in mock_console.print.call_args_list if call.args and isinstance(call.args[0], Table)
    ]
    assert printed_tables
    assert printed_tables[0].title == "Catalog Plugin Packages"
    assert [column.header for column in printed_tables[0].columns] == [
        "Package",
        "Description",
        "Runtime Plugins",
        "Compatible",
        "Docs",
    ]
    assert list(printed_tables[0].columns[1].cells) == ["Transform text records"]
    docs_cell = list(printed_tables[0].columns[4].cells)[0]
    assert isinstance(docs_cell, Text)
    assert docs_cell.plain == "docs"
    assert docs_cell.style is not None
    assert docs_cell.style.link == "https://docs.example.test/plugins/data-designer-text-transform/"

    rendered_output = StringIO()
    narrow_console = Console(
        file=rendered_output,
        force_terminal=True,
        color_system="standard",
        width=60,
        legacy_windows=False,
    )
    narrow_console.print(printed_tables[0])
    assert "https://docs.example.test/plugins/data-designer-text-transform/" in rendered_output.getvalue()
    controller.catalog_service.group_entries_by_package.assert_called_once_with(package_entries)


@patch("data_designer.cli.controllers.plugin_catalog_controller.console")
@patch("data_designer.cli.controllers.plugin_catalog_controller.display_config_preview")
def test_run_info_renders_package_metadata_with_nested_runtime_plugins(
    mock_display_config_preview: MagicMock,
    mock_console: MagicMock,
    controller: PluginCatalogController,
) -> None:
    package_entries = [
        _entry(name="text-column", plugin_type="column-generator"),
        _entry(name="text-processor", plugin_type="processor"),
    ]
    catalog = _catalog(trusted=True)
    controller.catalog_service.get_catalog.return_value = catalog
    controller.catalog_service.get_package_entries.return_value = package_entries
    controller.catalog_service.evaluate_compatibility.return_value = CompatibilityResult(True, [])
    controller.install_service.build_install_plan.return_value = _plan(catalog)

    controller.run_info("text-transform", catalog_alias="local")

    metadata = mock_display_config_preview.call_args.args[0]
    assert metadata["package"] == {
        "name": "data-designer-text-transform",
        "description": "Transform text records",
    }
    assert metadata["install"] == {
        "requirement": "data-designer-text-transform",
        "index_url": "https://docs.example.test/simple/",
    }
    assert metadata["plugins"] == [
        {
            "name": "text-column",
            "plugin_type": "column-generator",
            "entry_point": {
                "group": "data_designer.plugins",
                "name": "text-column",
                "value": "data_designer_text_transform.plugin:plugin",
            },
        },
        {
            "name": "text-processor",
            "plugin_type": "processor",
            "entry_point": {
                "group": "data_designer.plugins",
                "name": "text-processor",
                "value": "data_designer_text_transform.plugin:plugin",
            },
        },
    ]
    assert all("package" not in plugin for plugin in metadata["plugins"])
    assert all("install" not in plugin for plugin in metadata["plugins"])
    assert all("compatibility" not in plugin for plugin in metadata["plugins"])
    assert all("docs" not in plugin for plugin in metadata["plugins"])
    controller.catalog_service.get_package_entries.assert_called_once_with(
        "text-transform",
        "local",
        refresh=False,
        include_incompatible=True,
    )
    mock_display_config_preview.assert_called_once()
    assert mock_console.print.call_count >= 1


@patch("data_designer.cli.controllers.plugin_catalog_controller.print_warning")
@patch("data_designer.cli.controllers.plugin_catalog_controller.console")
@patch("data_designer.cli.controllers.plugin_catalog_controller.display_config_preview")
def test_run_info_warns_when_install_plan_has_source_warning(
    mock_display_config_preview: MagicMock,
    mock_console: MagicMock,
    mock_print_warning: MagicMock,
    controller: PluginCatalogController,
) -> None:
    entry = _entry()
    catalog = _catalog(trusted=True)
    controller.catalog_service.get_catalog.return_value = catalog
    controller.catalog_service.get_package_entries.return_value = [entry]
    controller.catalog_service.evaluate_compatibility.return_value = CompatibilityResult(True, [])
    controller.install_service.build_install_plan.return_value = _plan(
        catalog,
        source_warning="pip source warning",
    )

    controller.run_info("text-transform", catalog_alias="local")

    mock_print_warning.assert_called_once_with("pip source warning")
    mock_display_config_preview.assert_called_once()
    assert mock_console.print.call_count >= 1


@patch("data_designer.cli.controllers.plugin_catalog_controller.print_error")
def test_run_info_rejects_runtime_plugin_name_that_is_not_package_alias(
    mock_print_error: MagicMock,
    controller: PluginCatalogController,
) -> None:
    catalog = _catalog(trusted=True)
    controller.catalog_service.get_catalog.return_value = catalog
    controller.catalog_service.get_package_entries.return_value = []

    with pytest.raises(typer.Exit) as exc_info:
        controller.run_info("text-column", catalog_alias="local")

    assert exc_info.value.exit_code == 1
    controller.catalog_service.get_package_entries.assert_called_once_with(
        "text-column",
        "local",
        refresh=False,
        include_incompatible=True,
    )
    mock_print_error.assert_called_once_with("Plugin package or alias 'text-column' was not found in catalog 'local'")


@patch("data_designer.cli.controllers.plugin_catalog_controller.console")
@patch("data_designer.cli.controllers.plugin_catalog_controller.print_info")
def test_run_install_dry_run_renders_plan_without_installing(
    mock_print_info: MagicMock,
    mock_console: MagicMock,
    controller: PluginCatalogController,
) -> None:
    entry = _entry()
    catalog = _catalog(trusted=True)
    plan = _plan(catalog, data_designer_protection="pinned installed Data Designer packages; data-designer 0.5.10")
    controller.catalog_service.get_catalog.return_value = catalog
    controller.catalog_service.get_package_entries.return_value = [entry]
    controller.catalog_service.evaluate_compatibility.return_value = CompatibilityResult(True, [])
    controller.install_service.build_install_plan.return_value = plan

    controller.run_install("data-designer-text-transform", catalog_alias="local", dry_run=True)

    controller.catalog_service.get_package_entries.assert_called_once_with(
        "data-designer-text-transform",
        "local",
        refresh=False,
        include_incompatible=True,
    )
    controller.install_service.install.assert_not_called()
    controller.install_service.verify_entry_points.assert_not_called()
    mock_print_info.assert_any_call("Dry run complete; no changes made")
    mock_console.print.assert_any_call(
        "  Data Designer: [bold]pinned installed Data Designer packages; data-designer 0.5.10[/bold]"
    )
    assert all("Runtime plugins" not in str(call_args.args[0]) for call_args in mock_console.print.call_args_list)
    assert mock_console.print.call_count >= 1


@patch("data_designer.cli.controllers.plugin_catalog_controller.console")
@patch("data_designer.cli.controllers.plugin_catalog_controller.print_error")
def test_run_install_blocks_incompatible_package(
    mock_print_error: MagicMock,
    mock_console: MagicMock,
    controller: PluginCatalogController,
) -> None:
    entry = _entry()
    catalog = _catalog(trusted=True)
    controller.catalog_service.get_catalog.return_value = catalog
    controller.catalog_service.get_package_entries.return_value = [entry]
    controller.catalog_service.evaluate_compatibility.return_value = CompatibilityResult(
        False,
        ["Data Designer 0.5.7 does not satisfy >=99.0"],
    )

    with pytest.raises(typer.Exit) as exc_info:
        controller.run_install("data-designer-text-transform", catalog_alias="local")

    assert exc_info.value.exit_code == 1
    controller.catalog_service.get_package_entries.assert_called_once_with(
        "data-designer-text-transform",
        "local",
        refresh=False,
        include_incompatible=True,
    )
    controller.install_service.build_install_plan.assert_not_called()
    mock_print_error.assert_called_once_with(
        "Plugin package 'data-designer-text-transform' is not compatible with this environment"
    )
    mock_console.print.assert_any_call("  - Data Designer 0.5.7 does not satisfy >=99.0")


@patch("data_designer.cli.controllers.plugin_catalog_controller.print_error")
def test_run_install_rejects_runtime_plugin_name_as_target(
    mock_print_error: MagicMock,
    controller: PluginCatalogController,
) -> None:
    catalog = _catalog(trusted=True)
    controller.catalog_service.get_catalog.return_value = catalog
    controller.catalog_service.get_package_entries.return_value = []

    with pytest.raises(typer.Exit) as exc_info:
        controller.run_install("text-column", catalog_alias="local")

    assert exc_info.value.exit_code == 1
    controller.catalog_service.get_package_entries.assert_called_once_with(
        "text-column",
        "local",
        refresh=False,
        include_incompatible=True,
    )
    controller.install_service.build_install_plan.assert_not_called()
    mock_print_error.assert_called_once_with("Plugin package or alias 'text-column' was not found in catalog 'local'")


@patch("data_designer.cli.controllers.plugin_catalog_controller.console")
@patch("data_designer.cli.controllers.plugin_catalog_controller.print_warning")
def test_run_install_dry_run_renders_incompatible_plan_and_block_message(
    mock_print_warning: MagicMock,
    mock_console: MagicMock,
    controller: PluginCatalogController,
) -> None:
    entry = _entry()
    catalog = _catalog(trusted=True)
    controller.catalog_service.get_catalog.return_value = catalog
    controller.catalog_service.get_package_entries.return_value = [entry]
    controller.catalog_service.evaluate_compatibility.return_value = CompatibilityResult(
        False,
        ["Data Designer 0.5.7 does not satisfy >=99.0"],
    )
    controller.install_service.build_install_plan.return_value = _plan(catalog)

    controller.run_install("data-designer-text-transform", catalog_alias="local", dry_run=True)

    controller.install_service.build_install_plan.assert_called_once_with(entry, catalog, manager="auto")
    controller.install_service.install.assert_not_called()
    controller.install_service.verify_entry_points.assert_not_called()
    mock_console.print.assert_any_call("  Command: [bold]python -m pip install data-designer-text-transform[/bold]")
    mock_console.print.assert_any_call("  Compatibility: [bold yellow]not compatible[/bold yellow]")
    mock_console.print.assert_any_call("    - Data Designer 0.5.7 does not satisfy >=99.0")
    mock_print_warning.assert_called_once_with(
        "Dry run complete; no changes made. A real install would be blocked because compatibility checks failed."
    )


@patch("data_designer.cli.controllers.plugin_catalog_controller.console")
@patch("data_designer.cli.controllers.plugin_catalog_controller.print_error")
@patch("data_designer.cli.controllers.plugin_catalog_controller.print_warning")
def test_run_install_dry_run_allows_incompatible_entry_for_inspection(
    mock_print_warning: MagicMock,
    mock_print_error: MagicMock,
    mock_console: MagicMock,
    controller: PluginCatalogController,
) -> None:
    entry = _entry()
    catalog = _catalog(trusted=True)
    controller.catalog_service.get_catalog.return_value = catalog
    controller.catalog_service.get_package_entries.return_value = [entry]
    controller.catalog_service.evaluate_compatibility.return_value = CompatibilityResult(
        False,
        ["Data Designer 0.5.7 does not satisfy >=99.0"],
    )
    controller.install_service.build_install_plan.return_value = _plan(catalog)

    controller.run_install("data-designer-text-transform", catalog_alias="local", dry_run=True)

    controller.catalog_service.get_package_entries.assert_called_once_with(
        "data-designer-text-transform",
        "local",
        refresh=False,
        include_incompatible=True,
    )
    controller.install_service.build_install_plan.assert_called_once_with(entry, catalog, manager="auto")
    controller.install_service.install.assert_not_called()
    mock_print_error.assert_not_called()
    mock_print_warning.assert_called_once_with(
        "Dry run complete; no changes made. A real install would be blocked because compatibility checks failed."
    )
    assert mock_console.print.call_count >= 1


@patch("data_designer.cli.controllers.plugin_catalog_controller.console")
@patch("data_designer.cli.controllers.plugin_catalog_controller.print_warning")
def test_run_install_warns_when_install_plan_has_source_warning(
    mock_print_warning: MagicMock,
    mock_console: MagicMock,
    controller: PluginCatalogController,
) -> None:
    entry = _entry()
    catalog = _catalog(trusted=True)
    controller.catalog_service.get_catalog.return_value = catalog
    controller.catalog_service.get_package_entries.return_value = [entry]
    controller.catalog_service.evaluate_compatibility.return_value = CompatibilityResult(True, [])
    controller.install_service.build_install_plan.return_value = _plan(
        catalog,
        source_warning="pip source warning",
    )

    controller.run_install("data-designer-text-transform", catalog_alias="local", dry_run=True)

    mock_print_warning.assert_called_once_with("pip source warning")
    assert mock_console.print.call_count >= 1


@patch("data_designer.cli.controllers.plugin_catalog_controller.console")
@patch("data_designer.cli.controllers.plugin_catalog_controller.print_warning")
def test_run_install_warns_for_untrusted_catalog(
    mock_print_warning: MagicMock,
    mock_console: MagicMock,
    controller: PluginCatalogController,
) -> None:
    entry = _entry()
    catalog = _catalog(trusted=False)
    controller.catalog_service.get_catalog.return_value = catalog
    controller.catalog_service.get_package_entries.return_value = [entry]
    controller.catalog_service.evaluate_compatibility.return_value = CompatibilityResult(True, [])
    controller.install_service.build_install_plan.return_value = _plan(catalog)

    controller.run_install("data-designer-text-transform", catalog_alias="local", dry_run=True)

    mock_print_warning.assert_called_once_with(
        "This catalog is not marked trusted. Plugin package installation executes Python package code from "
        "the requirement above."
    )
    assert mock_console.print.call_count >= 1


@patch("data_designer.cli.controllers.plugin_catalog_controller.console")
@patch("data_designer.cli.controllers.plugin_catalog_controller.print_success")
def test_run_install_reports_success_when_verification_finds_entry_point(
    mock_print_success: MagicMock,
    mock_console: MagicMock,
    controller: PluginCatalogController,
) -> None:
    entry = _entry()
    catalog = _catalog(trusted=True)
    plan = _plan(catalog)
    controller.catalog_service.get_catalog.return_value = catalog
    controller.catalog_service.get_package_entries.return_value = [entry]
    controller.catalog_service.evaluate_compatibility.return_value = CompatibilityResult(True, [])
    controller.install_service.build_install_plan.return_value = plan
    controller.install_service.verify_entry_points.return_value = True

    controller.run_install("data-designer-text-transform", catalog_alias="local", yes=True)

    controller.install_service.install.assert_called_once_with(plan)
    controller.install_service.verify_entry_points.assert_called_once_with([entry])
    mock_print_success.assert_called_once_with("Plugin package 'data-designer-text-transform' installed and registered")
    assert mock_console.print.call_count >= 1


@patch("data_designer.cli.controllers.plugin_catalog_controller.console")
@patch("data_designer.cli.controllers.plugin_catalog_controller.print_warning")
def test_run_install_warns_when_verification_misses_entry_point(
    mock_print_warning: MagicMock,
    mock_console: MagicMock,
    controller: PluginCatalogController,
) -> None:
    entry = _entry()
    catalog = _catalog(trusted=True)
    plan = _plan(catalog)
    controller.catalog_service.get_catalog.return_value = catalog
    controller.catalog_service.get_package_entries.return_value = [entry]
    controller.catalog_service.evaluate_compatibility.return_value = CompatibilityResult(True, [])
    controller.install_service.build_install_plan.return_value = plan
    controller.install_service.verify_entry_points.return_value = False

    controller.run_install("data-designer-text-transform", catalog_alias="local", yes=True)

    controller.install_service.install.assert_called_once_with(plan)
    controller.install_service.verify_entry_points.assert_called_once_with([entry])
    mock_print_warning.assert_called_once_with(
        "Plugin package 'data-designer-text-transform' was installed, but Data Designer did not discover every "
        "declared package entry point. Restart the shell or check the package entry point metadata."
    )
    assert mock_console.print.call_count >= 1


@patch("data_designer.cli.controllers.plugin_catalog_controller.console")
@patch("data_designer.cli.controllers.plugin_catalog_controller.print_info")
def test_run_uninstall_dry_run_renders_plan_without_uninstalling(
    mock_print_info: MagicMock,
    mock_console: MagicMock,
    controller: PluginCatalogController,
) -> None:
    entry = _entry()
    catalog = _catalog(trusted=True)
    plan = _uninstall_plan(catalog)
    controller.catalog_service.get_catalog.return_value = catalog
    controller.catalog_service.get_package_entries.return_value = [entry]
    controller.install_service.build_uninstall_plan.return_value = plan

    controller.run_uninstall("data-designer-text-transform", catalog_alias="local", dry_run=True)

    controller.catalog_service.get_package_entries.assert_called_once_with(
        "data-designer-text-transform",
        "local",
        refresh=False,
        include_incompatible=True,
    )
    controller.install_service.build_uninstall_plan.assert_called_once_with(entry, catalog, manager="auto")
    controller.install_service.uninstall.assert_not_called()
    controller.install_service.verify_entry_points_removed.assert_not_called()
    mock_console.print.assert_any_call(
        "  Command: [bold]python -m pip uninstall --yes data-designer-text-transform[/bold]"
    )
    assert all("Runtime plugins" not in str(call_args.args[0]) for call_args in mock_console.print.call_args_list)
    mock_print_info.assert_any_call("Dry run complete; no changes made")


@patch("data_designer.cli.controllers.plugin_catalog_controller.print_error")
def test_run_uninstall_wraps_plan_error(
    mock_print_error: MagicMock,
    controller: PluginCatalogController,
) -> None:
    entry = _entry()
    catalog = _catalog(trusted=True)
    controller.catalog_service.get_catalog.return_value = catalog
    controller.catalog_service.get_package_entries.return_value = [entry]
    controller.install_service.build_uninstall_plan.side_effect = ValueError("uv was requested")

    with pytest.raises(typer.Exit) as exc_info:
        controller.run_uninstall("data-designer-text-transform", catalog_alias="local")

    assert exc_info.value.exit_code == 1
    controller.install_service.uninstall.assert_not_called()
    mock_print_error.assert_called_once_with("Failed to build plugin uninstall plan: uv was requested")


@patch("data_designer.cli.controllers.plugin_catalog_controller.console")
@patch("data_designer.cli.controllers.plugin_catalog_controller.print_success")
def test_run_uninstall_reports_success_when_entry_points_are_removed(
    mock_print_success: MagicMock,
    mock_console: MagicMock,
    controller: PluginCatalogController,
) -> None:
    entry = _entry()
    catalog = _catalog(trusted=True)
    plan = _uninstall_plan(catalog)
    controller.catalog_service.get_catalog.return_value = catalog
    controller.catalog_service.get_package_entries.return_value = [entry]
    controller.install_service.build_uninstall_plan.return_value = plan
    controller.install_service.verify_entry_points_removed.return_value = True

    controller.run_uninstall("data-designer-text-transform", catalog_alias="local", yes=True)

    controller.install_service.uninstall.assert_called_once_with(plan)
    controller.install_service.verify_entry_points_removed.assert_called_once_with([entry])
    mock_print_success.assert_called_once_with(
        "Plugin package 'data-designer-text-transform' uninstalled and no longer registered"
    )
    assert mock_console.print.call_count >= 1


@patch("data_designer.cli.controllers.plugin_catalog_controller.console")
@patch("data_designer.cli.controllers.plugin_catalog_controller.print_warning")
def test_run_uninstall_warns_when_entry_points_remain(
    mock_print_warning: MagicMock,
    mock_console: MagicMock,
    controller: PluginCatalogController,
) -> None:
    entry = _entry()
    catalog = _catalog(trusted=True)
    plan = _uninstall_plan(catalog)
    controller.catalog_service.get_catalog.return_value = catalog
    controller.catalog_service.get_package_entries.return_value = [entry]
    controller.install_service.build_uninstall_plan.return_value = plan
    controller.install_service.verify_entry_points_removed.return_value = False

    controller.run_uninstall("data-designer-text-transform", catalog_alias="local", yes=True)

    controller.install_service.uninstall.assert_called_once_with(plan)
    controller.install_service.verify_entry_points_removed.assert_called_once_with([entry])
    mock_print_warning.assert_called_once_with(
        "Plugin package 'data-designer-text-transform' was uninstalled, but Data Designer still discovers one or "
        "more declared package entry points. Restart the shell or check the package environment."
    )
    assert mock_console.print.call_count >= 1


@patch("data_designer.cli.controllers.plugin_catalog_controller.print_error")
def test_run_catalog_add_wraps_invalid_alias_validation_error(
    mock_print_error: MagicMock,
    tmp_path: Path,
) -> None:
    plugin_controller = PluginCatalogController(tmp_path)

    with pytest.raises(typer.Exit) as exc_info:
        plugin_controller.run_catalog_add(
            alias="foo/bar",
            url="https://github.com/acme/dd-plugins",
            trusted=False,
            cache_ttl_seconds=60,
        )

    assert exc_info.value.exit_code == 1
    mock_print_error.assert_called_once_with("Invalid catalog alias 'foo/bar': must match `^[A-Za-z0-9_.-]+$`")


@patch("data_designer.cli.controllers.plugin_catalog_controller.print_error")
def test_run_catalog_list_wraps_registry_load_error(
    mock_print_error: MagicMock,
    controller: PluginCatalogController,
) -> None:
    controller.catalog_service.list_catalogs.side_effect = PluginCatalogError("bad registry")

    with pytest.raises(typer.Exit) as exc_info:
        controller.run_catalog_list()

    assert exc_info.value.exit_code == 1
    mock_print_error.assert_called_once_with("Failed to list plugin catalogs: bad registry")


def _catalog(*, trusted: bool) -> PluginCatalogConfig:
    return PluginCatalogConfig(
        alias="local",
        url="https://raw.githubusercontent.com/acme/dd-plugins/main/catalog/plugins.json",
        trusted=trusted,
    )


def _plan(
    catalog: PluginCatalogConfig,
    *,
    source_warning: str | None = None,
    data_designer_protection: str | None = None,
) -> InstallPlan:
    return InstallPlan(
        package_name="data-designer-text-transform",
        source_description="data-designer-text-transform",
        command=["python", "-m", "pip", "install", "data-designer-text-transform"],
        manager="pip",
        catalog_alias=catalog.alias,
        trusted_catalog=catalog.trusted,
        source_warning=source_warning,
        data_designer_protection=data_designer_protection,
    )


def _uninstall_plan(catalog: PluginCatalogConfig) -> UninstallPlan:
    return UninstallPlan(
        package_name="data-designer-text-transform",
        command=["python", "-m", "pip", "uninstall", "--yes", "data-designer-text-transform"],
        manager="pip",
        catalog_alias=catalog.alias,
    )


def _entry(
    *,
    name: str = "text-transform",
    plugin_type: str = "processor",
    package_name: str = "data-designer-text-transform",
) -> PluginCatalogEntry:
    return PluginCatalogEntry.model_validate(
        {
            "name": name,
            "plugin_type": plugin_type,
            "description": "Transform text records",
            "package": {
                "name": package_name,
            },
            "install": {
                "requirement": package_name,
                "index_url": "https://docs.example.test/simple/",
            },
            "entry_point": {
                "group": "data_designer.plugins",
                "name": name,
                "value": "data_designer_text_transform.plugin:plugin",
            },
            "compatibility": {
                "python": {"specifier": ">=3.10"},
                "data_designer": {
                    "requirement": "data-designer>=0.5.7",
                    "specifier": ">=0.5.7",
                    "marker": None,
                },
            },
            "docs": {
                "url": f"https://docs.example.test/plugins/{package_name}/",
            },
        }
    )
