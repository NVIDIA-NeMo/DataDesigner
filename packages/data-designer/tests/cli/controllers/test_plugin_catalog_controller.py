# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer

from data_designer.cli.controllers.plugin_catalog_controller import PluginCatalogController
from data_designer.cli.plugin_catalog import (
    CompatibilityResult,
    InstallPlan,
    PluginCatalogConfig,
    PluginCatalogEntry,
)


@pytest.fixture
def controller(tmp_path: Path) -> PluginCatalogController:
    plugin_controller = PluginCatalogController(tmp_path)
    plugin_controller.catalog_service = MagicMock()
    plugin_controller.install_service = MagicMock()
    return plugin_controller


@patch("data_designer.cli.controllers.plugin_catalog_controller.console")
@patch("data_designer.cli.controllers.plugin_catalog_controller.print_info")
def test_run_install_dry_run_renders_plan_without_installing(
    mock_print_info: MagicMock,
    mock_console: MagicMock,
    controller: PluginCatalogController,
) -> None:
    entry = _entry()
    catalog = _catalog(trusted=True)
    plan = _plan(catalog)
    controller.catalog_service.get_catalog.return_value = catalog
    controller.catalog_service.get_entry.return_value = entry
    controller.catalog_service.get_package_entries.return_value = [entry]
    controller.catalog_service.evaluate_compatibility.return_value = CompatibilityResult(True, [])
    controller.install_service.build_install_plan.return_value = plan

    controller.run_install("text-transform", catalog_alias="local", dry_run=True)

    controller.catalog_service.get_entry.assert_called_once_with(
        "text-transform",
        "local",
        refresh=False,
        include_incompatible=True,
    )
    controller.install_service.install.assert_not_called()
    controller.install_service.verify_entry_points.assert_not_called()
    mock_print_info.assert_any_call("Dry run complete; no changes made")
    assert mock_console.print.call_count >= 1


@patch("data_designer.cli.controllers.plugin_catalog_controller.console")
@patch("data_designer.cli.controllers.plugin_catalog_controller.print_error")
def test_run_install_blocks_incompatible_plugin_without_force(
    mock_print_error: MagicMock,
    mock_console: MagicMock,
    controller: PluginCatalogController,
) -> None:
    entry = _entry()
    catalog = _catalog(trusted=True)
    controller.catalog_service.get_catalog.return_value = catalog
    controller.catalog_service.get_entry.return_value = entry
    controller.catalog_service.get_package_entries.return_value = [entry]
    controller.catalog_service.evaluate_compatibility.return_value = CompatibilityResult(
        False,
        ["Data Designer 0.5.7 does not satisfy >=99.0"],
    )

    with pytest.raises(typer.Exit) as exc_info:
        controller.run_install("text-transform", catalog_alias="local")

    assert exc_info.value.exit_code == 1
    controller.catalog_service.get_entry.assert_called_once_with(
        "text-transform",
        "local",
        refresh=False,
        include_incompatible=True,
    )
    controller.install_service.build_install_plan.assert_not_called()
    mock_print_error.assert_called_once_with("Plugin 'text-transform' is not compatible with this environment")
    mock_console.print.assert_any_call("  - Data Designer 0.5.7 does not satisfy >=99.0")


@patch("data_designer.cli.controllers.plugin_catalog_controller.console")
@patch("data_designer.cli.controllers.plugin_catalog_controller.print_error")
@patch("data_designer.cli.controllers.plugin_catalog_controller.print_info")
def test_run_install_force_allows_incompatible_entry_for_dry_run(
    mock_print_info: MagicMock,
    mock_print_error: MagicMock,
    mock_console: MagicMock,
    controller: PluginCatalogController,
) -> None:
    entry = _entry()
    catalog = _catalog(trusted=True)
    controller.catalog_service.get_catalog.return_value = catalog
    controller.catalog_service.get_entry.return_value = entry
    controller.catalog_service.get_package_entries.return_value = [entry]
    controller.catalog_service.evaluate_compatibility.return_value = CompatibilityResult(
        False,
        ["Data Designer 0.5.7 does not satisfy >=99.0"],
    )
    controller.install_service.build_install_plan.return_value = _plan(catalog)

    controller.run_install("text-transform", catalog_alias="local", dry_run=True, force=True)

    controller.catalog_service.get_entry.assert_called_once_with(
        "text-transform",
        "local",
        refresh=False,
        include_incompatible=True,
    )
    controller.install_service.build_install_plan.assert_called_once_with(entry, catalog, manager="auto")
    controller.install_service.install.assert_not_called()
    mock_print_error.assert_not_called()
    mock_print_info.assert_any_call("Dry run complete; no changes made")
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
    controller.catalog_service.get_entry.return_value = entry
    controller.catalog_service.get_package_entries.return_value = [entry]
    controller.catalog_service.evaluate_compatibility.return_value = CompatibilityResult(True, [])
    controller.install_service.build_install_plan.return_value = _plan(catalog)

    controller.run_install("text-transform", catalog_alias="local", dry_run=True)

    mock_print_warning.assert_called_once_with(
        "This catalog is not marked trusted. Plugin installation executes Python package code from the requirement above."
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
    controller.catalog_service.get_entry.return_value = entry
    controller.catalog_service.get_package_entries.return_value = [entry]
    controller.catalog_service.evaluate_compatibility.return_value = CompatibilityResult(True, [])
    controller.install_service.build_install_plan.return_value = plan
    controller.install_service.verify_entry_points.return_value = True

    controller.run_install("text-transform", catalog_alias="local", yes=True)

    controller.install_service.install.assert_called_once_with(plan)
    controller.install_service.verify_entry_points.assert_called_once_with([entry])
    mock_print_success.assert_called_once_with("Plugin package 'data-designer-text-transform' installed and discovered")
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
    controller.catalog_service.get_entry.return_value = entry
    controller.catalog_service.get_package_entries.return_value = [entry]
    controller.catalog_service.evaluate_compatibility.return_value = CompatibilityResult(True, [])
    controller.install_service.build_install_plan.return_value = plan
    controller.install_service.verify_entry_points.return_value = False

    controller.run_install("text-transform", catalog_alias="local", yes=True)

    controller.install_service.install.assert_called_once_with(plan)
    controller.install_service.verify_entry_points.assert_called_once_with([entry])
    mock_print_warning.assert_called_once_with(
        "Plugin package 'data-designer-text-transform' was installed, but Data Designer did not discover every "
        "declared entry point. "
        "Restart the shell or check the package entry point metadata."
    )
    assert mock_console.print.call_count >= 1


@patch("data_designer.cli.controllers.plugin_catalog_controller.print_error")
def test_run_catalogs_add_wraps_invalid_alias_validation_error(
    mock_print_error: MagicMock,
    tmp_path: Path,
) -> None:
    plugin_controller = PluginCatalogController(tmp_path)

    with pytest.raises(typer.Exit) as exc_info:
        plugin_controller.run_catalogs_add(
            alias="foo/bar",
            url="https://github.com/acme/dd-plugins",
            trusted=False,
            cache_ttl_seconds=60,
        )

    assert exc_info.value.exit_code == 1
    mock_print_error.assert_called_once_with("Invalid catalog alias 'foo/bar': must match `^[A-Za-z0-9_.-]+$`")


def _catalog(*, trusted: bool) -> PluginCatalogConfig:
    return PluginCatalogConfig(
        alias="local",
        url="https://raw.githubusercontent.com/acme/dd-plugins/main/catalog/plugins.json",
        trusted=trusted,
    )


def _plan(catalog: PluginCatalogConfig) -> InstallPlan:
    return InstallPlan(
        plugin_name="text-transform",
        package_name="data-designer-text-transform",
        source_description="data-designer-text-transform",
        command=["python", "-m", "pip", "install", "data-designer-text-transform"],
        manager="pip",
        catalog_alias=catalog.alias,
        trusted_catalog=catalog.trusted,
    )


def _entry() -> PluginCatalogEntry:
    return PluginCatalogEntry.model_validate(
        {
            "name": "text-transform",
            "plugin_type": "processor",
            "description": "Transform text records",
            "package": {
                "name": "data-designer-text-transform",
            },
            "install": {
                "requirement": "data-designer-text-transform",
                "index_url": "https://docs.example.test/simple/",
            },
            "entry_point": {
                "group": "data_designer.plugins",
                "name": "text-transform",
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
                "url": "https://docs.example.test/plugins/data-designer-text-transform/",
            },
        }
    )
