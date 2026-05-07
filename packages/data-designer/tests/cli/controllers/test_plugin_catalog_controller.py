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
    PluginCatalogEntry,
    PluginTapConfig,
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
    tap = _tap(trusted=True)
    plan = _plan(tap)
    controller.catalog_service.get_tap.return_value = tap
    controller.catalog_service.get_entry.return_value = entry
    controller.catalog_service.evaluate_compatibility.return_value = CompatibilityResult(True, [])
    controller.install_service.build_install_plan.return_value = plan

    controller.run_install("text-transform", tap_alias="local", dry_run=True)

    controller.catalog_service.get_entry.assert_called_once_with(
        "text-transform",
        "local",
        refresh=False,
        include_incompatible=True,
    )
    controller.install_service.install.assert_not_called()
    controller.install_service.verify_entry_point.assert_not_called()
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
    tap = _tap(trusted=True)
    controller.catalog_service.get_tap.return_value = tap
    controller.catalog_service.get_entry.return_value = entry
    controller.catalog_service.evaluate_compatibility.return_value = CompatibilityResult(
        False,
        ["Data Designer 0.5.7 does not satisfy >=99.0"],
    )

    with pytest.raises(typer.Exit) as exc_info:
        controller.run_install("text-transform", tap_alias="local")

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
    tap = _tap(trusted=True)
    controller.catalog_service.get_tap.return_value = tap
    controller.catalog_service.get_entry.return_value = entry
    controller.catalog_service.evaluate_compatibility.return_value = CompatibilityResult(
        False,
        ["Data Designer 0.5.7 does not satisfy >=99.0"],
    )
    controller.install_service.build_install_plan.return_value = _plan(tap)

    controller.run_install("text-transform", tap_alias="local", dry_run=True, force=True)

    controller.catalog_service.get_entry.assert_called_once_with(
        "text-transform",
        "local",
        refresh=False,
        include_incompatible=True,
    )
    controller.install_service.build_install_plan.assert_called_once_with(entry, tap, manager="auto")
    controller.install_service.install.assert_not_called()
    mock_print_error.assert_not_called()
    mock_print_info.assert_any_call("Dry run complete; no changes made")
    assert mock_console.print.call_count >= 1


@patch("data_designer.cli.controllers.plugin_catalog_controller.console")
@patch("data_designer.cli.controllers.plugin_catalog_controller.print_warning")
def test_run_install_warns_for_untrusted_tap(
    mock_print_warning: MagicMock,
    mock_console: MagicMock,
    controller: PluginCatalogController,
) -> None:
    entry = _entry()
    tap = _tap(trusted=False)
    controller.catalog_service.get_tap.return_value = tap
    controller.catalog_service.get_entry.return_value = entry
    controller.catalog_service.evaluate_compatibility.return_value = CompatibilityResult(True, [])
    controller.install_service.build_install_plan.return_value = _plan(tap)

    controller.run_install("text-transform", tap_alias="local", dry_run=True)

    mock_print_warning.assert_called_once_with(
        "This tap is not marked trusted. Plugin installation executes Python package code from the source above."
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
    tap = _tap(trusted=True)
    plan = _plan(tap)
    controller.catalog_service.get_tap.return_value = tap
    controller.catalog_service.get_entry.return_value = entry
    controller.catalog_service.evaluate_compatibility.return_value = CompatibilityResult(True, [])
    controller.install_service.build_install_plan.return_value = plan
    controller.install_service.verify_entry_point.return_value = True

    controller.run_install("text-transform", tap_alias="local", yes=True)

    controller.install_service.install.assert_called_once_with(plan)
    controller.install_service.verify_entry_point.assert_called_once_with(entry)
    mock_print_success.assert_called_once_with("Plugin 'text-transform' installed and discovered")
    assert mock_console.print.call_count >= 1


@patch("data_designer.cli.controllers.plugin_catalog_controller.console")
@patch("data_designer.cli.controllers.plugin_catalog_controller.print_warning")
def test_run_install_warns_when_verification_misses_entry_point(
    mock_print_warning: MagicMock,
    mock_console: MagicMock,
    controller: PluginCatalogController,
) -> None:
    entry = _entry()
    tap = _tap(trusted=True)
    plan = _plan(tap)
    controller.catalog_service.get_tap.return_value = tap
    controller.catalog_service.get_entry.return_value = entry
    controller.catalog_service.evaluate_compatibility.return_value = CompatibilityResult(True, [])
    controller.install_service.build_install_plan.return_value = plan
    controller.install_service.verify_entry_point.return_value = False

    controller.run_install("text-transform", tap_alias="local", yes=True)

    controller.install_service.install.assert_called_once_with(plan)
    controller.install_service.verify_entry_point.assert_called_once_with(entry)
    mock_print_warning.assert_called_once_with(
        "Plugin 'text-transform' was installed, but Data Designer did not discover its entry point. "
        "Restart the shell or check the package entry point metadata."
    )
    assert mock_console.print.call_count >= 1


@patch("data_designer.cli.controllers.plugin_catalog_controller.print_error")
def test_run_taps_add_wraps_invalid_alias_validation_error(
    mock_print_error: MagicMock,
    tmp_path: Path,
) -> None:
    plugin_controller = PluginCatalogController(tmp_path)

    with pytest.raises(typer.Exit) as exc_info:
        plugin_controller.run_taps_add(
            alias="foo/bar",
            url="https://github.com/acme/dd-plugins",
            trusted=False,
            cache_ttl_seconds=60,
        )

    assert exc_info.value.exit_code == 1
    mock_print_error.assert_called_once_with("Invalid tap alias 'foo/bar': must match `^[A-Za-z0-9_.-]+$`")


def _tap(*, trusted: bool) -> PluginTapConfig:
    return PluginTapConfig(
        alias="local",
        url="https://raw.githubusercontent.com/acme/dd-plugins/main/catalog/plugins.json",
        trusted=trusted,
    )


def _plan(tap: PluginTapConfig) -> InstallPlan:
    return InstallPlan(
        plugin_name="text-transform",
        package_name="data-designer-text-transform",
        source_description="data-designer-text-transform==0.1.0",
        command=["python", "-m", "pip", "install", "data-designer-text-transform==0.1.0"],
        manager="pip",
        tap_alias=tap.alias,
        trusted_tap=tap.trusted,
    )


def _entry() -> PluginCatalogEntry:
    return PluginCatalogEntry.model_validate(
        {
            "name": "text-transform",
            "plugin_type": "processor",
            "description": "Transform text records",
            "package": {
                "name": "data-designer-text-transform",
                "version": "0.1.0",
                "path": "plugins/data-designer-text-transform",
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
            "source": {
                "type": "pypi",
                "package": "data-designer-text-transform",
            },
            "docs": {
                "url": "https://docs.example.test/plugins/data-designer-text-transform/",
            },
        }
    )
