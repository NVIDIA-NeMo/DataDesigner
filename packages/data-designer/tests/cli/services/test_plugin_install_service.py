# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from data_designer.cli.plugin_catalog import PluginCatalogEntry, PluginTapConfig
from data_designer.cli.services.plugin_install_service import PluginInstallService


def test_build_pypi_install_plan_uses_exact_catalog_version() -> None:
    entry = _entry(source={"type": "pypi", "package": "data-designer-text-transform"})
    tap = PluginTapConfig(alias="local", url="/catalog/plugins.json")
    service = PluginInstallService()

    plan = service.build_install_plan(entry, tap, manager="pip")

    assert plan.command == [
        sys.executable,
        "-m",
        "pip",
        "install",
        "data-designer-text-transform==0.1.0",
    ]
    assert plan.source_description == "data-designer-text-transform==0.1.0"


def test_build_git_install_plan_includes_ref_and_subdirectory() -> None:
    entry = _entry(
        source={
            "type": "git",
            "url": "https://github.com/NVIDIA-NeMo/DataDesignerPlugins.git",
            "ref": "data-designer-text-transform/v0.1.0",
            "subdirectory": "plugins/data-designer-text-transform",
        }
    )
    tap = PluginTapConfig(alias="local", url="/catalog/plugins.json")
    service = PluginInstallService()

    plan = service.build_install_plan(entry, tap, manager="pip")

    assert plan.command[-1] == (
        "git+https://github.com/NVIDIA-NeMo/DataDesignerPlugins.git"
        "@data-designer-text-transform/v0.1.0"
        "#subdirectory=plugins/data-designer-text-transform"
    )


@patch("data_designer.cli.services.plugin_install_service.shutil.which", return_value="/usr/bin/uv")
def test_build_uv_install_plan_targets_current_python(mock_which: Mock) -> None:
    entry = _entry(source={"type": "pypi", "package": "data-designer-text-transform"})
    tap = PluginTapConfig(alias="local", url="/catalog/plugins.json")
    service = PluginInstallService()

    plan = service.build_install_plan(entry, tap, manager="uv")

    assert plan.command[:5] == ["uv", "pip", "install", "--python", sys.executable]


def test_build_path_install_plan_resolves_relative_path_from_local_tap_root(tmp_path: Path) -> None:
    entry = _entry(source={"type": "path", "editable": True})
    tap = PluginTapConfig(alias="local", url=str(tmp_path / "catalog" / "plugins.json"))
    service = PluginInstallService()

    plan = service.build_install_plan(entry, tap, manager="pip")

    assert plan.command == [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-e",
        str(tmp_path / "plugins" / "data-designer-text-transform"),
    ]


def test_build_install_plan_requires_source() -> None:
    entry = _entry(source=None)
    tap = PluginTapConfig(alias="local", url="/catalog/plugins.json")
    service = PluginInstallService()

    with pytest.raises(ValueError, match="does not declare a source"):
        service.build_install_plan(entry, tap, manager="pip")


def test_install_raises_when_runner_fails() -> None:
    service = PluginInstallService(runner=lambda command: 2)
    entry = _entry(source={"type": "pypi", "package": "data-designer-text-transform"})
    tap = PluginTapConfig(alias="local", url="/catalog/plugins.json")
    plan = service.build_install_plan(entry, tap, manager="pip")

    with pytest.raises(RuntimeError, match="status 2"):
        service.install(plan)


@patch("data_designer.cli.services.plugin_install_service.PluginRegistry")
def test_verify_entry_point_resets_and_checks_runtime_registry(mock_registry_cls: Mock) -> None:
    entry = _entry(source={"type": "pypi", "package": "data-designer-text-transform"})
    mock_registry = Mock()
    mock_registry.plugin_exists.return_value = True
    mock_registry_cls.return_value = mock_registry
    service = PluginInstallService()

    assert service.verify_entry_point(entry) is True
    mock_registry_cls.reset.assert_called_once_with()
    mock_registry.plugin_exists.assert_called_once_with("text-transform")


def _entry(source: dict | None) -> PluginCatalogEntry:
    payload = {
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
            "data_designer": {"specifier": ">=0.5.7"},
        },
        "source": source,
    }
    return PluginCatalogEntry.model_validate(payload)
