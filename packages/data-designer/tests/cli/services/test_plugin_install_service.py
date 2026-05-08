# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
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
        "data-designer-text-transform @ git+https://github.com/NVIDIA-NeMo/DataDesignerPlugins.git"
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


def test_build_git_install_plan_requires_ref_and_subdirectory() -> None:
    entry = _entry(
        source={
            "type": "git",
            "url": "https://github.com/NVIDIA-NeMo/DataDesignerPlugins.git",
        }
    )
    tap = PluginTapConfig(alias="local", url="/catalog/plugins.json")
    service = PluginInstallService()

    with pytest.raises(ValueError, match="requires 'ref'"):
        service.build_install_plan(entry, tap, manager="pip")


def test_build_path_install_plan_resolves_relative_path_from_local_tap_root(tmp_path: Path) -> None:
    entry = _entry(
        source={
            "type": "path",
            "path": "plugins/data-designer-text-transform",
            "editable": True,
        }
    )
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


@patch("data_designer.cli.services.plugin_install_service.importlib.metadata.entry_points")
@patch("data_designer.cli.services.plugin_install_service.importlib.invalidate_caches")
def test_verify_entry_point_invalidates_caches_and_checks_declared_entry_point(
    mock_invalidate_caches: Mock,
    mock_entry_points: Mock,
) -> None:
    entry = _entry(
        source={"type": "pypi", "package": "data-designer-text-transform"},
        plugin_name="text-transform-v2",
        entry_point_name="text-transform",
    )
    mock_entry_points.return_value = [
        SimpleNamespace(name="other-plugin"),
        SimpleNamespace(name="text-transform"),
    ]
    service = PluginInstallService()

    assert service.verify_entry_point(entry) is True
    mock_invalidate_caches.assert_called_once_with()
    mock_entry_points.assert_called_once_with(group="data_designer.plugins")


def _entry(
    source: dict | None,
    *,
    plugin_name: str = "text-transform",
    entry_point_name: str = "text-transform",
) -> PluginCatalogEntry:
    payload = {
        "name": plugin_name,
        "plugin_type": "processor",
        "description": "Transform text records",
        "package": {
            "name": "data-designer-text-transform",
            "version": "0.1.0",
            "path": "plugins/data-designer-text-transform",
        },
        "entry_point": {
            "group": "data_designer.plugins",
            "name": entry_point_name,
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
        "source": source,
        "docs": {
            "url": "https://docs.example.test/plugins/data-designer-text-transform/",
        },
    }
    return PluginCatalogEntry.model_validate(payload)
