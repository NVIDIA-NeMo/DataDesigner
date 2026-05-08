# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from data_designer.cli.plugin_catalog import PluginCatalogConfig, PluginCatalogEntry
from data_designer.cli.services.plugin_install_service import PluginInstallService


def test_build_pip_install_plan_uses_requirement_and_extra_index() -> None:
    entry = _entry(
        package_name="data-designer-template",
        install={
            "requirement": "data-designer-template",
            "index_url": "https://nvidia-nemo.github.io/DataDesignerPlugins/simple/",
        },
    )
    catalog = PluginCatalogConfig(
        alias="nvidia", url="https://nvidia-nemo.github.io/DataDesignerPlugins/catalog/plugins.json"
    )
    service = PluginInstallService()

    plan = service.build_install_plan(entry, catalog, manager="pip")

    assert plan.command == [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--extra-index-url",
        "https://nvidia-nemo.github.io/DataDesignerPlugins/simple/",
        "data-designer-template",
    ]
    assert plan.source_description == (
        "data-designer-template via https://nvidia-nemo.github.io/DataDesignerPlugins/simple/"
    )


def test_build_direct_reference_install_plan_uses_requirement_verbatim() -> None:
    requirement = (
        "data-designer-template @ "
        "git+https://github.com/NVIDIA-NeMo/DataDesignerPlugins.git@data-designer-template/v0.1.0"
    )
    entry = _entry(package_name="data-designer-template", install={"requirement": requirement})
    catalog = PluginCatalogConfig(alias="local", url="/catalog/plugins.json")
    service = PluginInstallService()

    plan = service.build_install_plan(entry, catalog, manager="pip")

    assert plan.command[-1] == requirement
    assert "--extra-index-url" not in plan.command


@patch("data_designer.cli.services.plugin_install_service.shutil.which", return_value="/usr/bin/uv")
def test_build_uv_install_plan_targets_current_python_and_adds_catalog_index(mock_which: Mock) -> None:
    entry = _entry(
        package_name="data-designer-template",
        install={
            "requirement": "data-designer-template",
            "index_url": "https://nvidia-nemo.github.io/DataDesignerPlugins/simple/",
        },
    )
    catalog = PluginCatalogConfig(
        alias="nvidia", url="https://nvidia-nemo.github.io/DataDesignerPlugins/catalog/plugins.json"
    )
    service = PluginInstallService()

    plan = service.build_install_plan(entry, catalog, manager="uv")

    assert plan.command == [
        "uv",
        "pip",
        "install",
        "--python",
        sys.executable,
        "--default-index",
        "https://pypi.org/simple/",
        "--index",
        "https://nvidia-nemo.github.io/DataDesignerPlugins/simple/",
        "data-designer-template",
    ]


def test_install_raises_when_runner_fails() -> None:
    service = PluginInstallService(runner=lambda command: 2)
    entry = _entry(package_name="data-designer-template", install={"requirement": "data-designer-template"})
    catalog = PluginCatalogConfig(alias="local", url="/catalog/plugins.json")
    plan = service.build_install_plan(entry, catalog, manager="pip")

    with pytest.raises(RuntimeError, match="status 2"):
        service.install(plan)


@patch("data_designer.cli.services.plugin_install_service.importlib.metadata.entry_points")
@patch("data_designer.cli.services.plugin_install_service.importlib.invalidate_caches")
def test_verify_entry_point_invalidates_caches_and_checks_declared_entry_point(
    mock_invalidate_caches: Mock,
    mock_entry_points: Mock,
) -> None:
    entry = _entry(
        package_name="data-designer-template",
        plugin_name="text-transform-v2",
        entry_point_name="text-transform",
        install={"requirement": "data-designer-template"},
    )
    mock_entry_points.return_value = [
        SimpleNamespace(name="other-plugin"),
        SimpleNamespace(name="text-transform"),
    ]
    service = PluginInstallService()

    assert service.verify_entry_point(entry) is True
    mock_invalidate_caches.assert_called_once_with()
    mock_entry_points.assert_called_once_with(group="data_designer.plugins")


@patch("data_designer.cli.services.plugin_install_service.importlib.metadata.entry_points")
def test_verify_entry_points_requires_every_declared_entry_point(mock_entry_points: Mock) -> None:
    entries = [
        _entry(
            package_name="data-designer-retrieval-sdg",
            plugin_name="document-chunker",
            entry_point_name="document-chunker",
            install={"requirement": "data-designer-retrieval-sdg"},
        ),
        _entry(
            package_name="data-designer-retrieval-sdg",
            plugin_name="embedding-dedup",
            entry_point_name="embedding-dedup",
            install={"requirement": "data-designer-retrieval-sdg"},
        ),
    ]
    mock_entry_points.return_value = [SimpleNamespace(name="document-chunker")]
    service = PluginInstallService()

    assert service.verify_entry_points(entries) is False


def _entry(
    *,
    package_name: str,
    install: dict,
    plugin_name: str = "text-transform",
    entry_point_name: str = "text-transform",
) -> PluginCatalogEntry:
    payload = {
        "name": plugin_name,
        "plugin_type": "processor",
        "description": "Transform text records",
        "package": {
            "name": package_name,
        },
        "install": install,
        "entry_point": {
            "group": "data_designer.plugins",
            "name": entry_point_name,
            "value": "data_designer_template.plugin:plugin",
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
    return PluginCatalogEntry.model_validate(payload)
