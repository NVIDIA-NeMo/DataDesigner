# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from data_designer.cli.repositories.plugin_tap_repository import PluginTapRepository
from data_designer.cli.services.plugin_catalog_service import PluginCatalogService
from data_designer.plugins.plugin import PluginType


def test_list_entries_filters_incompatible_plugins_by_default(tmp_path: Path) -> None:
    repository = _repository_with_catalog(tmp_path)
    service = PluginCatalogService(repository, python_version="3.11.0", data_designer_version="0.5.7")

    entries = service.list_entries("local")
    all_entries = service.list_entries("local", include_incompatible=True)

    assert [entry.name for entry in entries] == [
        "compatible-plugin",
        "shared-column",
        "shared-processor",
    ]
    assert [entry.name for entry in all_entries] == [
        "compatible-plugin",
        "future-plugin",
        "shared-column",
        "shared-processor",
    ]


def test_search_entries_matches_name_type_package_and_docs(tmp_path: Path) -> None:
    repository = _repository_with_catalog(tmp_path)
    service = PluginCatalogService(repository, python_version="3.11.0", data_designer_version="0.5.7")

    name_matches = service.search_entries("compatible", "local")
    package_matches = service.search_entries("shared-package", "local")
    type_matches = service.search_entries("seed-reader", "local")

    assert [entry.name for entry in name_matches] == ["compatible-plugin"]
    assert [entry.name for entry in package_matches] == ["shared-column", "shared-processor"]
    assert [entry.name for entry in type_matches] == ["compatible-plugin"]


def test_evaluate_compatibility_reports_data_designer_constraint(tmp_path: Path) -> None:
    repository = _repository_with_catalog(tmp_path)
    service = PluginCatalogService(repository, python_version="3.11.0", data_designer_version="0.5.7")
    entry = service.get_entry("future-plugin", "local")

    result = service.evaluate_compatibility(entry)

    assert result.is_compatible is False
    assert result.reasons == ["Data Designer 0.5.7 does not satisfy >=99.0"]


def test_evaluate_compatibility_accepts_local_dev_version_above_lower_bound(tmp_path: Path) -> None:
    repository = _repository_with_catalog(tmp_path)
    service = PluginCatalogService(
        repository,
        python_version="3.11.0",
        data_designer_version="0.5.10.dev18+604fdd96",
    )
    entry = service.get_entry("compatible-plugin", "local")

    result = service.evaluate_compatibility(entry)

    assert result.is_compatible is True
    assert result.reasons == []


def test_get_entry_rejects_incompatible_plugin_when_requested(tmp_path: Path) -> None:
    repository = _repository_with_catalog(tmp_path)
    service = PluginCatalogService(repository, python_version="3.11.0", data_designer_version="0.5.7")

    with pytest.raises(ValueError, match="no compatible version"):
        service.get_entry("future-plugin", "local", include_incompatible=False)


def test_group_entries_by_package_groups_multi_plugin_packages(tmp_path: Path) -> None:
    repository = _repository_with_catalog(tmp_path)
    service = PluginCatalogService(repository, python_version="3.11.0", data_designer_version="0.5.7")
    entries = service.list_entries("local", include_incompatible=True)

    grouped_entries = service.group_entries_by_package(entries)

    assert [entry.name for entry in grouped_entries["data-designer-shared-package"]] == [
        "shared-column",
        "shared-processor",
    ]


@patch("data_designer.cli.services.plugin_catalog_service.PluginRegistry")
def test_list_installed_plugins_uses_runtime_registry(mock_registry_cls: Mock, tmp_path: Path) -> None:
    plugin = Mock()
    plugin.name = "installed-plugin"
    plugin.plugin_type = PluginType.PROCESSOR
    plugin.config_qualified_name = "pkg.config.Config"
    plugin.impl_qualified_name = "pkg.impl.Impl"
    mock_registry = Mock()
    mock_registry.get_plugins.side_effect = lambda plugin_type: [plugin] if plugin_type == PluginType.PROCESSOR else []
    mock_registry_cls.return_value = mock_registry
    service = PluginCatalogService(PluginTapRepository(tmp_path))

    installed = service.list_installed_plugins()

    assert len(installed) == 1
    assert installed[0].name == "installed-plugin"
    assert installed[0].plugin_type == PluginType.PROCESSOR


def _repository_with_catalog(tmp_path: Path) -> PluginTapRepository:
    catalog_path = tmp_path / "plugins.json"
    catalog_path.write_text(json.dumps(_catalog_payload()))
    repository = PluginTapRepository(tmp_path)
    repository.add_tap("local", str(catalog_path))
    return repository


def _catalog_payload() -> dict:
    return {
        "schema_version": 2,
        "plugins": [
            _entry(
                name="compatible-plugin",
                plugin_type="seed-reader",
                package_name="data-designer-compatible-plugin",
                data_designer_specifier=">=0.5.7",
            ),
            _entry(
                name="future-plugin",
                plugin_type="processor",
                package_name="data-designer-future-plugin",
                data_designer_specifier=">=99.0",
            ),
            _entry(
                name="shared-column",
                plugin_type="column-generator",
                package_name="data-designer-shared-package",
                data_designer_specifier=">=0.5.7",
            ),
            _entry(
                name="shared-processor",
                plugin_type="processor",
                package_name="data-designer-shared-package",
                data_designer_specifier=">=0.5.7",
            ),
        ],
    }


def _entry(
    *,
    name: str,
    plugin_type: str,
    package_name: str,
    data_designer_specifier: str,
    package_version: str = "0.1.0",
) -> dict:
    return {
        "name": name,
        "plugin_type": plugin_type,
        "description": f"{name} description",
        "package": {
            "name": package_name,
            "version": package_version,
            "path": f"plugins/{package_name}",
        },
        "entry_point": {
            "group": "data_designer.plugins",
            "name": name,
            "value": f"{package_name.replace('-', '_')}.plugin:plugin",
        },
        "compatibility": {
            "python": {"specifier": ">=3.10"},
            "data_designer": {
                "requirement": f"data-designer{data_designer_specifier}",
                "specifier": data_designer_specifier,
                "marker": None,
            },
        },
        "source": {
            "type": "pypi",
            "package": package_name,
        },
        "docs": {
            "url": f"https://docs.example.test/plugins/{package_name}/",
        },
    }
