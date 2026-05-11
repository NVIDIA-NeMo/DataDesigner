# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from importlib.metadata import EntryPoint
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from data_designer.cli.plugin_catalog import PluginCatalog, PluginCatalogEntry
from data_designer.cli.repositories.plugin_catalog_repository import PluginCatalogRepository
from data_designer.cli.services.plugin_catalog_service import PluginCatalogService


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


def test_search_entries_matches_package_description_name_and_type(tmp_path: Path) -> None:
    repository = _repository_with_catalog(tmp_path)
    service = PluginCatalogService(repository, python_version="3.11.0", data_designer_version="0.5.7")

    name_matches = service.search_entries("compatible", "local")
    package_matches = service.search_entries("shared-package", "local")
    type_matches = service.search_entries("seed-reader", "local")

    assert [entry.name for entry in name_matches] == ["compatible-plugin"]
    assert [entry.name for entry in package_matches] == ["shared-column", "shared-processor"]
    assert [entry.name for entry in type_matches] == ["compatible-plugin"]


def test_search_entries_matches_natural_language_synonyms(tmp_path: Path) -> None:
    package = _package(
        package_name="data-designer-github",
        data_designer_specifier=">=0.5.7",
        plugins=[_runtime_plugin(name="github", plugin_type="seed-reader")],
    )
    package["description"] = "GitHub and local git repository seed reader for Data Designer"
    catalog_path = tmp_path / "plugins.json"
    catalog_path.write_text(json.dumps({"schema_version": 2, "packages": [package]}))
    repository = PluginCatalogRepository(tmp_path)
    repository.add_catalog("local", str(catalog_path))
    service = PluginCatalogService(repository, python_version="3.11.0", data_designer_version="0.5.7")

    matches = service.search_entries("GitHub repo import", "local")

    assert [entry.package.name for entry in matches] == ["data-designer-github"]


def test_search_entries_ignores_install_docs_and_entry_point_metadata(tmp_path: Path) -> None:
    package = _package(
        package_name="data-designer-retrieval-sdg",
        data_designer_specifier=">=0.5.7",
        plugins=[_runtime_plugin(name="document-chunker", plugin_type="seed-reader")],
    )
    package["install"]["index_url"] = "https://nvidia-nemo.github.io/DataDesignerPlugins/simple/"
    package["docs"]["url"] = "https://nvidia-nemo.github.io/DataDesignerPlugins/plugins/data-designer-retrieval-sdg/"
    package["plugins"][0]["entry_point"]["value"] = "data_designer_github_noise.plugin:plugin"
    catalog_path = tmp_path / "plugins.json"
    catalog_path.write_text(json.dumps({"schema_version": 2, "packages": [package]}))
    repository = PluginCatalogRepository(tmp_path)
    repository.add_catalog("local", str(catalog_path))
    service = PluginCatalogService(repository, python_version="3.11.0", data_designer_version="0.5.7")

    matches = service.search_entries("github", "local")

    assert matches == []


def test_evaluate_compatibility_reports_data_designer_constraint(tmp_path: Path) -> None:
    repository = _repository_with_catalog(tmp_path)
    service = PluginCatalogService(repository, python_version="3.11.0", data_designer_version="0.5.7")
    entry = _entry_by_name(service.list_entries("local", include_incompatible=True), "future-plugin")

    result = service.evaluate_compatibility(entry)

    assert result.is_compatible is False
    assert result.reasons == ["Data Designer 0.5.7 does not satisfy >=99.0"]


def test_evaluate_compatibility_reports_python_constraint() -> None:
    service = PluginCatalogService(
        Mock(spec=PluginCatalogRepository),
        python_version="3.11.0",
        data_designer_version="0.5.7",
    )
    entry = PluginCatalogEntry.model_validate(
        _entry(
            name="future-python-plugin",
            plugin_type="processor",
            package_name="data-designer-future-python-plugin",
            python_specifier=">=3.12",
            data_designer_specifier=">=0.5.7",
        )
    )

    result = service.evaluate_compatibility(entry)

    assert result.is_compatible is False
    assert result.reasons == ["Python 3.11.0 does not satisfy >=3.12"]


@pytest.mark.parametrize(
    ("marker", "expected_is_compatible", "expected_reasons"),
    [
        ("python_version >= '3.12'", True, []),
        ("python_version < '3.12'", False, ["Data Designer 0.5.7 does not satisfy >=99.0"]),
    ],
)
def test_evaluate_compatibility_respects_data_designer_marker(
    marker: str,
    expected_is_compatible: bool,
    expected_reasons: list[str],
) -> None:
    service = PluginCatalogService(
        Mock(spec=PluginCatalogRepository),
        python_version="3.11.0",
        data_designer_version="0.5.7",
    )
    entry = PluginCatalogEntry.model_validate(
        _entry(
            name="marker-gated-plugin",
            plugin_type="processor",
            package_name="data-designer-marker-gated-plugin",
            data_designer_specifier=">=99.0",
            data_designer_marker=marker,
        )
    )

    result = service.evaluate_compatibility(entry)

    assert result.is_compatible is expected_is_compatible
    assert result.reasons == expected_reasons


@patch("data_designer.cli.services.plugin_catalog_service._get_installed_data_designer_version", return_value=None)
def test_evaluate_compatibility_reports_missing_data_designer_version(mock_version: Mock) -> None:
    service = PluginCatalogService(Mock(spec=PluginCatalogRepository), python_version="3.11.0")
    entry = PluginCatalogEntry.model_validate(
        _entry(
            name="compatible-plugin",
            plugin_type="processor",
            package_name="data-designer-compatible-plugin",
            data_designer_specifier=">=0.5.7",
        )
    )

    result = service.evaluate_compatibility(entry)

    assert result.is_compatible is False
    assert result.reasons == ["Unable to resolve installed Data Designer version for constraint '>=0.5.7'"]
    mock_version.assert_called_once_with()


def test_evaluate_compatibility_accepts_local_dev_version_above_lower_bound(tmp_path: Path) -> None:
    repository = _repository_with_catalog(tmp_path)
    service = PluginCatalogService(
        repository,
        python_version="3.11.0",
        data_designer_version="0.5.10.dev18+604fdd96",
    )
    entry = _entry_by_name(service.list_entries("local", include_incompatible=True), "compatible-plugin")

    result = service.evaluate_compatibility(entry)

    assert result.is_compatible is True
    assert result.reasons == []


def test_get_package_entries_resolves_package_alias() -> None:
    repository = Mock(spec=PluginCatalogRepository)
    repository.load_catalog.return_value = PluginCatalog.model_validate(
        {
            "schema_version": 2,
            "packages": [
                _package(
                    package_name="data-designer-calculator",
                    data_designer_specifier=">=0.5.7",
                    plugins=[
                        _runtime_plugin(name="arithmetic-column", plugin_type="column-generator"),
                        _runtime_plugin(name="arithmetic-processor", plugin_type="processor"),
                    ],
                ),
            ],
        }
    )
    service = PluginCatalogService(repository, python_version="3.11.0", data_designer_version="0.5.7")

    entries = service.get_package_entries("calculator", "local", include_incompatible=True)

    assert [entry.name for entry in entries] == ["arithmetic-column", "arithmetic-processor"]
    assert {entry.package.name for entry in entries} == {"data-designer-calculator"}


def test_get_package_entries_prefers_exact_package_name_over_package_alias() -> None:
    repository = Mock(spec=PluginCatalogRepository)
    repository.load_catalog.return_value = PluginCatalog.model_validate(
        {
            "schema_version": 2,
            "packages": [
                _package(
                    package_name="calculator",
                    data_designer_specifier=">=0.5.7",
                    plugins=[_runtime_plugin(name="plain-calculator", plugin_type="processor")],
                ),
                _package(
                    package_name="data-designer-calculator",
                    data_designer_specifier=">=0.5.7",
                    plugins=[_runtime_plugin(name="namespaced-calculator", plugin_type="processor")],
                ),
            ],
        }
    )
    service = PluginCatalogService(repository, python_version="3.11.0", data_designer_version="0.5.7")

    entries = service.get_package_entries("calculator", "local", include_incompatible=True)

    assert [entry.name for entry in entries] == ["plain-calculator"]
    assert entries[0].package.name == "calculator"


def test_get_package_entries_does_not_resolve_runtime_plugin_name_that_is_not_package_alias() -> None:
    repository = Mock(spec=PluginCatalogRepository)
    repository.load_catalog.return_value = PluginCatalog.model_validate(
        {
            "schema_version": 2,
            "packages": [
                _package(
                    package_name="data-designer-calculator",
                    data_designer_specifier=">=0.5.7",
                    plugins=[_runtime_plugin(name="arithmetic", plugin_type="processor")],
                ),
            ],
        }
    )
    service = PluginCatalogService(repository, python_version="3.11.0", data_designer_version="0.5.7")

    assert service.get_package_entries("arithmetic", "local", include_incompatible=True) == []


def test_get_runtime_plugin_entries_resolves_runtime_name_without_package_aliasing() -> None:
    repository = Mock(spec=PluginCatalogRepository)
    repository.load_catalog.return_value = PluginCatalog.model_validate(
        {
            "schema_version": 2,
            "packages": [
                _package(
                    package_name="data-designer-retrieval-sdg",
                    data_designer_specifier=">=0.5.7",
                    plugins=[
                        _runtime_plugin(name="document-chunker", plugin_type="seed-reader"),
                        _runtime_plugin(name="embedding-dedup", plugin_type="column-generator"),
                    ],
                ),
            ],
        }
    )
    service = PluginCatalogService(repository, python_version="3.11.0", data_designer_version="0.5.7")

    entries = service.get_runtime_plugin_entries("document-chunker", "local", include_incompatible=True)

    assert [entry.name for entry in entries] == ["document-chunker"]
    assert entries[0].package.name == "data-designer-retrieval-sdg"


def test_group_entries_by_package_groups_multi_plugin_packages(tmp_path: Path) -> None:
    repository = _repository_with_catalog(tmp_path)
    service = PluginCatalogService(repository, python_version="3.11.0", data_designer_version="0.5.7")
    entries = service.list_entries("local", include_incompatible=True)

    grouped_entries = service.group_entries_by_package(entries)

    assert [entry.name for entry in grouped_entries["data-designer-shared-package"]] == [
        "shared-column",
        "shared-processor",
    ]


def test_group_entries_by_package_canonicalizes_package_names(tmp_path: Path) -> None:
    repository = _repository_with_catalog(tmp_path)
    service = PluginCatalogService(repository, python_version="3.11.0", data_designer_version="0.5.7")
    entries = [
        PluginCatalogEntry.model_validate(
            _entry(
                name="hyphen-package",
                plugin_type="processor",
                package_name="data-designer-shared-package",
                data_designer_specifier=">=0.5.7",
            )
        ),
        PluginCatalogEntry.model_validate(
            _entry(
                name="underscore-package",
                plugin_type="processor",
                package_name="data_designer_shared_package",
                data_designer_specifier=">=0.5.7",
            )
        ),
    ]

    grouped_entries = service.group_entries_by_package(entries)

    assert list(grouped_entries) == ["data-designer-shared-package"]
    assert [entry.name for entry in grouped_entries["data-designer-shared-package"]] == [
        "hyphen-package",
        "underscore-package",
    ]


@patch("data_designer.cli.services.plugin_catalog_service.importlib.metadata.entry_points")
def test_list_installed_plugins_uses_entry_point_metadata_without_loading_plugins(
    mock_entry_points: Mock,
    tmp_path: Path,
) -> None:
    mock_entry_points.return_value = [
        EntryPoint(
            name="installed-plugin",
            value="pkg.plugin:plugin",
            group="data_designer.plugins",
        )
    ]
    service = PluginCatalogService(PluginCatalogRepository(tmp_path))

    installed = service.list_installed_plugins()

    assert len(installed) == 1
    assert installed[0].name == "installed-plugin"
    assert installed[0].entry_point_value == "pkg.plugin:plugin"
    assert installed[0].package_name is None
    assert installed[0].package_version is None
    mock_entry_points.assert_called_once_with(group="data_designer.plugins")


@patch("data_designer.cli.services.plugin_catalog_service.importlib.metadata.entry_points")
def test_list_installed_plugins_includes_distribution_metadata_when_available(
    mock_entry_points: Mock,
    tmp_path: Path,
) -> None:
    mock_entry_points.return_value = [
        SimpleNamespace(
            name="github",
            value="data_designer_github.plugin:plugin",
            dist=SimpleNamespace(metadata={"Name": "data-designer-github"}, version="0.1.0"),
        )
    ]
    service = PluginCatalogService(PluginCatalogRepository(tmp_path))

    installed = service.list_installed_plugins()

    assert len(installed) == 1
    assert installed[0].name == "github"
    assert installed[0].package_name == "data-designer-github"
    assert installed[0].package_version == "0.1.0"


def _repository_with_catalog(tmp_path: Path) -> PluginCatalogRepository:
    catalog_path = tmp_path / "plugins.json"
    catalog_path.write_text(json.dumps(_catalog_payload()))
    repository = PluginCatalogRepository(tmp_path)
    repository.add_catalog("local", str(catalog_path))
    return repository


def _entry_by_name(entries: list[PluginCatalogEntry], name: str) -> PluginCatalogEntry:
    return next(entry for entry in entries if entry.name == name)


def _catalog_payload() -> dict:
    return {
        "schema_version": 2,
        "packages": [
            _package(
                package_name="data-designer-compatible-plugin",
                data_designer_specifier=">=0.5.7",
                plugins=[_runtime_plugin(name="compatible-plugin", plugin_type="seed-reader")],
            ),
            _package(
                package_name="data-designer-future-plugin",
                data_designer_specifier=">=99.0",
                plugins=[_runtime_plugin(name="future-plugin", plugin_type="processor")],
            ),
            _package(
                package_name="data-designer-shared-package",
                data_designer_specifier=">=0.5.7",
                plugins=[
                    _runtime_plugin(name="shared-column", plugin_type="column-generator"),
                    _runtime_plugin(name="shared-processor", plugin_type="processor"),
                ],
            ),
        ],
    }


def _package(
    *,
    package_name: str,
    data_designer_specifier: str,
    plugins: list[dict],
    data_designer_marker: str | None = None,
    python_specifier: str = ">=3.10",
) -> dict:
    return {
        "name": package_name,
        "description": f"{package_name} description",
        "install": {
            "requirement": package_name,
            "index_url": "https://docs.example.test/simple/",
        },
        "compatibility": {
            "python": {"specifier": python_specifier},
            "data_designer": {
                "requirement": f"data-designer{data_designer_specifier}",
                "specifier": data_designer_specifier,
                "marker": data_designer_marker,
            },
        },
        "docs": {
            "url": f"https://docs.example.test/plugins/{package_name}/",
        },
        "plugins": plugins,
    }


def _runtime_plugin(*, name: str, plugin_type: str) -> dict:
    return {
        "name": name,
        "plugin_type": plugin_type,
        "entry_point": {
            "group": "data_designer.plugins",
            "name": name,
            "value": f"data_designer_{name.replace('-', '_')}.plugin:plugin",
        },
    }


def _entry(
    *,
    name: str,
    plugin_type: str,
    package_name: str,
    data_designer_specifier: str,
    data_designer_marker: str | None = None,
    python_specifier: str = ">=3.10",
) -> dict:
    return {
        "name": name,
        "plugin_type": plugin_type,
        "description": f"{name} description",
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
            "value": f"{package_name.replace('-', '_')}.plugin:plugin",
        },
        "compatibility": {
            "python": {"specifier": python_specifier},
            "data_designer": {
                "requirement": f"data-designer{data_designer_specifier}",
                "specifier": data_designer_specifier,
                "marker": data_designer_marker,
            },
        },
        "docs": {
            "url": f"https://docs.example.test/plugins/{package_name}/",
        },
    }
