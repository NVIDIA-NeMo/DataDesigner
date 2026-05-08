# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest

from data_designer.cli.plugin_catalog import (
    DEFAULT_PLUGIN_CATALOG_ALIAS,
    DEFAULT_PLUGIN_CATALOG_URL_ENV_VAR,
    PluginCatalogError,
)
from data_designer.cli.repositories.plugin_catalog_repository import PluginCatalogRepository, normalize_catalog_location


def test_repository_includes_default_nvidia_catalog(tmp_path: Path) -> None:
    repository = PluginCatalogRepository(tmp_path)

    catalogs = repository.list_catalogs()

    assert [catalog.alias for catalog in catalogs] == [DEFAULT_PLUGIN_CATALOG_ALIAS]
    assert catalogs[0].trusted is True


def test_default_catalog_honors_url_environment_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(DEFAULT_PLUGIN_CATALOG_URL_ENV_VAR, "https://example.test/catalog/plugins.json")
    repository = PluginCatalogRepository(tmp_path)

    catalog = repository.default_catalog()

    assert catalog.url == "https://example.test/catalog/plugins.json"


def test_add_catalog_normalizes_github_repository_url(tmp_path: Path) -> None:
    repository = PluginCatalogRepository(tmp_path)

    catalog = repository.add_catalog("research", "https://github.com/acme/dd-plugins")

    assert catalog.url == "https://raw.githubusercontent.com/acme/dd-plugins/main/catalog/plugins.json"
    assert repository.get_catalog("research") == catalog


def test_add_catalog_normalizes_github_tree_url_with_subdirectory(tmp_path: Path) -> None:
    repository = PluginCatalogRepository(tmp_path)

    catalog = repository.add_catalog("research", "https://github.com/acme/dd-plugins/tree/main/custom-catalog")

    assert catalog.url == "https://raw.githubusercontent.com/acme/dd-plugins/main/custom-catalog/catalog/plugins.json"


def test_catalog_aliases_are_case_insensitive(tmp_path: Path) -> None:
    repository = PluginCatalogRepository(tmp_path)

    catalog = repository.add_catalog("Research", "https://github.com/acme/dd-plugins")

    assert repository.get_catalog("research") == catalog
    with pytest.raises(ValueError, match="already exists"):
        repository.add_catalog("research", "https://github.com/acme/other-plugins")
    with pytest.raises(ValueError, match="already exists"):
        repository.add_catalog("NVIDIA", "https://github.com/acme/nvidia-plugins")

    repository.remove_catalog("research")

    assert repository.get_catalog("Research") is None


def test_normalize_local_catalog_directory() -> None:
    normalized = normalize_catalog_location("~/plugins")

    assert normalized.endswith("/plugins/catalog/plugins.json")


def test_load_catalog_uses_cache_when_source_is_unavailable(tmp_path: Path) -> None:
    catalog_path = _write_catalog(tmp_path)
    repository = PluginCatalogRepository(tmp_path)
    repository.add_catalog("local", str(catalog_path))

    first_catalog = repository.load_catalog("local")
    catalog_path.unlink()
    cached_catalog = repository.load_catalog("local")

    assert first_catalog.plugins[0].name == "text-transform"
    assert cached_catalog.plugins[0].name == "text-transform"


def test_load_catalog_with_zero_cache_ttl_refreshes_source(tmp_path: Path) -> None:
    catalog_path = _write_catalog(tmp_path, plugin_name="text-transform")
    repository = PluginCatalogRepository(tmp_path)
    repository.add_catalog("local", str(catalog_path), cache_ttl_seconds=0)

    first_catalog = repository.load_catalog("local")
    catalog_path.write_text(json.dumps(_catalog_payload(plugin_name="fresh-transform")))
    refreshed_catalog = repository.load_catalog("local")

    assert first_catalog.plugins[0].name == "text-transform"
    assert refreshed_catalog.plugins[0].name == "fresh-transform"


def test_load_catalog_cache_file_is_keyed_by_alias_and_url(tmp_path: Path) -> None:
    catalog_path = _write_catalog(tmp_path)
    repository = PluginCatalogRepository(tmp_path)
    repository.add_catalog("local", str(catalog_path))

    repository.load_catalog("local")

    cache_files = list(repository.cache_dir.glob("*.json"))
    assert len(cache_files) == 1
    assert cache_files[0].name.startswith("local-")
    assert cache_files[0].name != "local.json"


def test_load_catalog_rejects_unsupported_schema_version(tmp_path: Path) -> None:
    catalog_path = _write_catalog(tmp_path, schema_version=999)
    repository = PluginCatalogRepository(tmp_path)
    repository.add_catalog("local", str(catalog_path))

    with pytest.raises(PluginCatalogError, match="unsupported catalog schema_version"):
        repository.load_catalog("local", refresh=True)


def test_load_catalog_accepts_schema_v2_package_catalog(tmp_path: Path) -> None:
    catalog_path = _write_catalog(
        tmp_path,
        packages=[
            _package_entry(
                package_name="data-designer-index-package",
                plugins=[
                    _runtime_plugin("index-column", plugin_type="column-generator"),
                    _runtime_plugin("index-processor", plugin_type="processor"),
                ],
                install={
                    "requirement": "data-designer-index-package",
                    "index_url": "https://docs.example.test/simple/",
                },
            ),
            _package_entry(
                package_name="data-designer-git-plugin",
                plugins=[_runtime_plugin("git-plugin", plugin_type="seed-reader")],
                install={
                    "requirement": (
                        "data-designer-git-plugin @ "
                        "git+https://github.com/NVIDIA-NeMo/DataDesignerPlugins.git@"
                        "data-designer-git-plugin/v0.1.0"
                    ),
                },
            ),
            _package_entry(
                package_name="data-designer-url-plugin",
                plugins=[_runtime_plugin("url-plugin", plugin_type="processor")],
                install={
                    "requirement": (
                        "data-designer-url-plugin @ "
                        "https://packages.example.test/data_designer_url_plugin-0.1.0-py3-none-any.whl"
                    ),
                },
            ),
        ],
    )
    repository = PluginCatalogRepository(tmp_path)
    repository.add_catalog("local", str(catalog_path))

    catalog = repository.load_catalog("local", refresh=True)

    assert [package.name for package in catalog.packages] == [
        "data-designer-index-package",
        "data-designer-git-plugin",
        "data-designer-url-plugin",
    ]
    assert [entry.name for entry in catalog.plugins] == [
        "index-column",
        "index-processor",
        "git-plugin",
        "url-plugin",
    ]
    assert catalog.plugins[0].install.index_url == "https://docs.example.test/simple/"


def test_load_catalog_rejects_invalid_schema_v2_install_metadata(tmp_path: Path) -> None:
    catalog_path = _write_catalog(
        tmp_path,
        packages=[
            _package_entry(
                package_name="data-designer-invalid-install",
                plugins=[_runtime_plugin("invalid-install")],
                install={
                    "requirement": "data-designer-other",
                    "index_url": "https://docs.example.test/simple/",
                },
            )
        ],
    )
    repository = PluginCatalogRepository(tmp_path)
    repository.add_catalog("local", str(catalog_path))

    with pytest.raises(PluginCatalogError, match="expected a requirement for 'data-designer-invalid-install'"):
        repository.load_catalog("local", refresh=True)


def test_load_catalog_rejects_unexpected_schema_v2_fields(tmp_path: Path) -> None:
    package = _package_entry()
    package["tags"] = ["extra"]
    catalog_path = _write_catalog(tmp_path, packages=[package])
    repository = PluginCatalogRepository(tmp_path)
    repository.add_catalog("local", str(catalog_path))

    with pytest.raises(PluginCatalogError, match="catalog packages\\[0\\] has invalid fields"):
        repository.load_catalog("local", refresh=True)


def test_load_catalog_rejects_duplicate_runtime_plugin_names(tmp_path: Path) -> None:
    catalog_path = _write_catalog(
        tmp_path,
        packages=[
            _package_entry(
                package_name="data-designer-one",
                plugins=[_runtime_plugin("duplicate", entry_point_name="first-entry")],
            ),
            _package_entry(
                package_name="data-designer-two",
                plugins=[_runtime_plugin("duplicate", entry_point_name="second-entry")],
            ),
        ],
    )
    repository = PluginCatalogRepository(tmp_path)
    repository.add_catalog("local", str(catalog_path))

    with pytest.raises(PluginCatalogError, match="duplicate runtime plugin name"):
        repository.load_catalog("local", refresh=True)


def _write_catalog(
    tmp_path: Path,
    *,
    schema_version: int = 2,
    plugin_name: str = "text-transform",
    packages: list[dict] | None = None,
) -> Path:
    catalog_dir = tmp_path / "catalog"
    catalog_dir.mkdir()
    catalog_path = catalog_dir / "plugins.json"
    catalog_path.write_text(
        json.dumps(_catalog_payload(schema_version=schema_version, plugin_name=plugin_name, packages=packages))
    )
    return catalog_path


def _catalog_payload(
    *,
    schema_version: int = 2,
    plugin_name: str = "text-transform",
    packages: list[dict] | None = None,
) -> dict:
    return {
        "schema_version": schema_version,
        "packages": packages if packages is not None else [_package_entry(plugins=[_runtime_plugin(plugin_name)])],
    }


def _package_entry(
    *,
    package_name: str = "data-designer-text-transform",
    plugins: list[dict] | None = None,
    install: dict | None = None,
) -> dict:
    return {
        "name": package_name,
        "description": f"{package_name} package",
        "install": install
        or {
            "requirement": package_name,
            "index_url": "https://docs.example.test/simple/",
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
        "plugins": plugins if plugins is not None else [_runtime_plugin("text-transform")],
    }


def _runtime_plugin(
    plugin_name: str,
    *,
    plugin_type: str = "processor",
    entry_point_name: str | None = None,
) -> dict:
    runtime_entry_point_name = plugin_name if entry_point_name is None else entry_point_name
    return {
        "name": plugin_name,
        "plugin_type": plugin_type,
        "entry_point": {
            "group": "data_designer.plugins",
            "name": runtime_entry_point_name,
            "value": f"data_designer_{plugin_name.replace('-', '_')}.plugin:plugin",
        },
    }
