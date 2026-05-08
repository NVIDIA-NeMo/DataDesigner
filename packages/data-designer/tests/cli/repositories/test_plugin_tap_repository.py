# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest

from data_designer.cli.plugin_catalog import (
    DEFAULT_PLUGIN_TAP_ALIAS,
    DEFAULT_PLUGIN_TAP_URL_ENV_VAR,
    PluginCatalogError,
)
from data_designer.cli.repositories.plugin_tap_repository import PluginTapRepository, normalize_tap_location


def test_repository_includes_default_nvidia_tap(tmp_path: Path) -> None:
    repository = PluginTapRepository(tmp_path)

    taps = repository.list_taps()

    assert [tap.alias for tap in taps] == [DEFAULT_PLUGIN_TAP_ALIAS]
    assert taps[0].trusted is True


def test_default_tap_honors_url_environment_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(DEFAULT_PLUGIN_TAP_URL_ENV_VAR, "https://example.test/catalog/plugins.json")
    repository = PluginTapRepository(tmp_path)

    tap = repository.default_tap()

    assert tap.url == "https://example.test/catalog/plugins.json"


def test_add_tap_normalizes_github_repository_url(tmp_path: Path) -> None:
    repository = PluginTapRepository(tmp_path)

    tap = repository.add_tap("research", "https://github.com/acme/dd-plugins")

    assert tap.url == "https://raw.githubusercontent.com/acme/dd-plugins/main/catalog/plugins.json"
    assert repository.get_tap("research") == tap


def test_add_tap_normalizes_github_tree_url_with_subdirectory(tmp_path: Path) -> None:
    repository = PluginTapRepository(tmp_path)

    tap = repository.add_tap("research", "https://github.com/acme/dd-plugins/tree/main/custom-catalog")

    assert tap.url == "https://raw.githubusercontent.com/acme/dd-plugins/main/custom-catalog/catalog/plugins.json"


def test_tap_aliases_are_case_insensitive(tmp_path: Path) -> None:
    repository = PluginTapRepository(tmp_path)

    tap = repository.add_tap("Research", "https://github.com/acme/dd-plugins")

    assert repository.get_tap("research") == tap
    with pytest.raises(ValueError, match="already exists"):
        repository.add_tap("research", "https://github.com/acme/other-plugins")
    with pytest.raises(ValueError, match="already exists"):
        repository.add_tap("NVIDIA", "https://github.com/acme/nvidia-plugins")

    repository.remove_tap("research")

    assert repository.get_tap("Research") is None


def test_normalize_local_tap_directory() -> None:
    normalized = normalize_tap_location("~/plugins")

    assert normalized.endswith("/plugins/catalog/plugins.json")


def test_load_catalog_uses_cache_when_source_is_unavailable(tmp_path: Path) -> None:
    catalog_path = _write_catalog(tmp_path)
    repository = PluginTapRepository(tmp_path)
    repository.add_tap("local", str(catalog_path))

    first_catalog = repository.load_catalog("local")
    catalog_path.unlink()
    cached_catalog = repository.load_catalog("local")

    assert first_catalog.plugins[0].name == "text-transform"
    assert cached_catalog.plugins[0].name == "text-transform"


def test_load_catalog_with_zero_cache_ttl_refreshes_source(tmp_path: Path) -> None:
    catalog_path = _write_catalog(tmp_path, plugin_name="text-transform")
    repository = PluginTapRepository(tmp_path)
    repository.add_tap("local", str(catalog_path), cache_ttl_seconds=0)

    first_catalog = repository.load_catalog("local")
    catalog_path.write_text(json.dumps(_catalog_payload(plugin_name="fresh-transform")))
    refreshed_catalog = repository.load_catalog("local")

    assert first_catalog.plugins[0].name == "text-transform"
    assert refreshed_catalog.plugins[0].name == "fresh-transform"


def test_load_catalog_cache_file_is_keyed_by_alias_and_url(tmp_path: Path) -> None:
    catalog_path = _write_catalog(tmp_path)
    repository = PluginTapRepository(tmp_path)
    repository.add_tap("local", str(catalog_path))

    repository.load_catalog("local")

    cache_files = list(repository.cache_dir.glob("*.json"))
    assert len(cache_files) == 1
    assert cache_files[0].name.startswith("local-")
    assert cache_files[0].name != "local.json"


def test_load_catalog_rejects_unsupported_schema_version(tmp_path: Path) -> None:
    catalog_path = _write_catalog(tmp_path, schema_version=999)
    repository = PluginTapRepository(tmp_path)
    repository.add_tap("local", str(catalog_path))

    with pytest.raises(PluginCatalogError, match="unsupported catalog schema_version"):
        repository.load_catalog("local", refresh=True)


def test_load_catalog_accepts_schema_v2_source_union(tmp_path: Path) -> None:
    catalog_path = _write_catalog(
        tmp_path,
        plugins=[
            _plugin_entry(
                "pypi-plugin",
                package_name="data-designer-pypi-plugin",
                source={"type": "pypi", "package": "data-designer-pypi-plugin"},
            ),
            _plugin_entry(
                "git-plugin",
                package_name="data-designer-git-plugin",
                source={
                    "type": "git",
                    "url": "https://github.com/NVIDIA-NeMo/DataDesignerPlugins.git",
                    "ref": "data-designer-git-plugin/v0.1.0",
                    "subdirectory": "plugins/data-designer-git-plugin",
                },
            ),
            _plugin_entry(
                "path-plugin",
                package_name="data-designer-path-plugin",
                source={
                    "type": "path",
                    "path": "plugins/data-designer-path-plugin",
                    "editable": True,
                },
            ),
        ],
    )
    repository = PluginTapRepository(tmp_path)
    repository.add_tap("local", str(catalog_path))

    catalog = repository.load_catalog("local", refresh=True)

    assert [entry.name for entry in catalog.plugins] == ["pypi-plugin", "git-plugin", "path-plugin"]


def test_load_catalog_rejects_invalid_schema_v2_source(tmp_path: Path) -> None:
    catalog_path = _write_catalog(
        tmp_path,
        plugins=[
            _plugin_entry(
                "invalid-git-source",
                package_name="data-designer-invalid-git-source",
                source={
                    "type": "git",
                    "url": "https://github.com/NVIDIA-NeMo/DataDesignerPlugins.git",
                },
            )
        ],
    )
    repository = PluginTapRepository(tmp_path)
    repository.add_tap("local", str(catalog_path))

    with pytest.raises(PluginCatalogError, match="invalid 'git' source fields"):
        repository.load_catalog("local", refresh=True)


def test_load_catalog_rejects_unexpected_schema_v2_fields(tmp_path: Path) -> None:
    plugin = _plugin_entry("text-transform")
    plugin["tags"] = ["extra"]
    catalog_path = _write_catalog(tmp_path, plugins=[plugin])
    repository = PluginTapRepository(tmp_path)
    repository.add_tap("local", str(catalog_path))

    with pytest.raises(PluginCatalogError, match="catalog plugins\\[0\\] has invalid fields"):
        repository.load_catalog("local", refresh=True)


def test_load_catalog_rejects_duplicate_runtime_plugin_names(tmp_path: Path) -> None:
    catalog_path = _write_catalog(
        tmp_path,
        plugins=[
            _plugin_entry("catalog-one", package_name="data-designer-one", entry_point_name="duplicate"),
            _plugin_entry("catalog-two", package_name="data-designer-two", entry_point_name="duplicate"),
        ],
    )
    repository = PluginTapRepository(tmp_path)
    repository.add_tap("local", str(catalog_path))

    with pytest.raises(PluginCatalogError, match="duplicate runtime plugin name"):
        repository.load_catalog("local", refresh=True)


def _write_catalog(
    tmp_path: Path,
    *,
    schema_version: int = 2,
    plugin_name: str = "text-transform",
    plugins: list[dict] | None = None,
) -> Path:
    catalog_dir = tmp_path / "catalog"
    catalog_dir.mkdir()
    catalog_path = catalog_dir / "plugins.json"
    catalog_path.write_text(
        json.dumps(_catalog_payload(schema_version=schema_version, plugin_name=plugin_name, plugins=plugins))
    )
    return catalog_path


def _catalog_payload(
    *,
    schema_version: int = 2,
    plugin_name: str = "text-transform",
    plugins: list[dict] | None = None,
) -> dict:
    return {
        "schema_version": schema_version,
        "plugins": plugins if plugins is not None else [_plugin_entry(plugin_name)],
    }


def _plugin_entry(
    plugin_name: str,
    *,
    package_name: str = "data-designer-text-transform",
    entry_point_name: str | None = None,
    source: dict | None = None,
) -> dict:
    runtime_plugin_name = plugin_name if entry_point_name is None else entry_point_name
    return {
        "name": plugin_name,
        "plugin_type": "processor",
        "description": "Transform text records",
        "package": {
            "name": package_name,
            "version": "0.1.0",
            "path": f"plugins/{package_name}",
        },
        "entry_point": {
            "group": "data_designer.plugins",
            "name": runtime_plugin_name,
            "value": f"{package_name.replace('-', '_')}.plugin:plugin",
        },
        "compatibility": {
            "python": {"specifier": ">=3.10"},
            "data_designer": {
                "requirement": "data-designer>=0.5.7",
                "specifier": ">=0.5.7",
                "marker": None,
            },
        },
        "source": source
        or {
            "type": "pypi",
            "package": package_name,
        },
        "docs": {
            "url": f"https://docs.example.test/plugins/{package_name}/",
        },
    }
