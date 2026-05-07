# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest

from data_designer.cli.plugin_catalog import DEFAULT_PLUGIN_TAP_ALIAS, PluginCatalogError
from data_designer.cli.repositories.plugin_tap_repository import PluginTapRepository, normalize_tap_location


def test_repository_includes_default_nvidia_tap(tmp_path: Path) -> None:
    repository = PluginTapRepository(tmp_path)

    taps = repository.list_taps()

    assert [tap.alias for tap in taps] == [DEFAULT_PLUGIN_TAP_ALIAS]
    assert taps[0].trusted is True


def test_add_tap_normalizes_github_repository_url(tmp_path: Path) -> None:
    repository = PluginTapRepository(tmp_path)

    tap = repository.add_tap("research", "https://github.com/acme/dd-plugins")

    assert tap.url == "https://raw.githubusercontent.com/acme/dd-plugins/main/catalog/plugins.json"
    assert repository.get_tap("research") == tap


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


def test_load_catalog_rejects_unsupported_schema_version(tmp_path: Path) -> None:
    catalog_path = _write_catalog(tmp_path, schema_version=999)
    repository = PluginTapRepository(tmp_path)
    repository.add_tap("local", str(catalog_path))

    with pytest.raises(PluginCatalogError, match="Unsupported plugin catalog schema_version"):
        repository.load_catalog("local", refresh=True)


def _write_catalog(tmp_path: Path, *, schema_version: int = 2, plugin_name: str = "text-transform") -> Path:
    catalog_dir = tmp_path / "catalog"
    catalog_dir.mkdir()
    catalog_path = catalog_dir / "plugins.json"
    catalog_path.write_text(json.dumps(_catalog_payload(schema_version=schema_version, plugin_name=plugin_name)))
    return catalog_path


def _catalog_payload(*, schema_version: int = 2, plugin_name: str = "text-transform") -> dict:
    return {
        "schema_version": schema_version,
        "plugins": [
            {
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
                    "name": plugin_name,
                    "value": "data_designer_text_transform.plugin:plugin",
                },
                "compatibility": {
                    "python": {"specifier": ">=3.10"},
                    "data_designer": {"specifier": ">=0.5.7"},
                },
                "source": {
                    "type": "pypi",
                    "package": "data-designer-text-transform",
                },
                "docs": {
                    "url": "https://example.com/text-transform",
                },
            },
        ],
    }
