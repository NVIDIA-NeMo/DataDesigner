# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.error import HTTPError

import pytest
from pydantic import ValidationError

from data_designer.cli.plugin_catalog import (
    DEFAULT_PLUGIN_CATALOG_ALIAS,
    DEFAULT_PLUGIN_CATALOG_URL_ENV_VAR,
    MAX_PLUGIN_CATALOG_SIZE_BYTES,
    PluginCatalog,
    PluginCatalogError,
)
from data_designer.cli.repositories.plugin_catalog_repository import PluginCatalogRepository, normalize_catalog_location

UPSTREAM_CATALOG_FIXTURES_DIR = Path(__file__).parents[1] / "fixtures" / "upstream-catalogs"


def test_repository_includes_default_nvidia_catalog(tmp_path: Path) -> None:
    repository = PluginCatalogRepository(tmp_path)

    catalogs = repository.list_catalogs()

    assert [catalog.alias for catalog in catalogs] == [DEFAULT_PLUGIN_CATALOG_ALIAS]


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


def test_add_catalog_persists_only_public_catalog_fields(tmp_path: Path) -> None:
    repository = PluginCatalogRepository(tmp_path)

    repository.add_catalog("research", "https://github.com/acme/dd-plugins")

    saved_registry = repository.config_file.read_text()
    assert "alias: research" in saved_registry
    assert "url: https://raw.githubusercontent.com/acme/dd-plugins/main/catalog/plugins.json" in saved_registry
    assert "cache_ttl_seconds" not in saved_registry


def test_add_catalog_normalizes_github_tree_url_with_subdirectory(tmp_path: Path) -> None:
    repository = PluginCatalogRepository(tmp_path)

    catalog = repository.add_catalog("research", "https://github.com/acme/dd-plugins/tree/main/custom-catalog")

    assert catalog.url == "https://raw.githubusercontent.com/acme/dd-plugins/main/custom-catalog/catalog/plugins.json"


def test_add_catalog_normalizes_github_tree_url_ending_with_catalog(tmp_path: Path) -> None:
    repository = PluginCatalogRepository(tmp_path)

    catalog = repository.add_catalog("research", "https://github.com/acme/dd-plugins/tree/main/catalog")

    assert catalog.url == "https://raw.githubusercontent.com/acme/dd-plugins/main/catalog/plugins.json"


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


def test_normalize_local_catalog_directory_ending_with_catalog(tmp_path: Path) -> None:
    normalized = normalize_catalog_location(str(tmp_path / "plugins" / "catalog"))

    assert normalized == str((tmp_path / "plugins" / "catalog" / "plugins.json").resolve(strict=False))


def test_load_invalid_catalog_registry_raises_user_facing_error(tmp_path: Path) -> None:
    repository = PluginCatalogRepository(tmp_path)
    repository.config_file.write_text("catalogs:\n- alias: research\n")

    with pytest.raises(PluginCatalogError, match="Failed to load plugin catalog registry"):
        repository.load()


def test_add_catalog_does_not_replace_invalid_catalog_registry(tmp_path: Path) -> None:
    repository = PluginCatalogRepository(tmp_path)
    saved_registry = "catalogs:\n- alias: research\n"
    repository.config_file.write_text(saved_registry)

    with pytest.raises(PluginCatalogError, match="Failed to load plugin catalog registry"):
        repository.add_catalog("local", "https://github.com/acme/dd-plugins")

    assert repository.config_file.read_text() == saved_registry


def test_load_catalog_uses_cache_when_source_is_unavailable(tmp_path: Path) -> None:
    catalog_path = _write_catalog(tmp_path)
    repository = PluginCatalogRepository(tmp_path)
    repository.add_catalog("local", str(catalog_path))

    first_catalog = repository.load_catalog("local")
    catalog_path.unlink()
    cached_catalog = repository.load_catalog("local")

    assert first_catalog.plugins[0].name == "text-transform"
    assert cached_catalog.plugins[0].name == "text-transform"


def test_load_catalog_falls_back_to_stale_cache_when_refresh_fetch_fails(tmp_path: Path) -> None:
    catalog_path = _write_catalog(tmp_path, plugin_name="cached-transform")
    repository = PluginCatalogRepository(tmp_path)
    repository.add_catalog("local", str(catalog_path), cache_ttl_seconds=0)

    repository.load_catalog("local")
    catalog_path.unlink()
    cached_catalog = repository.load_catalog("local")

    assert cached_catalog.plugins[0].name == "cached-transform"


def test_load_catalog_does_not_fall_back_to_stale_cache_when_fresh_catalog_is_invalid(tmp_path: Path) -> None:
    catalog_path = _write_catalog(tmp_path, plugin_name="cached-transform")
    repository = PluginCatalogRepository(tmp_path)
    repository.add_catalog("local", str(catalog_path), cache_ttl_seconds=0)

    repository.load_catalog("local")
    catalog_path.write_text(json.dumps(_catalog_payload(schema_version=999, plugin_name="invalid-transform")))

    with pytest.raises(PluginCatalogError, match="unsupported catalog schema_version"):
        repository.load_catalog("local")


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


@patch("data_designer.cli.repositories.plugin_catalog_repository.urlopen")
def test_load_catalog_reports_remote_http_error(mock_urlopen: Mock, tmp_path: Path) -> None:
    mock_urlopen.side_effect = HTTPError(
        "https://example.test/catalog/plugins.json",
        404,
        "Not Found",
        {},
        None,
    )
    repository = PluginCatalogRepository(tmp_path)
    repository.add_catalog("remote", "https://example.test/catalog/plugins.json")

    with pytest.raises(PluginCatalogError, match="HTTP 404"):
        repository.load_catalog("remote", refresh=True)


@patch("data_designer.cli.repositories.plugin_catalog_repository.urlopen")
def test_load_catalog_rejects_oversized_remote_catalog(mock_urlopen: Mock, tmp_path: Path) -> None:
    mock_urlopen.return_value = _RemoteResponse(b"{" + (b" " * MAX_PLUGIN_CATALOG_SIZE_BYTES) + b"}")
    repository = PluginCatalogRepository(tmp_path)
    repository.add_catalog("remote", "https://example.test/catalog/plugins.json")

    with pytest.raises(PluginCatalogError, match="exceeds maximum size"):
        repository.load_catalog("remote", refresh=True)


@patch("data_designer.cli.repositories.plugin_catalog_repository.urlopen")
def test_load_catalog_reports_remote_json_decode_error(mock_urlopen: Mock, tmp_path: Path) -> None:
    mock_urlopen.return_value = _RemoteResponse(b"{")
    repository = PluginCatalogRepository(tmp_path)
    repository.add_catalog("remote", "https://example.test/catalog/plugins.json")

    with pytest.raises(PluginCatalogError, match="Failed to parse plugin catalog JSON"):
        repository.load_catalog("remote", refresh=True)


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


def test_consumer_accepts_upstream_valid_catalog_fixture(tmp_path: Path) -> None:
    repository = PluginCatalogRepository(tmp_path)
    repository.add_catalog("upstream", str(UPSTREAM_CATALOG_FIXTURES_DIR / "catalog-valid.json"))

    catalog = repository.load_catalog("upstream", refresh=True)

    assert [package.name for package in catalog.packages] == [
        "data-designer-compatible-column",
        "data-designer-git-seed-reader",
        "data-designer-url-processor",
        "data-designer-python312-column",
        "data-designer-future-dd-processor",
        "data-designer-multi-plugin-package",
    ]
    assert [entry.name for entry in catalog.plugins][-2:] == ["multi-seed-reader", "multi-processor"]


def test_consumer_rejects_upstream_invalid_install_fixture(tmp_path: Path) -> None:
    repository = PluginCatalogRepository(tmp_path)
    repository.add_catalog("upstream", str(UPSTREAM_CATALOG_FIXTURES_DIR / "catalog-invalid-install.json"))

    with pytest.raises(PluginCatalogError, match="expected a requirement for 'data-designer-invalid-install'"):
        repository.load_catalog("upstream", refresh=True)


def test_consumer_rejects_upstream_unsupported_version_fixture(tmp_path: Path) -> None:
    repository = PluginCatalogRepository(tmp_path)
    repository.add_catalog("upstream", str(UPSTREAM_CATALOG_FIXTURES_DIR / "catalog-unsupported-version.json"))

    with pytest.raises(PluginCatalogError, match="unsupported catalog schema_version 999"):
        repository.load_catalog("upstream", refresh=True)


def test_catalog_model_requires_contract_required_package_metadata() -> None:
    package = _package_entry()
    package.pop("compatibility")

    with pytest.raises(ValidationError, match="compatibility"):
        PluginCatalog.model_validate(_catalog_payload(packages=[package]))


def test_catalog_model_requires_non_empty_runtime_plugins() -> None:
    package = _package_entry(plugins=[])

    with pytest.raises(ValidationError, match="plugins"):
        PluginCatalog.model_validate(_catalog_payload(packages=[package]))


def test_fetches_production_catalog_when_enabled(tmp_path: Path) -> None:
    if os.getenv("DATA_DESIGNER_TEST_REMOTE_PLUGIN_CATALOG") != "1":
        pytest.skip("Set DATA_DESIGNER_TEST_REMOTE_PLUGIN_CATALOG=1 to run the live catalog smoke test")

    catalog = PluginCatalogRepository(tmp_path).load_catalog(refresh=True)

    assert catalog.packages


def test_load_catalog_accepts_equivalent_data_designer_marker_quoting(tmp_path: Path) -> None:
    package = _package_entry()
    package["compatibility"]["data_designer"] = {
        "requirement": "data-designer>=0.5.7; python_version < '3.12'",
        "specifier": ">=0.5.7",
        "marker": "python_version < '3.12'",
    }
    catalog_path = _write_catalog(tmp_path, packages=[package])
    repository = PluginCatalogRepository(tmp_path)
    repository.add_catalog("local", str(catalog_path))

    catalog = repository.load_catalog("local", refresh=True)

    assert catalog.plugins[0].compatibility is not None
    assert catalog.plugins[0].compatibility.data_designer is not None
    assert catalog.plugins[0].compatibility.data_designer.marker == "python_version < '3.12'"


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


def test_load_catalog_rejects_null_schema_v2_install_index_url(tmp_path: Path) -> None:
    catalog_path = _write_catalog(
        tmp_path,
        packages=[
            _package_entry(
                package_name="data-designer-invalid-index",
                plugins=[_runtime_plugin("invalid-index")],
                install={
                    "requirement": "data-designer-invalid-index",
                    "index_url": None,
                },
            )
        ],
    )
    repository = PluginCatalogRepository(tmp_path)
    repository.add_catalog("local", str(catalog_path))

    with pytest.raises(PluginCatalogError, match="install.index_url.*expected a non-empty string"):
        repository.load_catalog("local", refresh=True)


def test_load_catalog_rejects_empty_schema_v2_install_index_url(tmp_path: Path) -> None:
    catalog_path = _write_catalog(
        tmp_path,
        packages=[
            _package_entry(
                package_name="data-designer-empty-index",
                plugins=[_runtime_plugin("empty-index")],
                install={
                    "requirement": "data-designer-empty-index",
                    "index_url": "",
                },
            )
        ],
    )
    repository = PluginCatalogRepository(tmp_path)
    repository.add_catalog("local", str(catalog_path))

    with pytest.raises(PluginCatalogError, match="install.index_url.*expected a non-empty string"):
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


def test_load_catalog_rejects_duplicate_canonical_package_names(tmp_path: Path) -> None:
    catalog_path = _write_catalog(
        tmp_path,
        packages=[
            _package_entry(
                package_name="data-designer-foo",
                plugins=[_runtime_plugin("first-plugin")],
            ),
            _package_entry(
                package_name="data_designer_foo",
                plugins=[_runtime_plugin("second-plugin")],
            ),
        ],
    )
    repository = PluginCatalogRepository(tmp_path)
    repository.add_catalog("local", str(catalog_path))

    with pytest.raises(PluginCatalogError, match="duplicate package name"):
        repository.load_catalog("local", refresh=True)


class _RemoteResponse:
    def __init__(self, content: bytes, *, status: int = 200) -> None:
        self._content = content
        self.status = status

    def __enter__(self) -> "_RemoteResponse":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        _ = size
        return self._content


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
