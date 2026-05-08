# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from pydantic import ValidationError

from data_designer.cli.plugin_catalog import (
    DEFAULT_PLUGIN_CATALOG_ALIAS,
    DEFAULT_PLUGIN_CATALOG_URL,
    MAX_PLUGIN_CATALOG_SIZE_BYTES,
    PLUGIN_CATALOG_CACHE_DIR_NAME,
    PLUGIN_CATALOG_DEFAULT_CACHE_TTL_SECONDS,
    PLUGIN_CATALOGS_FILE_NAME,
    PluginCatalog,
    PluginCatalogConfig,
    PluginCatalogError,
    PluginCatalogRegistry,
    get_default_plugin_catalog_url,
    validate_plugin_catalog_payload,
)
from data_designer.cli.repositories.base import ConfigRepository
from data_designer.config.errors import InvalidConfigError, InvalidFileFormatError, InvalidFilePathError
from data_designer.config.utils.io_helpers import load_config_file, save_config_file


class PluginCatalogRepository(ConfigRepository[PluginCatalogRegistry]):
    """Repository for plugin catalog aliases and cached catalog payloads."""

    @property
    def config_file(self) -> Path:
        """Get the plugin catalog configuration file path."""
        return self.config_dir / PLUGIN_CATALOGS_FILE_NAME

    @property
    def cache_dir(self) -> Path:
        """Get the plugin catalog cache directory path."""
        return self.config_dir / PLUGIN_CATALOG_CACHE_DIR_NAME

    def load(self) -> PluginCatalogRegistry | None:
        """Load user-configured plugin catalogs."""
        if not self.exists():
            return None

        try:
            config_dict = load_config_file(self.config_file)
            return PluginCatalogRegistry.model_validate(config_dict)
        except (InvalidConfigError, InvalidFileFormatError, InvalidFilePathError, OSError, ValidationError) as e:
            raise PluginCatalogError(f"Failed to load plugin catalog registry at {self.config_file}: {e}") from e

    def save(self, config: PluginCatalogRegistry) -> None:
        """Save user-configured plugin catalogs."""
        config_dict = config.model_dump(mode="json", exclude_none=True)
        save_config_file(self.config_file, config_dict)

    def list_catalogs(self) -> list[PluginCatalogConfig]:
        """Return the built-in NVIDIA catalog followed by user-configured catalogs."""
        catalogs = [self.default_catalog()]
        registry = self.load()
        if registry is not None:
            catalogs.extend(sorted(registry.catalogs, key=lambda catalog: catalog.alias.casefold()))
        return catalogs

    def get_catalog(self, alias: str | None = None) -> PluginCatalogConfig | None:
        """Return a catalog by alias, defaulting to the built-in NVIDIA catalog."""
        resolved_alias = alias or DEFAULT_PLUGIN_CATALOG_ALIAS
        return next((catalog for catalog in self.list_catalogs() if _same_alias(catalog.alias, resolved_alias)), None)

    def add_catalog(
        self,
        alias: str,
        url: str,
        *,
        trusted: bool = False,
        cache_ttl_seconds: int = PLUGIN_CATALOG_DEFAULT_CACHE_TTL_SECONDS,
    ) -> PluginCatalogConfig:
        """Persist a new catalog alias.

        Raises:
            ValueError: If the alias already exists or is reserved for the built-in catalog.
        """
        if self.get_catalog(alias) is not None:
            raise ValueError(f"Plugin catalog alias {alias!r} already exists")

        catalog = PluginCatalogConfig(
            alias=alias,
            url=normalize_catalog_location(url),
            trusted=trusted,
            cache_ttl_seconds=cache_ttl_seconds,
        )
        registry = self.load() or PluginCatalogRegistry()
        registry.catalogs.append(catalog)
        registry.catalogs = sorted(registry.catalogs, key=lambda item: item.alias.casefold())
        self.save(registry)
        return catalog

    def remove_catalog(self, alias: str) -> None:
        """Remove a user-configured catalog alias.

        Raises:
            ValueError: If the alias is reserved or does not exist.
        """
        if _same_alias(alias, DEFAULT_PLUGIN_CATALOG_ALIAS):
            raise ValueError(f"Cannot remove the built-in {DEFAULT_PLUGIN_CATALOG_ALIAS!r} plugin catalog")

        registry = self.load()
        matching_catalog = (
            next((catalog for catalog in registry.catalogs if _same_alias(catalog.alias, alias)), None)
            if registry
            else None
        )
        if registry is None or matching_catalog is None:
            raise ValueError(f"Plugin catalog alias {alias!r} not found")

        registry.catalogs = [catalog for catalog in registry.catalogs if not _same_alias(catalog.alias, alias)]
        if registry.catalogs:
            self.save(registry)
        else:
            self.delete()

        self._remove_cache_files(matching_catalog)

    def load_catalog(self, alias: str | None = None, *, refresh: bool = False) -> PluginCatalog:
        """Load a catalog from cache or source."""
        catalog_config = self.get_catalog(alias)
        if catalog_config is None:
            raise ValueError(f"Plugin catalog alias {alias!r} not found")

        if not refresh:
            cached_catalog = self._load_cached_catalog(catalog_config, require_fresh=True)
            if cached_catalog is not None:
                return cached_catalog

        try:
            payload = self._fetch_catalog_payload(catalog_config.url)
            catalog = self._validate_catalog(payload, source=catalog_config.url)
        except (PluginCatalogError, OSError, ValueError):
            if not refresh:
                cached_catalog = self._load_cached_catalog(catalog_config, require_fresh=False)
                if cached_catalog is not None:
                    return cached_catalog
            raise

        self._save_catalog_cache(catalog_config, payload)
        return catalog

    def _load_cached_catalog(self, catalog: PluginCatalogConfig, *, require_fresh: bool) -> PluginCatalog | None:
        cache_file = self._cache_file(catalog)
        if not cache_file.exists():
            return None

        try:
            with open(cache_file) as f:
                cache_payload = json.load(f)
            fetched_at = datetime.fromisoformat(cache_payload["fetched_at"])
            if fetched_at.tzinfo is None:
                fetched_at = fetched_at.replace(tzinfo=timezone.utc)
            if require_fresh and catalog.cache_ttl_seconds == 0:
                return None
            if require_fresh:
                age_seconds = (datetime.now(timezone.utc) - fetched_at).total_seconds()
                if age_seconds > catalog.cache_ttl_seconds:
                    return None
            catalog_payload = cache_payload["catalog"]
            return self._validate_catalog(catalog_payload, source=str(cache_file))
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            return None

    def _save_catalog_cache(self, catalog: PluginCatalogConfig, catalog_payload: dict[str, object]) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_payload = {
            "catalog_alias": catalog.alias,
            "catalog_url": catalog.url,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "catalog": catalog_payload,
        }
        with open(self._cache_file(catalog), "w") as f:
            json.dump(cache_payload, f, indent=2, sort_keys=True)

    def _cache_file(self, catalog: PluginCatalogConfig) -> Path:
        url_hash = hashlib.sha256(catalog.url.encode("utf-8")).hexdigest()[:12]
        return self.cache_dir / f"{catalog.alias}-{url_hash}.json"

    def _remove_cache_files(self, catalog: PluginCatalogConfig) -> None:
        if not self.cache_dir.exists():
            return

        self._cache_file(catalog).unlink(missing_ok=True)
        legacy_cache_file = self.cache_dir / f"{catalog.alias}.json"
        legacy_cache_file.unlink(missing_ok=True)

        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file) as f:
                    cache_payload = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            cached_alias = cache_payload.get("catalog_alias")
            if isinstance(cached_alias, str) and _same_alias(cached_alias, catalog.alias):
                cache_file.unlink(missing_ok=True)

    @staticmethod
    def _fetch_catalog_payload(location: str) -> dict[str, object]:
        if _is_http_url(location):
            return _fetch_remote_catalog(location)
        return _fetch_local_catalog(location)

    @staticmethod
    def _validate_catalog(payload: dict, *, source: str) -> PluginCatalog:
        validate_plugin_catalog_payload(payload, source=source)
        try:
            catalog = PluginCatalog.model_validate(payload)
        except ValidationError as e:
            raise PluginCatalogError(f"Invalid plugin catalog at {source!r}: {e}") from e
        return catalog

    @staticmethod
    def default_catalog() -> PluginCatalogConfig:
        """Return the built-in NVIDIA plugin catalog configuration."""
        catalog_url = get_default_plugin_catalog_url()
        return PluginCatalogConfig(
            alias=DEFAULT_PLUGIN_CATALOG_ALIAS,
            url=catalog_url,
            trusted=catalog_url == DEFAULT_PLUGIN_CATALOG_URL,
            cache_ttl_seconds=PLUGIN_CATALOG_DEFAULT_CACHE_TTL_SECONDS,
        )


def normalize_catalog_location(location: str) -> str:
    """Normalize a catalog repository, catalog URL, or local path to a catalog location."""
    if _is_http_url(location):
        return _normalize_catalog_url(location)

    path = Path(location).expanduser()
    if path.suffix.lower() == ".json":
        return str(path.resolve(strict=False))
    return str(_catalog_plugins_path(path).resolve(strict=False))


def _same_alias(left: str, right: str) -> bool:
    return left.casefold() == right.casefold()


def _normalize_catalog_url(url: str) -> str:
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    segments = [segment for segment in parsed.path.split("/") if segment]

    if hostname in {"github.com", "www.github.com"} and len(segments) >= 2:
        owner, repo = segments[0], segments[1]
        if len(segments) == 2:
            return f"https://raw.githubusercontent.com/{owner}/{repo}/main/catalog/plugins.json"
        if len(segments) >= 5 and segments[2] == "blob":
            ref = segments[3]
            path = "/".join(segments[4:])
            return f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{path}"
        if len(segments) >= 4 and segments[2] == "tree":
            ref = segments[3]
            catalog_root = "/".join(segments[4:])
            catalog_path = _catalog_plugins_url_path(catalog_root)
            return f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{catalog_path}"

    return url


def _catalog_plugins_path(path: Path) -> Path:
    if path.name == "catalog":
        return path / "plugins.json"
    return path / "catalog" / "plugins.json"


def _catalog_plugins_url_path(catalog_root: str) -> str:
    if not catalog_root:
        return "catalog/plugins.json"
    if catalog_root.rstrip("/").endswith("/catalog") or catalog_root == "catalog":
        return f"{catalog_root}/plugins.json"
    return f"{catalog_root}/catalog/plugins.json"


def _fetch_local_catalog(location: str) -> dict[str, object]:
    path = Path(location).expanduser()
    if not path.exists():
        raise PluginCatalogError(f"Plugin catalog file not found: {path}")
    if path.stat().st_size > MAX_PLUGIN_CATALOG_SIZE_BYTES:
        raise PluginCatalogError(
            f"Plugin catalog at {path} exceeds maximum size of {MAX_PLUGIN_CATALOG_SIZE_BYTES} bytes"
        )

    try:
        with open(path) as f:
            payload = json.load(f)
    except json.JSONDecodeError as e:
        raise PluginCatalogError(f"Failed to parse plugin catalog JSON at {path}: {e}") from e

    if not isinstance(payload, dict):
        raise PluginCatalogError(f"Plugin catalog at {path} must be a JSON object")
    return payload


def _fetch_remote_catalog(url: str) -> dict[str, object]:
    request = Request(url, headers={"User-Agent": "data-designer"})
    try:
        with urlopen(request, timeout=10) as response:
            status = getattr(response, "status", 200)
            if isinstance(status, int) and status >= 400:
                raise PluginCatalogError(f"Failed to fetch plugin catalog {url!r}: HTTP {status}")
            content = response.read(MAX_PLUGIN_CATALOG_SIZE_BYTES + 1)
    except HTTPError as e:
        raise PluginCatalogError(f"Failed to fetch plugin catalog {url!r}: HTTP {e.code}") from e
    except URLError as e:
        raise PluginCatalogError(f"Failed to fetch plugin catalog {url!r}: {e.reason}") from e

    if len(content) > MAX_PLUGIN_CATALOG_SIZE_BYTES:
        raise PluginCatalogError(
            f"Plugin catalog at {url!r} exceeds maximum size of {MAX_PLUGIN_CATALOG_SIZE_BYTES} bytes"
        )

    try:
        payload = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise PluginCatalogError(f"Failed to parse plugin catalog JSON at {url!r}: {e}") from e

    if not isinstance(payload, dict):
        raise PluginCatalogError(f"Plugin catalog at {url!r} must be a JSON object")
    return payload


def _is_http_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
