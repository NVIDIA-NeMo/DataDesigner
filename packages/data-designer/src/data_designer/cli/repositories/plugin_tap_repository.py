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
    DEFAULT_PLUGIN_TAP_ALIAS,
    MAX_PLUGIN_CATALOG_SIZE_BYTES,
    PLUGIN_TAP_CACHE_DIR_NAME,
    PLUGIN_TAP_DEFAULT_CACHE_TTL_SECONDS,
    PLUGIN_TAPS_FILE_NAME,
    PluginCatalog,
    PluginCatalogError,
    PluginTapConfig,
    PluginTapRegistry,
    get_default_plugin_tap_url,
    validate_plugin_catalog_payload,
)
from data_designer.cli.repositories.base import ConfigRepository
from data_designer.config.utils.io_helpers import load_config_file, save_config_file


class PluginTapRepository(ConfigRepository[PluginTapRegistry]):
    """Repository for plugin tap aliases and cached catalog payloads."""

    @property
    def config_file(self) -> Path:
        """Get the plugin tap configuration file path."""
        return self.config_dir / PLUGIN_TAPS_FILE_NAME

    @property
    def cache_dir(self) -> Path:
        """Get the plugin tap cache directory path."""
        return self.config_dir / PLUGIN_TAP_CACHE_DIR_NAME

    def load(self) -> PluginTapRegistry | None:
        """Load user-configured plugin taps."""
        if not self.exists():
            return None

        try:
            config_dict = load_config_file(self.config_file)
            return PluginTapRegistry.model_validate(config_dict)
        except Exception:
            return None

    def save(self, config: PluginTapRegistry) -> None:
        """Save user-configured plugin taps."""
        config_dict = config.model_dump(mode="json", exclude_none=True)
        save_config_file(self.config_file, config_dict)

    def list_taps(self) -> list[PluginTapConfig]:
        """Return the built-in NVIDIA tap followed by user-configured taps."""
        taps = [self.default_tap()]
        registry = self.load()
        if registry is not None:
            taps.extend(sorted(registry.taps, key=lambda tap: tap.alias.casefold()))
        return taps

    def get_tap(self, alias: str | None = None) -> PluginTapConfig | None:
        """Return a tap by alias, defaulting to the built-in NVIDIA tap."""
        resolved_alias = alias or DEFAULT_PLUGIN_TAP_ALIAS
        return next((tap for tap in self.list_taps() if _same_alias(tap.alias, resolved_alias)), None)

    def add_tap(
        self,
        alias: str,
        url: str,
        *,
        trusted: bool = False,
        cache_ttl_seconds: int = PLUGIN_TAP_DEFAULT_CACHE_TTL_SECONDS,
    ) -> PluginTapConfig:
        """Persist a new tap alias.

        Raises:
            ValueError: If the alias already exists or is reserved for the built-in tap.
        """
        if self.get_tap(alias) is not None:
            raise ValueError(f"Plugin tap alias {alias!r} already exists")

        tap = PluginTapConfig(
            alias=alias,
            url=normalize_tap_location(url),
            trusted=trusted,
            cache_ttl_seconds=cache_ttl_seconds,
        )
        registry = self.load() or PluginTapRegistry()
        registry.taps.append(tap)
        registry.taps = sorted(registry.taps, key=lambda item: item.alias.casefold())
        self.save(registry)
        return tap

    def remove_tap(self, alias: str) -> None:
        """Remove a user-configured tap alias.

        Raises:
            ValueError: If the alias is reserved or does not exist.
        """
        if _same_alias(alias, DEFAULT_PLUGIN_TAP_ALIAS):
            raise ValueError(f"Cannot remove the built-in {DEFAULT_PLUGIN_TAP_ALIAS!r} plugin tap")

        registry = self.load()
        matching_tap = next((tap for tap in registry.taps if _same_alias(tap.alias, alias)), None) if registry else None
        if registry is None or matching_tap is None:
            raise ValueError(f"Plugin tap alias {alias!r} not found")

        registry.taps = [tap for tap in registry.taps if not _same_alias(tap.alias, alias)]
        if registry.taps:
            self.save(registry)
        else:
            self.delete()

        self._remove_cache_files(matching_tap)

    def load_catalog(self, alias: str | None = None, *, refresh: bool = False) -> PluginCatalog:
        """Load a tap catalog from cache or source."""
        tap = self.get_tap(alias)
        if tap is None:
            raise ValueError(f"Plugin tap alias {alias!r} not found")

        if not refresh:
            cached_catalog = self._load_cached_catalog(tap, require_fresh=True)
            if cached_catalog is not None:
                return cached_catalog

        try:
            payload = self._fetch_catalog_payload(tap.url)
            catalog = self._validate_catalog(payload, source=tap.url)
        except Exception:
            if not refresh:
                cached_catalog = self._load_cached_catalog(tap, require_fresh=False)
                if cached_catalog is not None:
                    return cached_catalog
            raise

        self._save_catalog_cache(tap, payload)
        return catalog

    def _load_cached_catalog(self, tap: PluginTapConfig, *, require_fresh: bool) -> PluginCatalog | None:
        cache_file = self._cache_file(tap)
        if not cache_file.exists():
            return None

        try:
            with open(cache_file) as f:
                cache_payload = json.load(f)
            fetched_at = datetime.fromisoformat(cache_payload["fetched_at"])
            if fetched_at.tzinfo is None:
                fetched_at = fetched_at.replace(tzinfo=timezone.utc)
            if require_fresh and tap.cache_ttl_seconds == 0:
                return None
            if require_fresh:
                age_seconds = (datetime.now(timezone.utc) - fetched_at).total_seconds()
                if age_seconds > tap.cache_ttl_seconds:
                    return None
            catalog_payload = cache_payload["catalog"]
            return self._validate_catalog(catalog_payload, source=str(cache_file))
        except Exception:
            return None

    def _save_catalog_cache(self, tap: PluginTapConfig, catalog_payload: dict[str, object]) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_payload = {
            "tap_alias": tap.alias,
            "tap_url": tap.url,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "catalog": catalog_payload,
        }
        with open(self._cache_file(tap), "w") as f:
            json.dump(cache_payload, f, indent=2, sort_keys=True)

    def _cache_file(self, tap: PluginTapConfig) -> Path:
        url_hash = hashlib.sha256(tap.url.encode("utf-8")).hexdigest()[:12]
        return self.cache_dir / f"{tap.alias}-{url_hash}.json"

    def _remove_cache_files(self, tap: PluginTapConfig) -> None:
        if not self.cache_dir.exists():
            return

        self._cache_file(tap).unlink(missing_ok=True)
        legacy_cache_file = self.cache_dir / f"{tap.alias}.json"
        legacy_cache_file.unlink(missing_ok=True)

        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file) as f:
                    cache_payload = json.load(f)
            except Exception:
                continue
            cached_alias = cache_payload.get("tap_alias")
            if isinstance(cached_alias, str) and _same_alias(cached_alias, tap.alias):
                cache_file.unlink(missing_ok=True)

    @staticmethod
    def _fetch_catalog_payload(location: str) -> dict:
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
    def default_tap() -> PluginTapConfig:
        """Return the built-in NVIDIA plugin tap configuration."""
        return PluginTapConfig(
            alias=DEFAULT_PLUGIN_TAP_ALIAS,
            url=get_default_plugin_tap_url(),
            trusted=True,
            cache_ttl_seconds=PLUGIN_TAP_DEFAULT_CACHE_TTL_SECONDS,
        )


def normalize_tap_location(location: str) -> str:
    """Normalize a tap repository, catalog URL, or local path to a catalog location."""
    if _is_http_url(location):
        return _normalize_tap_url(location)

    path = Path(location).expanduser()
    if path.suffix.lower() == ".json":
        return str(path.resolve(strict=False))
    return str((path / "catalog" / "plugins.json").resolve(strict=False))


def _same_alias(left: str, right: str) -> bool:
    return left.casefold() == right.casefold()


def _normalize_tap_url(url: str) -> str:
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
            return f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/catalog/plugins.json"

    return url


def _fetch_local_catalog(location: str) -> dict:
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


def _fetch_remote_catalog(url: str) -> dict:
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
