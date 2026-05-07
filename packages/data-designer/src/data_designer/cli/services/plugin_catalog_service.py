# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.metadata
import platform
from collections import defaultdict
from collections.abc import Iterable

from packaging.markers import InvalidMarker, Marker
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

from data_designer.cli.plugin_catalog import (
    DEFAULT_PLUGIN_TAP_ALIAS,
    CompatibilityResult,
    InstalledPluginInfo,
    PluginCatalogEntry,
    PluginCompatibilityTarget,
    PluginTapConfig,
)
from data_designer.cli.repositories.plugin_tap_repository import PluginTapRepository
from data_designer.plugins.plugin import PluginType
from data_designer.plugins.registry import PluginRegistry


class PluginCatalogService:
    """Business logic for plugin catalog discovery and compatibility checks."""

    def __init__(
        self,
        repository: PluginTapRepository,
        *,
        python_version: str | None = None,
        data_designer_version: str | None = None,
    ) -> None:
        self.repository = repository
        self.python_version = python_version or platform.python_version()
        self.data_designer_version = data_designer_version or _get_installed_data_designer_version()

    def list_entries(
        self,
        tap_alias: str | None = None,
        *,
        refresh: bool = False,
        include_incompatible: bool = False,
    ) -> list[PluginCatalogEntry]:
        """List catalog entries for a tap, filtering incompatible entries by default."""
        catalog = self.repository.load_catalog(tap_alias, refresh=refresh)
        entries = sorted(catalog.plugins, key=lambda entry: (entry.name, entry.package.version or ""))
        if include_incompatible:
            return entries
        return [entry for entry in entries if self.evaluate_compatibility(entry).is_compatible]

    def search_entries(
        self,
        query: str,
        tap_alias: str | None = None,
        *,
        refresh: bool = False,
        include_incompatible: bool = False,
    ) -> list[PluginCatalogEntry]:
        """Search catalog entries by simple token matching."""
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        return [
            entry
            for entry in self.list_entries(
                tap_alias,
                refresh=refresh,
                include_incompatible=include_incompatible,
            )
            if all(token in _entry_search_text(entry) for token in query_tokens)
        ]

    def get_entry(
        self,
        name: str,
        tap_alias: str | None = None,
        *,
        refresh: bool = False,
        include_incompatible: bool = True,
    ) -> PluginCatalogEntry:
        """Return the newest catalog entry by plugin name."""
        entries = self.list_entries(tap_alias, refresh=refresh, include_incompatible=True)
        matches = [entry for entry in entries if entry.name == name]
        matched_incompatible = False
        if matches and not include_incompatible:
            compatible_matches = [entry for entry in matches if self.evaluate_compatibility(entry).is_compatible]
            matched_incompatible = bool(matches) and not compatible_matches
            matches = compatible_matches
        if matches:
            return max(matches, key=_entry_version_sort_key)

        resolved_alias = tap_alias or DEFAULT_PLUGIN_TAP_ALIAS
        if matched_incompatible:
            raise ValueError(
                f"Plugin {name!r} was found in tap {resolved_alias!r}, but no compatible version is available"
            )
        raise ValueError(f"Plugin {name!r} was not found in tap {resolved_alias!r}")

    @staticmethod
    def group_entries_by_package(entries: Iterable[PluginCatalogEntry]) -> dict[str, list[PluginCatalogEntry]]:
        """Group catalog entries by installable package name."""
        grouped_entries: dict[str, list[PluginCatalogEntry]] = defaultdict(list)
        for entry in entries:
            grouped_entries[entry.package.name].append(entry)
        return {
            package_name: sorted(items, key=lambda item: item.name) for package_name, items in grouped_entries.items()
        }

    def evaluate_compatibility(self, entry: PluginCatalogEntry) -> CompatibilityResult:
        """Evaluate whether a catalog entry is compatible with the local environment."""
        compatibility = entry.compatibility
        if compatibility is None:
            return CompatibilityResult(is_compatible=True, reasons=[])

        reasons = []
        reasons.extend(
            self._evaluate_target(
                target=compatibility.python,
                label="Python",
                version=self.python_version,
                marker_environment={"python_version": _major_minor(self.python_version)},
            )
        )
        reasons.extend(
            self._evaluate_target(
                target=compatibility.data_designer,
                label="Data Designer",
                version=self.data_designer_version,
                marker_environment={"python_version": _major_minor(self.python_version)},
            )
        )
        return CompatibilityResult(is_compatible=not reasons, reasons=reasons)

    def list_taps(self) -> list[PluginTapConfig]:
        """List available plugin taps."""
        return self.repository.list_taps()

    def get_tap(self, alias: str | None = None) -> PluginTapConfig:
        """Return a plugin tap or raise a user-facing error."""
        tap = self.repository.get_tap(alias)
        if tap is None:
            raise ValueError(f"Plugin tap alias {alias!r} not found")
        return tap

    def add_tap(
        self,
        alias: str,
        url: str,
        *,
        trusted: bool,
        cache_ttl_seconds: int,
    ) -> PluginTapConfig:
        """Add a plugin tap alias."""
        return self.repository.add_tap(
            alias,
            url,
            trusted=trusted,
            cache_ttl_seconds=cache_ttl_seconds,
        )

    def remove_tap(self, alias: str) -> None:
        """Remove a plugin tap alias."""
        self.repository.remove_tap(alias)

    def list_installed_plugins(self) -> list[InstalledPluginInfo]:
        """List runtime plugins currently discoverable through entry points."""
        registry = PluginRegistry()
        installed_plugins = []
        for plugin_type in PluginType:
            for plugin in registry.get_plugins(plugin_type):
                installed_plugins.append(
                    InstalledPluginInfo(
                        name=plugin.name,
                        plugin_type=plugin.plugin_type,
                        config_qualified_name=plugin.config_qualified_name,
                        impl_qualified_name=plugin.impl_qualified_name,
                    )
                )
        return sorted(installed_plugins, key=lambda plugin: (plugin.plugin_type.value, plugin.name))

    def _evaluate_target(
        self,
        *,
        target: PluginCompatibilityTarget | None,
        label: str,
        version: str | None,
        marker_environment: dict[str, str],
    ) -> list[str]:
        if target is None or not target.specifier:
            return []

        marker_error = _marker_error(target.marker, marker_environment)
        if marker_error is not None:
            return [f"{label} marker {target.marker!r} is invalid: {marker_error}"]
        if target.marker and not Marker(target.marker).evaluate(marker_environment):
            return []

        if version is None:
            return [f"Unable to resolve installed {label} version for constraint {target.specifier!r}"]

        try:
            specifier = SpecifierSet(target.specifier)
        except InvalidSpecifier as e:
            return [f"{label} specifier {target.specifier!r} is invalid: {e}"]

        try:
            parsed_version = Version(version)
        except InvalidVersion as e:
            return [f"Installed {label} version {version!r} is invalid: {e}"]

        if not specifier.contains(parsed_version, prereleases=True):
            return [f"{label} {version} does not satisfy {target.specifier}"]
        return []


def _get_installed_data_designer_version() -> str | None:
    try:
        return importlib.metadata.version("data-designer")
    except importlib.metadata.PackageNotFoundError:
        return None


def _tokenize(value: str) -> list[str]:
    return [token.strip().lower() for token in value.split() if token.strip()]


def _entry_search_text(entry: PluginCatalogEntry) -> str:
    values = [
        entry.name,
        entry.plugin_type.value,
        entry.description,
        entry.package.name,
        entry.package.version or "",
        entry.package.path or "",
        entry.entry_point.name,
        entry.entry_point.value,
        entry.source.type if entry.source is not None else "",
        entry.source.package if entry.source is not None and entry.source.package else "",
        entry.source.url if entry.source is not None and entry.source.url else "",
        entry.docs.url if entry.docs is not None and entry.docs.url else "",
    ]
    return " ".join(values).lower()


def _entry_version_sort_key(entry: PluginCatalogEntry) -> Version:
    try:
        return Version(entry.package.version or "0")
    except InvalidVersion:
        return Version("0")


def _major_minor(version: str) -> str:
    parts = version.split(".")
    if len(parts) < 2:
        return version
    return ".".join(parts[:2])


def _marker_error(marker: str | None, environment: dict[str, str]) -> str | None:
    if marker is None:
        return None
    try:
        Marker(marker).evaluate(environment)
    except InvalidMarker as e:
        return str(e)
    return None
