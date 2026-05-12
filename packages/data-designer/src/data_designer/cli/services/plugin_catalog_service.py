# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.metadata
import platform
import re
from collections import defaultdict
from collections.abc import Iterable
from html.parser import HTMLParser
from pathlib import PurePosixPath
from urllib.error import URLError
from urllib.parse import unquote, urljoin, urlparse
from urllib.request import Request, urlopen

from packaging.markers import InvalidMarker, Marker
from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.utils import (
    InvalidSdistFilename,
    InvalidWheelFilename,
    canonicalize_name,
    parse_sdist_filename,
    parse_wheel_filename,
)
from packaging.version import InvalidVersion, Version

from data_designer.cli.plugin_catalog import (
    DATA_DESIGNER_PLUGIN_PACKAGE_PREFIX,
    PLUGIN_ENTRY_POINT_GROUP,
    CompatibilityResult,
    InstalledPluginInfo,
    PluginCatalogConfig,
    PluginCatalogEntry,
    PluginCompatibilityTarget,
)
from data_designer.cli.repositories.plugin_catalog_repository import PluginCatalogRepository


class PluginCatalogService:
    """Business logic for plugin catalog discovery and compatibility checks."""

    def __init__(
        self,
        repository: PluginCatalogRepository,
        *,
        python_version: str | None = None,
        data_designer_version: str | None = None,
    ) -> None:
        self.repository = repository
        self.python_version = python_version or platform.python_version()
        self.data_designer_version = data_designer_version or _get_installed_data_designer_version()
        self._package_current_version_cache: dict[tuple[str, str, str, str], str | None] = {}

    def list_entries(
        self,
        catalog_alias: str | None = None,
        *,
        refresh: bool = False,
        include_incompatible: bool = False,
    ) -> list[PluginCatalogEntry]:
        """List catalog entries for a catalog, filtering incompatible entries by default."""
        catalog = self.repository.load_catalog(catalog_alias, refresh=refresh)
        entries = sorted(catalog.entries, key=lambda entry: (canonicalize_name(entry.package.name), entry.name))
        if include_incompatible:
            return entries
        return [entry for entry in entries if self.evaluate_compatibility(entry).is_compatible]

    def search_entries(
        self,
        query: str,
        catalog_alias: str | None = None,
        *,
        refresh: bool = False,
        include_incompatible: bool = False,
    ) -> list[PluginCatalogEntry]:
        """Search catalog entries by package metadata and runtime plugin metadata."""
        query_terms = _query_term_groups(query)
        if not query_terms:
            return []

        return [
            entry
            for entry in self.list_entries(
                catalog_alias,
                refresh=refresh,
                include_incompatible=include_incompatible,
            )
            if _matches_all_query_terms(entry, query_terms)
        ]

    def suggest_entries(
        self,
        query: str,
        catalog_alias: str | None = None,
        *,
        refresh: bool = False,
        include_incompatible: bool = False,
        limit: int = 3,
    ) -> list[PluginCatalogEntry]:
        """Suggest nearby catalog entries when a strict search has no matches."""
        query_terms = _query_term_groups(query)
        if not query_terms or limit < 1:
            return []

        scored_entries: list[tuple[int, str, str, PluginCatalogEntry]] = []
        for entry in self.list_entries(
            catalog_alias,
            refresh=refresh,
            include_incompatible=include_incompatible,
        ):
            search_text = _entry_search_text(entry)
            score = sum(1 for term_group in query_terms if _term_group_matches(term_group, search_text))
            if score:
                scored_entries.append((score, canonicalize_name(entry.package.name), entry.name, entry))

        scored_entries.sort(key=lambda item: (-item[0], item[1], item[2]))
        return [item[3] for item in scored_entries[:limit]]

    def get_package_entries(
        self,
        package: str,
        catalog_alias: str | None = None,
        *,
        refresh: bool = False,
        include_incompatible: bool = True,
    ) -> list[PluginCatalogEntry]:
        """Return all runtime plugin entries declared by one catalog package name or package alias."""
        entries = self.list_entries(
            catalog_alias,
            refresh=refresh,
            include_incompatible=include_incompatible,
        )
        canonical_package = canonicalize_name(package)
        exact_matches = [entry for entry in entries if canonicalize_name(entry.package.name) == canonical_package]
        if exact_matches:
            return exact_matches

        return [
            entry for entry in entries if _package_alias(canonicalize_name(entry.package.name)) == canonical_package
        ]

    def get_runtime_plugin_entries(
        self,
        runtime_plugin_name: str,
        catalog_alias: str | None = None,
        *,
        refresh: bool = False,
        include_incompatible: bool = True,
    ) -> list[PluginCatalogEntry]:
        """Return catalog entries whose runtime plugin name exactly matches a user-provided name."""
        entries = self.list_entries(
            catalog_alias,
            refresh=refresh,
            include_incompatible=include_incompatible,
        )
        canonical_runtime_plugin_name = canonicalize_name(runtime_plugin_name)
        return [entry for entry in entries if canonicalize_name(entry.name) == canonical_runtime_plugin_name]

    @staticmethod
    def group_entries_by_package(entries: Iterable[PluginCatalogEntry]) -> dict[str, list[PluginCatalogEntry]]:
        """Group catalog entries by installable package name."""
        grouped_entries: dict[str, list[PluginCatalogEntry]] = defaultdict(list)
        for entry in entries:
            grouped_entries[canonicalize_name(entry.package.name)].append(entry)
        return {
            package_name: sorted(items, key=lambda item: item.name) for package_name, items in grouped_entries.items()
        }

    def get_package_current_version(self, entry: PluginCatalogEntry, *, requirement: str | None = None) -> str | None:
        """Return the catalog package version selected by the entry metadata or package index."""
        resolved_requirement = requirement or entry.install.requirement
        cache_key = (
            canonicalize_name(entry.package.name),
            entry.package.version or "",
            resolved_requirement,
            entry.install.index_url or "",
        )
        if cache_key not in self._package_current_version_cache:
            self._package_current_version_cache[cache_key] = _resolve_package_current_version(
                entry,
                requirement=resolved_requirement,
            )
        return self._package_current_version_cache[cache_key]

    def evaluate_compatibility(self, entry: PluginCatalogEntry) -> CompatibilityResult:
        """Evaluate whether a catalog entry is compatible with the local environment."""
        compatibility = entry.compatibility
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

    def list_catalogs(self) -> list[PluginCatalogConfig]:
        """List available plugin catalogs."""
        return self.repository.list_catalogs()

    def get_catalog(self, alias: str | None = None) -> PluginCatalogConfig:
        """Return a plugin catalog or raise a user-facing error."""
        catalog = self.repository.get_catalog(alias)
        if catalog is None:
            raise ValueError(f"Plugin catalog alias {alias!r} not found")
        return catalog

    def add_catalog(
        self,
        alias: str,
        url: str,
    ) -> PluginCatalogConfig:
        """Add a plugin catalog alias."""
        return self.repository.add_catalog(
            alias,
            url,
        )

    def remove_catalog(self, alias: str) -> None:
        """Remove a plugin catalog alias."""
        self.repository.remove_catalog(alias)

    def list_installed_plugins(self) -> list[InstalledPluginInfo]:
        """List installed Data Designer runtime plugins without importing plugin modules."""
        entry_points = importlib.metadata.entry_points(group=PLUGIN_ENTRY_POINT_GROUP)
        installed_plugins = []
        for entry_point in entry_points:
            package_name, package_version = _entry_point_distribution_metadata(entry_point)
            installed_plugins.append(
                InstalledPluginInfo(
                    name=entry_point.name,
                    entry_point_value=entry_point.value,
                    package_name=package_name,
                    package_version=package_version,
                )
            )
        return sorted(installed_plugins, key=lambda plugin: (plugin.name, plugin.package_name or ""))

    def _evaluate_target(
        self,
        *,
        target: PluginCompatibilityTarget,
        label: str,
        version: str | None,
        marker_environment: dict[str, str],
    ) -> list[str]:
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


def _resolve_package_current_version(entry: PluginCatalogEntry, *, requirement: str) -> str | None:
    if entry.package.version is not None:
        return entry.package.version

    requirement_version = _package_version_from_requirement(requirement, entry.package.name)
    if requirement_version is not None:
        return requirement_version

    if entry.install.index_url is None:
        return None
    return _latest_simple_index_version(entry.install.index_url, entry.package.name)


def _package_version_from_requirement(requirement_text: str, package_name: str) -> str | None:
    try:
        requirement = Requirement(requirement_text)
    except InvalidRequirement:
        return None

    if canonicalize_name(requirement.name) != canonicalize_name(package_name):
        return None

    if requirement.url is not None:
        return _package_version_from_distribution_url(requirement.url, package_name)

    parsed_versions = []
    for specifier in requirement.specifier:
        if specifier.operator != "==" or "*" in specifier.version:
            continue
        try:
            parsed_versions.append(Version(specifier.version))
        except InvalidVersion:
            return None
    if len(parsed_versions) == 1:
        return str(parsed_versions[0])
    return None


def _package_version_from_distribution_url(url: str, package_name: str) -> str | None:
    filename = PurePosixPath(unquote(urlparse(url).path)).name
    version = _package_version_from_distribution_filename(filename, package_name)
    if version is None:
        return None
    return str(version)


def _latest_simple_index_version(index_url: str, package_name: str) -> str | None:
    project_url = urljoin(f"{index_url.rstrip('/')}/", f"{canonicalize_name(package_name)}/")
    request = Request(
        project_url,
        headers={"Accept": "text/html, application/vnd.pypi.simple.v1+html"},
    )
    try:
        with urlopen(request, timeout=SIMPLE_INDEX_VERSION_FETCH_TIMEOUT_SECONDS) as response:
            content = response.read(SIMPLE_INDEX_MAX_BYTES + 1)
    except (URLError, OSError, ValueError):
        return None

    if len(content) > SIMPLE_INDEX_MAX_BYTES:
        return None

    parser = _SimpleIndexPackageFileParser()
    parser.feed(content.decode("utf-8", errors="replace"))
    versions = [
        version
        for filename in parser.filenames
        if (version := _package_version_from_distribution_filename(filename, package_name)) is not None
    ]
    if not versions:
        return None
    return str(max(versions))


def _package_version_from_distribution_filename(filename: str, package_name: str) -> Version | None:
    try:
        if filename.endswith(".whl"):
            name, version, _build, _tags = parse_wheel_filename(filename)
            parsed_name = canonicalize_name(name)
        elif filename.endswith((".tar.gz", ".zip")):
            name, version = parse_sdist_filename(filename)
            parsed_name = canonicalize_name(name)
        else:
            return None
    except (InvalidSdistFilename, InvalidVersion, InvalidWheelFilename):
        return None

    if parsed_name != canonicalize_name(package_name):
        return None
    return version


class _SimpleIndexPackageFileParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.filenames: list[str] = []
        self._inside_link = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        self._inside_link = True
        for name, value in attrs:
            if name.lower() == "href" and value:
                self._add_candidate(value)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a":
            self._inside_link = False

    def handle_data(self, data: str) -> None:
        if self._inside_link:
            self._add_candidate(data)

    def _add_candidate(self, value: str) -> None:
        path = urlparse(value.strip()).path
        filename = PurePosixPath(unquote(path)).name
        if filename:
            self.filenames.append(filename)


SEARCH_SYNONYMS = {
    "chunk": ("chunker", "chunking", "document chunker"),
    "dedup": ("dedupe", "deduplicate", "deduplication"),
    "dedupe": ("dedup", "deduplicate", "deduplication"),
    "import": ("ingest", "load", "reader", "seed reader"),
    "ingest": ("import", "load", "reader", "seed reader"),
    "load": ("import", "ingest", "reader", "seed reader"),
    "reader": ("import", "ingest", "load"),
    "repo": ("repository", "repositories"),
    "repos": ("repository", "repositories"),
    "repositories": ("repo", "repository"),
    "repository": ("repo", "repos"),
}

SEARCH_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_.-]+")
SIMPLE_INDEX_VERSION_FETCH_TIMEOUT_SECONDS = 3
SIMPLE_INDEX_MAX_BYTES = 512 * 1024


def _tokenize(value: str) -> list[str]:
    return [token.lower() for token in SEARCH_TOKEN_PATTERN.findall(value)]


def _query_term_groups(query: str) -> list[set[str]]:
    return [_expand_query_token(token) for token in _tokenize(query)]


def _expand_query_token(token: str) -> set[str]:
    normalized_token = _normalize_search_value(token)
    terms = {token, normalized_token}
    for synonym in SEARCH_SYNONYMS.get(normalized_token, ()):
        terms.add(synonym)
        terms.add(_normalize_search_value(synonym))
    return {term for term in terms if term}


def _matches_all_query_terms(entry: PluginCatalogEntry, query_terms: list[set[str]]) -> bool:
    search_text = _entry_search_text(entry)
    return all(_term_group_matches(term_group, search_text) for term_group in query_terms)


def _term_group_matches(term_group: set[str], search_text: str) -> bool:
    return any(term in search_text for term in term_group)


def _entry_search_text(entry: PluginCatalogEntry) -> str:
    package_name = canonicalize_name(entry.package.name)
    values = [
        entry.package.name,
        _package_alias(package_name) or "",
        entry.description,
        entry.name,
        entry.plugin_type.value,
    ]
    raw_text = " ".join(values).lower()
    return f"{raw_text} {_normalize_search_value(raw_text)}"


def _normalize_search_value(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.lower()).split())


def _package_alias(canonical_package_name: str) -> str | None:
    if not canonical_package_name.startswith(DATA_DESIGNER_PLUGIN_PACKAGE_PREFIX):
        return None
    return canonical_package_name.removeprefix(DATA_DESIGNER_PLUGIN_PACKAGE_PREFIX)


def _entry_point_distribution_metadata(entry_point: importlib.metadata.EntryPoint) -> tuple[str | None, str | None]:
    distribution = getattr(entry_point, "dist", None)
    if distribution is None:
        return None, None

    package_name = None
    metadata = getattr(distribution, "metadata", None)
    if hasattr(metadata, "get"):
        name_value = metadata.get("Name")
        if isinstance(name_value, str):
            package_name = name_value
    if package_name is None:
        name_value = getattr(distribution, "name", None)
        if isinstance(name_value, str):
            package_name = name_value

    version_value = getattr(distribution, "version", None)
    package_version = version_value if isinstance(version_value, str) else None
    return package_name, package_version


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
