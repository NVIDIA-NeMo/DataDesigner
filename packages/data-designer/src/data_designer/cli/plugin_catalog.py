# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from urllib.parse import urlparse

from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.utils import InvalidName, canonicalize_name
from packaging.version import InvalidVersion, Version
from pydantic import BaseModel, ConfigDict, Field

from data_designer.plugins.plugin import PluginType

DEFAULT_PLUGIN_TAP_ALIAS = "nvidia"
DEFAULT_PLUGIN_TAP_URL = "https://raw.githubusercontent.com/NVIDIA-NeMo/DataDesignerPlugins/main/catalog/plugins.json"
DEFAULT_PLUGIN_TAP_URL_ENV_VAR = "DATA_DESIGNER_DEFAULT_PLUGIN_TAP_URL"
PLUGIN_TAPS_FILE_NAME = "plugin_taps.yaml"
PLUGIN_TAP_CACHE_DIR_NAME = "plugin-tap-cache"
PLUGIN_TAP_DEFAULT_CACHE_TTL_SECONDS = 24 * 60 * 60
MAX_PLUGIN_CATALOG_SIZE_BYTES = 1 * 1024 * 1024
PLUGIN_CATALOG_SCHEMA_VERSION = 2
SUPPORTED_PLUGIN_CATALOG_SCHEMA_VERSIONS = {PLUGIN_CATALOG_SCHEMA_VERSION}
PLUGIN_TAP_ALIAS_PATTERN = r"^[A-Za-z0-9_.-]+$"
DATA_DESIGNER_DISTRIBUTION_NAME = "data-designer"
PLUGIN_ENTRY_POINT_GROUP = "data_designer.plugins"
CATALOG_DOCUMENT_KEYS = {"plugins", "schema_version"}
CATALOG_PLUGIN_KEYS = {
    "compatibility",
    "description",
    "docs",
    "entry_point",
    "name",
    "package",
    "plugin_type",
    "source",
}
CATALOG_PACKAGE_KEYS = {"name", "path", "version"}
CATALOG_ENTRY_POINT_KEYS = {"group", "name", "value"}
CATALOG_COMPATIBILITY_KEYS = {"data_designer", "python"}
CATALOG_PYTHON_COMPATIBILITY_KEYS = {"specifier"}
CATALOG_DATA_DESIGNER_COMPATIBILITY_KEYS = {"marker", "requirement", "specifier"}
CATALOG_DOCS_KEYS = {"url"}
SUPPORTED_PLUGIN_TYPE_VALUES = {plugin_type.value for plugin_type in PluginType}
PACKAGE_PATH_ROOT = "plugins"
PACKAGE_PATH_SEGMENT_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


class PluginCatalogError(ValueError):
    """Raised when a plugin catalog cannot be loaded or validated."""


class PluginCompatibilityTarget(BaseModel):
    """Version requirement for one environment target."""

    model_config = ConfigDict(extra="allow")

    requirement: str | None = None
    specifier: str | None = None
    marker: str | None = None


class PluginCompatibility(BaseModel):
    """Compatibility requirements declared by a catalog entry."""

    model_config = ConfigDict(extra="allow")

    python: PluginCompatibilityTarget | None = None
    data_designer: PluginCompatibilityTarget | None = None


class PluginPackageInfo(BaseModel):
    """Python package metadata for a catalog entry."""

    model_config = ConfigDict(extra="allow")

    name: str
    version: str | None = None
    path: str | None = None


class PluginEntryPointInfo(BaseModel):
    """Runtime entry point exposed by an installable plugin package."""

    model_config = ConfigDict(extra="allow")

    group: str = "data_designer.plugins"
    name: str
    value: str


class PluginSourceInfo(BaseModel):
    """Install source metadata for a catalog entry."""

    model_config = ConfigDict(extra="allow")

    type: str
    package: str | None = None
    url: str | None = None
    ref: str | None = None
    path: str | None = None
    subdirectory: str | None = None
    editable: bool | None = None


class PluginDocsInfo(BaseModel):
    """Documentation metadata for a catalog entry."""

    model_config = ConfigDict(extra="allow")

    url: str | None = None


class PluginCatalogEntry(BaseModel):
    """One discoverable Data Designer plugin entry from a tap catalog."""

    model_config = ConfigDict(extra="allow")

    name: str
    plugin_type: PluginType
    description: str = ""
    package: PluginPackageInfo
    entry_point: PluginEntryPointInfo
    compatibility: PluginCompatibility | None = None
    source: PluginSourceInfo | None = None
    docs: PluginDocsInfo | None = None


class PluginCatalog(BaseModel):
    """Versioned plugin tap catalog."""

    model_config = ConfigDict(extra="allow")

    schema_version: int
    plugins: list[PluginCatalogEntry] = Field(default_factory=list)


class PluginTapConfig(BaseModel):
    """Persisted tap configuration."""

    alias: str = Field(pattern=PLUGIN_TAP_ALIAS_PATTERN)
    url: str
    trusted: bool = False
    cache_ttl_seconds: int = Field(default=PLUGIN_TAP_DEFAULT_CACHE_TTL_SECONDS, ge=0)


class PluginTapRegistry(BaseModel):
    """Persisted collection of user-configured plugin taps."""

    taps: list[PluginTapConfig] = Field(default_factory=list)


@dataclass(frozen=True)
class CompatibilityResult:
    """Compatibility result for one catalog entry in the local environment."""

    is_compatible: bool
    reasons: list[str]


@dataclass(frozen=True)
class InstallPlan:
    """Resolved package-manager command for installing one plugin entry."""

    plugin_name: str
    package_name: str
    source_description: str
    command: list[str]
    manager: str
    tap_alias: str
    trusted_tap: bool


@dataclass(frozen=True)
class InstalledPluginInfo:
    """Installed plugin entry point discovered without importing plugin code."""

    name: str
    entry_point_value: str


def get_default_plugin_tap_url() -> str:
    """Return the built-in plugin tap URL, honoring a local override for QA/staging."""
    return os.getenv(DEFAULT_PLUGIN_TAP_URL_ENV_VAR, DEFAULT_PLUGIN_TAP_URL)


def validate_plugin_catalog_payload(payload: object, *, source: str) -> None:
    """Validate a decoded plugin tap catalog against the schema v2 contract."""
    try:
        _validate_plugin_catalog_payload(payload)
    except PluginCatalogError as e:
        raise PluginCatalogError(f"Invalid plugin catalog at {source!r}: {e}") from e


def _validate_plugin_catalog_payload(payload: object) -> None:
    catalog = _required_catalog_object("catalog document", payload, CATALOG_DOCUMENT_KEYS)
    schema_version = catalog["schema_version"]
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != PLUGIN_CATALOG_SCHEMA_VERSION
    ):
        raise PluginCatalogError(
            f"unsupported catalog schema_version {schema_version!r}; expected {PLUGIN_CATALOG_SCHEMA_VERSION}"
        )

    plugins = catalog["plugins"]
    if not isinstance(plugins, list):
        raise PluginCatalogError("catalog document has invalid plugins; expected a list")

    runtime_names: dict[str, tuple[str, str]] = {}
    for index, raw_plugin in enumerate(plugins):
        package_name, plugin_name, entry_point_name = _validate_catalog_plugin(raw_plugin, index)
        previous = runtime_names.get(entry_point_name)
        if previous is not None:
            previous_package, previous_plugin_name = previous
            raise PluginCatalogError(
                f"duplicate runtime plugin name {entry_point_name!r} from "
                f"catalog plugin {previous_plugin_name!r} in package {previous_package!r} and "
                f"catalog plugin {plugin_name!r} in package {package_name!r}"
            )
        runtime_names[entry_point_name] = (package_name, plugin_name)


def _validate_catalog_plugin(raw_plugin: object, index: int) -> tuple[str, str, str]:
    context = f"catalog plugins[{index}]"
    plugin = _required_catalog_object(context, raw_plugin, CATALOG_PLUGIN_KEYS)
    package = _required_catalog_object(f"{context}.package", plugin["package"], CATALOG_PACKAGE_KEYS)
    entry_point = _required_catalog_object(
        f"{context}.entry_point",
        plugin["entry_point"],
        CATALOG_ENTRY_POINT_KEYS,
    )
    compatibility = _required_catalog_object(
        f"{context}.compatibility",
        plugin["compatibility"],
        CATALOG_COMPATIBILITY_KEYS,
    )
    python_compatibility = _required_catalog_object(
        f"{context}.compatibility.python",
        compatibility["python"],
        CATALOG_PYTHON_COMPATIBILITY_KEYS,
    )
    data_designer_compatibility = _required_catalog_object(
        f"{context}.compatibility.data_designer",
        compatibility["data_designer"],
        CATALOG_DATA_DESIGNER_COMPATIBILITY_KEYS,
    )
    source = _required_catalog_object(f"{context}.source", plugin["source"])
    docs = _required_catalog_object(f"{context}.docs", plugin["docs"], CATALOG_DOCS_KEYS)

    package_name = _catalog_package_name(f"{context}.package.name", package["name"])
    _catalog_version(package_name, f"{context}.package.version", package["version"])
    _validate_package_path(package_name, _required_catalog_string(f"{context}.package.path", package["path"]))

    plugin_type = _required_catalog_string(f"{context}.plugin_type", plugin["plugin_type"])
    if plugin_type not in SUPPORTED_PLUGIN_TYPE_VALUES:
        raise PluginCatalogError(
            f"{context}.plugin_type {plugin_type!r} is invalid; expected one of "
            f"{_format_catalog_choices(SUPPORTED_PLUGIN_TYPE_VALUES)}"
        )

    plugin_name = _required_catalog_string(f"{context}.name", plugin["name"])
    entry_point_group = _required_catalog_string(f"{context}.entry_point.group", entry_point["group"])
    if entry_point_group != PLUGIN_ENTRY_POINT_GROUP:
        raise PluginCatalogError(
            f"{context}.entry_point.group {entry_point_group!r} is invalid; expected {PLUGIN_ENTRY_POINT_GROUP!r}"
        )
    entry_point_name = _required_catalog_string(f"{context}.entry_point.name", entry_point["name"])
    _required_catalog_string(f"{context}.entry_point.value", entry_point["value"])
    _required_catalog_string(f"{context}.description", plugin["description"])
    _catalog_version_specifier(
        package_name,
        f"{context}.compatibility.python.specifier",
        python_compatibility["specifier"],
    )
    _catalog_data_designer_compatibility(
        package_name,
        f"{context}.compatibility.data_designer",
        data_designer_compatibility,
    )
    _validate_source_metadata(package_name, source)
    _catalog_http_url(f"{context}.docs.url", docs["url"])
    return package_name, plugin_name, entry_point_name


def _required_catalog_object(
    context: str,
    value: object,
    expected_keys: set[str] | None = None,
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise PluginCatalogError(f"{context} is invalid; expected an object")
    if expected_keys is not None:
        _validate_catalog_object_keys(context, value, expected_keys)
    return value


def _validate_catalog_object_keys(context: str, value: dict[str, object], expected_keys: set[str]) -> None:
    keys = set(value)
    if keys != expected_keys:
        raise PluginCatalogError(
            f"{context} has invalid fields; expected {{{_format_catalog_keys(expected_keys)}}}, "
            f"got {{{_format_catalog_keys(keys)}}}"
        )


def _required_catalog_string(context: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise PluginCatalogError(f"{context} is invalid; expected a non-empty string")
    return value


def _required_catalog_nullable_string(context: str, value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    raise PluginCatalogError(f"{context} is invalid; expected a string or null")


def _catalog_package_name(context: str, value: object) -> str:
    package_name = _required_catalog_string(context, value)
    try:
        canonicalize_name(package_name, validate=True)
    except InvalidName as e:
        raise PluginCatalogError(f"{context} {package_name!r} is invalid; expected a valid package name") from e
    return package_name


def _catalog_version(package_name: str, context: str, value: object) -> str:
    raw_version = _required_catalog_string(context, value)
    try:
        Version(raw_version)
    except InvalidVersion as e:
        raise PluginCatalogError(f"package {package_name!r} has invalid {context} {raw_version!r}: {e}") from e
    return raw_version


def _catalog_version_specifier(package_name: str, context: str, value: object) -> str:
    raw_specifier = _required_catalog_string(context, value)
    try:
        specifier = SpecifierSet(raw_specifier)
    except InvalidSpecifier as e:
        raise PluginCatalogError(f"package {package_name!r} has invalid {context} {raw_specifier!r}: {e}") from e
    if not str(specifier):
        raise PluginCatalogError(f"package {package_name!r} has invalid {context}; expected at least one specifier")
    return str(specifier)


def _catalog_data_designer_compatibility(
    package_name: str,
    context: str,
    compatibility: dict[str, object],
) -> None:
    requirement_text = _required_catalog_string(f"{context}.requirement", compatibility["requirement"])
    try:
        requirement = Requirement(requirement_text)
    except InvalidRequirement as e:
        raise PluginCatalogError(
            f"package {package_name!r} has invalid {context}.requirement {requirement_text!r}: {e}"
        ) from e
    if canonicalize_name(requirement.name) != DATA_DESIGNER_DISTRIBUTION_NAME:
        raise PluginCatalogError(
            f"package {package_name!r} has invalid {context}.requirement {requirement_text!r}; "
            f"expected a {DATA_DESIGNER_DISTRIBUTION_NAME!r} requirement"
        )
    if not requirement.specifier:
        raise PluginCatalogError(f"package {package_name!r} has invalid {context}.requirement; expected a specifier")

    specifier = _catalog_version_specifier(package_name, f"{context}.specifier", compatibility["specifier"])
    if specifier != str(requirement.specifier):
        raise PluginCatalogError(
            f"package {package_name!r} has invalid {context}.specifier {specifier!r}; "
            f"expected {str(requirement.specifier)!r} from requirement"
        )

    marker = _required_catalog_nullable_string(f"{context}.marker", compatibility["marker"])
    expected_marker = str(requirement.marker) if requirement.marker is not None else None
    if marker != expected_marker:
        raise PluginCatalogError(
            f"package {package_name!r} has invalid {context}.marker {marker!r}; expected {expected_marker!r}"
        )


def _validate_source_metadata(package_name: str, source: dict[str, object]) -> None:
    source_type = source.get("type")
    if source_type == "pypi":
        _validate_pypi_source_metadata(package_name, source)
        return
    if source_type == "git":
        _validate_git_source_metadata(package_name, source)
        return
    if source_type == "path":
        _validate_path_source_metadata(package_name, source)
        return
    raise PluginCatalogError(
        f"package {package_name!r} has invalid source.type {source_type!r}; expected one of 'pypi', 'git', or 'path'"
    )


def _validate_pypi_source_metadata(package_name: str, source: dict[str, object]) -> None:
    _validate_source_keys(package_name, source, "pypi", {"type", "package"})
    source_package = _required_source_string(package_name, source, "pypi", "package")
    if source_package != package_name:
        raise PluginCatalogError(
            f"package {package_name!r} has invalid pypi source package {source_package!r}; "
            "expected the source package to match package.name"
        )


def _validate_git_source_metadata(package_name: str, source: dict[str, object]) -> None:
    _validate_source_keys(package_name, source, "git", {"type", "url", "ref", "subdirectory"})
    url = _required_source_string(package_name, source, "git", "url")
    _required_source_string(package_name, source, "git", "ref")
    subdirectory = _required_source_string(package_name, source, "git", "subdirectory")
    _validate_package_path(package_name, subdirectory)
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise PluginCatalogError(
            f"package {package_name!r} has invalid git source url {url!r}; expected an absolute HTTP(S) URL"
        )


def _validate_path_source_metadata(package_name: str, source: dict[str, object]) -> None:
    _validate_source_keys(package_name, source, "path", {"type", "path", "editable"})
    path = _required_source_string(package_name, source, "path", "path")
    _validate_package_path(package_name, path)
    editable = source.get("editable")
    if not isinstance(editable, bool):
        raise PluginCatalogError(f"package {package_name!r} has invalid path source field 'editable'; expected a bool")


def _validate_source_keys(
    package_name: str,
    source: dict[str, object],
    source_type: str,
    expected_keys: set[str],
) -> None:
    keys = set(source)
    if keys != expected_keys:
        raise PluginCatalogError(
            f"package {package_name!r} has invalid {source_type!r} source fields; "
            f"expected {{{_format_catalog_keys(expected_keys)}}}, got {{{_format_catalog_keys(keys)}}}"
        )


def _required_source_string(package_name: str, source: dict[str, object], source_type: str, key: str) -> str:
    value = source.get(key)
    if not isinstance(value, str) or not value:
        raise PluginCatalogError(
            f"package {package_name!r} has invalid {source_type!r} source field {key!r}; expected a non-empty string"
        )
    return value


def _validate_package_path(package_name: str, value: str) -> None:
    parts = value.split("/")
    if (
        "\\" in value
        or value.startswith("/")
        or len(parts) < 2
        or parts[0] != PACKAGE_PATH_ROOT
        or any(part in {"", ".", ".."} for part in parts)
        or not all(PACKAGE_PATH_SEGMENT_PATTERN.fullmatch(part) for part in parts[1:])
    ):
        raise PluginCatalogError(
            f"package {package_name!r} has invalid package path {value!r}; "
            f"expected a normalized repository-relative path under {PACKAGE_PATH_ROOT!r}"
        )


def _catalog_http_url(context: str, value: object) -> str:
    url = _required_catalog_string(context, value)
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise PluginCatalogError(f"{context} {url!r} is invalid; expected an absolute HTTP(S) URL")
    return url


def _format_catalog_keys(keys: set[str]) -> str:
    return ", ".join(sorted(keys))


def _format_catalog_choices(choices: set[str]) -> str:
    return ", ".join(repr(choice) for choice in sorted(choices))
